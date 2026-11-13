#!/usr/bin/env python3
"""Four-step agent for CUAD records: extracts the answers and builds a confidence for each in stages.

Used by 4_extract_example.py for every CUAD record. The same code, with explanations and tables, is in
notebooks/cuad_agent_confidence.ipynb.

    from agent_confidence import agent_confidence
    row = agent_confidence(rec)          # rec = one CUAD record from data/unified/*.jsonl (gold is never read)
    # row = {"doc_id": ..., "fields": {clause: {"value": ..., "confidence": ...}}}

Step 1  is the answer present?                          (all 41 clauses, one call)
Step 2  where is it? quote the passages                 (clauses judged present, one call)
Step 3  which quarter of the contract is it in?         (one call; checked against the quote's real position)
Step 4  semantic answer or data (date, name, amount)?   (one call; for data, the value's occurrences are counted)
Final confidence = geometric mean of the four step confidences, after the code checks.

Needs LLM_BASE_URL, LLM_API_KEY and TEXT_LLM_MODEL (or LLM_MODEL). Standard library only.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
REGIONS = ["0-25%", "25-50%", "50-75%", "75-100%"]
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}      # running totals for this process
_lock = threading.Lock()
_contracts = None


def model() -> str | None:
    return os.environ.get("TEXT_LLM_MODEL") or os.environ.get("LLM_MODEL")


# ---------- helpers ----------

def attach_contract(rec: dict) -> dict:
    """Adds the contract text ("context") and the clause descriptions ("questions") to a CUAD record.
    Reads `context` and `question` from the raw file named in rec["pages"]; never the answers."""
    global _contracts
    with _lock:
        if _contracts is None:
            raw = json.loads((DATA / rec["pages"][0]).read_text(encoding="utf-8"))["data"]
            _contracts = {d["title"]: d["paragraphs"][0] for d in raw}
    para = _contracts[rec["doc_id"].split(":", 1)[1]]
    questions = {qa["id"].rsplit("__", 1)[1]: qa["question"].split("Details:", 1)[-1].strip() for qa in para["qas"]}
    return {**rec, "context": para["context"], "questions": {n: questions.get(n, "") for n in rec["schema"]["properties"]}}


def ask(prompt: str, tries: int = 12) -> dict:
    """One LLM call; returns the JSON object in the reply. Waits and retries on rate limits and server errors."""
    body = json.dumps({"model": model(), "max_tokens": 16000, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    error = "no attempt made"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data, cost = json.load(r), r.headers.get("x-litellm-response-cost")
            usage = data.get("usage") or {}
            with _lock:
                USAGE["calls"] += 1
                USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
                USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
                USAGE["cost_usd"] += float(cost) if cost not in (None, "", "None") else 0.0
            reply = data["choices"][0]["message"]["content"] or ""
            out = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
            return {str(k).strip().strip("[]").strip(): v for k, v in out.items()}
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            if e.code not in (408, 409, 429) and e.code < 500:
                raise RuntimeError(error) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = f"{type(e).__name__}: {e}"
        except ValueError:
            error = "the reply was not JSON"
        time.sleep(min(60, 10 + 5 * attempt) + random.random() * 5)
    raise RuntimeError(error)


def squash(text: str):
    """Whitespace runs collapsed to one space, lower-cased, plus the original index of every kept character."""
    out, pos, in_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space and out:
                out.append(" "); pos.append(i)
            in_space = True
        else:
            out.append(ch.casefold()[:1] or ch); pos.append(i); in_space = False
    return "".join(out), pos


def locate(span: str, squashed: str, pos: list) -> tuple[int, int] | None:
    """(start, end) character positions of the span in the contract, or None if it is not there."""
    needle = squash(span)[0].strip()
    at = squashed.find(needle) if needle else -1
    return None if at < 0 else (pos[at], pos[at + len(needle) - 1] + 1)


def region_of(start: int, length: int) -> str:
    return REGIONS[min(int(4 * start / max(length, 1)), 3)]


def conf(value) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


# ---------- the four steps ----------

STEP1 = """You review a contract. For each clause below, decide only ONE thing: is there anything in the contract that answers it?
Do not quote anything yet.

Reply with ONE JSON object and nothing else:
{{"<clause name>": {{"present": true or false, "confidence": <0..1>}}, ...}}

- "present": true if the contract contains a passage a careful lawyer would highlight for this clause. Most clauses are absent from most contracts.
- "confidence": how sure you are of that yes or no, from 0 to 1.
- Include all {n} clauses.

Clauses (name: what to look for):
{clauses}

--- CONTRACT ---
{contract}
"""

STEP2 = """You review a contract. An earlier pass decided that the clauses below ARE present in the contract.
For each one, show WHERE: quote the passage or passages that answer it.

Reply with ONE JSON object and nothing else:
{{"<clause name>": {{"spans": ["<exact quote>", ...], "confidence": <0..1>}}, ...}}

- Copy the words of every span exactly as they are in the contract. Never paraphrase or summarise.
- A passage that runs over several lines is ONE span. Keep it whole and write line breaks and runs of spaces as a single space.
  Never cut a sentence at a line break, and never return half a sentence.
- Only passages from different places in the contract are separate spans.
- For Document Name, Parties and dates, quote only the name or the date. For other clauses, quote the whole sentence or clause.
- If, on a closer look, the contract has nothing for a clause, return {{"spans": [], "confidence": 0}}.
- "confidence": how sure you are that these are exactly the passages a careful lawyer would highlight, from 0 to 1.

Clauses (name: what to look for):
{clauses}

--- CONTRACT ---
{contract}
"""

STEP3 = """Below is a contract cut into four equal parts, each marked with a header such as <<< REGION 25-50% >>>.
For each clause you get the passage(s) that were quoted as its answer. Say in which region each passage sits.

Reply with ONE JSON object and nothing else:
{{"<clause name>": {{"regions": ["0-25%" | "25-50%" | "50-75%" | "75-100%", ...], "confidence": <0..1>}}, ...}}

- "regions": one entry per quoted passage, in the same order as the passages are listed. Use the region where the passage starts.
- "confidence": how sure you are of the regions you gave, from 0 to 1. Lower it if you could not find a passage.

Clauses and their quoted passages:
{items}

--- CONTRACT, IN FOUR REGIONS ---
{contract}
"""

STEP4 = """For each clause below you get a question about a contract and the answer that was quoted from the contract.
Decide what KIND of answer the question needs.

- "data": the answer is a specific value that can be picked out and counted: a date, a name of a person or company,
  a place, an amount of money, a number, a duration, a document title.
- "semantic": the answer is a provision that has to be read and understood; it cannot be reduced to one value.

Reply with ONE JSON object and nothing else:
{{"<clause name>": {{"kind": "data" or "semantic",
                    "entity_type": "DATE | PERSON | ORG | GPE | MONEY | DURATION | QUANTITY | DOC | LAW | none",
                    "value": "<for data: the key value exactly as it is written in the contract, several values separated by ' | '; for semantic: empty>",
                    "occurrences": <for data: how many times that value occurs in the whole contract (for several values, the rarest one); for semantic: 0>,
                    "reasoning": "<one sentence: for semantic, why this passage answers the question; for data, why this value is the answer>",
                    "confidence": <0..1>}}, ...}}

- "confidence": how sure you are that the quoted answer correctly answers the question, given the kind of answer it is, from 0 to 1.
  For data, a value that occurs once and is clearly labelled deserves more confidence than one of several competing values.
  For semantic, be strict: is this the passage a lawyer would highlight, and is anything missing?

Clauses:
{items}

--- CONTRACT ---
{contract}
"""


def step1_is_present(rec: dict) -> dict:
    clauses = "\n".join(f"- {n}: {q}" for n, q in rec["questions"].items())
    reply = ask(STEP1.format(n=len(rec["questions"]), clauses=clauses, contract=rec["context"]))
    out = {}
    for name in rec["questions"]:
        a = reply.get(name) if isinstance(reply.get(name), dict) else {}
        out[name] = {"present": a.get("present") is True, "confidence": conf(a.get("confidence"))}
    return out


def step2_where(rec: dict, present: list) -> dict:
    if not present:
        return {}
    clauses = "\n".join(f"- {n}: {rec['questions'][n]}" for n in present)
    reply = ask(STEP2.format(clauses=clauses, contract=rec["context"]))
    out = {}
    for name in present:
        a = reply.get(name) if isinstance(reply.get(name), dict) else {}
        spans = a.get("spans") or []
        spans = [spans] if isinstance(spans, str) else [s for s in spans if isinstance(s, str) and s.strip()]
        out[name] = {"spans": list(dict.fromkeys(spans)), "confidence": conf(a.get("confidence")) if spans else 0.0}
    return out


def step3_region(rec: dict, where: dict) -> dict:
    todo = {n: w for n, w in where.items() if w["spans"]}
    if not todo:
        return {}
    text, q = rec["context"], len(rec["context"]) // 4
    parts = [text[:q], text[q: 2 * q], text[2 * q: 3 * q], text[3 * q:]]
    marked = "\n".join(f"\n<<< REGION {REGIONS[i]} >>>\n{part}" for i, part in enumerate(parts))
    items = "\n\n".join(f"[{n}]\n" + "\n".join(f"  passage {i}: {s}" for i, s in enumerate(w["spans"], 1)) for n, w in todo.items())
    reply = ask(STEP3.format(items=items, contract=marked))
    out = {}
    for name in todo:
        a = reply.get(name) if isinstance(reply.get(name), dict) else {}
        regions = a.get("regions") or []
        regions = [regions] if isinstance(regions, str) else [str(r) for r in regions]
        out[name] = {"regions": regions, "confidence": conf(a.get("confidence"))}
    return out


def step4_kind(rec: dict, where: dict) -> dict:
    todo = {n: w for n, w in where.items() if w["spans"]}
    if not todo:
        return {}
    items = "\n\n".join(f"[{n}]\nQuestion: {rec['questions'][n]}\nAnswer: {' | '.join(w['spans'])}" for n, w in todo.items())
    reply = ask(STEP4.format(items=items, contract=rec["context"]))
    out = {}
    for name in todo:
        a = reply.get(name) if isinstance(reply.get(name), dict) else {}
        kind = "data" if str(a.get("kind", "")).lower() == "data" else "semantic"
        out[name] = {"kind": kind, "value": str(a.get("value") or "") if kind == "data" else "", "confidence": conf(a.get("confidence"))}
    return out


# ---------- the function ----------

def agent_confidence(rec: dict, details: bool = False) -> dict:
    """Runs the four-step agent on one CUAD record and returns the prediction row in the scorer's format.
    With details=True each answered field also carries the four step confidences (c1..c4) and the answer kind."""
    if "context" not in rec:
        rec = attach_contract(rec)
    text = rec["context"]
    squashed, pos = squash(text)

    s1 = step1_is_present(rec)
    s2 = step2_where(rec, [n for n, v in s1.items() if v["present"]])
    with concurrent.futures.ThreadPoolExecutor(2) as pool:            # steps 3 and 4 both build on step 2 only
        f3, f4 = pool.submit(step3_region, rec, s2), pool.submit(step4_kind, rec, s2)
        s3, s4 = f3.result(), f4.result()

    fields = {}
    for name in rec["questions"]:
        spans = s2.get(name, {}).get("spans", [])
        found = [(s, hit) for s, hit in ((s, locate(s, squashed, pos)) for s in spans) if hit is not None]
        if not s1[name]["present"] or not found:
            fields[name] = {"value": None}
            continue
        c1 = s1[name]["confidence"]
        c2 = s2[name]["confidence"] * len(found) / len(spans)                      # share of quotes really in the contract
        actual = [region_of(start, len(text)) for _, (start, _) in found]
        claimed = s3.get(name, {}).get("regions", [])
        c3 = s3.get(name, {}).get("confidence", 0.0) if claimed and set(actual) <= set(claimed) else 0.2
        k = s4.get(name, {"kind": "semantic", "value": "", "confidence": 0.0})
        parts = [squash(v)[0].strip() for v in k["value"].split("|") if v.strip()] if k["kind"] == "data" else []
        c4 = 0.1 if parts and min(squashed.count(v) for v in parts) == 0 else k["confidence"]   # a named value must occur
        field = {"value": " | ".join(text[start:end] for _, (start, end) in found),
                 "confidence": round((c1 * c2 * c3 * c4) ** 0.25, 3)}
        if details:
            field.update(c1=round(c1, 3), c2=round(c2, 3), c3=round(c3, 3), c4=round(c4, 3), kind=k["kind"])
        fields[name] = field
    return {"doc_id": rec["doc_id"], "fields": fields}
