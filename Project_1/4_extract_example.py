#!/usr/bin/env python3
"""Extractor. Offline keyword rules by default; an LLM if LLM_BASE_URL, LLM_API_KEY and TEXT_LLM_MODEL (or LLM_MODEL) are set.
    python 4_extract_example.py --data data/unified/dev.jsonl --out preds.jsonl --limit 10 [--mode llm|offline] [--workers 4]

In LLM mode, CUAD records go through the four-step agent in agent_confidence.py, which produces the answers and their
confidence. Other records (ExtractBench) use the single-call extract() below. --no-agent uses extract() for everything.
The variables are read from the environment, or from .env next to this file.
"""
import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))
import agent_confidence as agent      # the four-step agent used for CUAD records
MAX_CHARS = 60_000

PROMPT = """Extract the fields below from the document. Reply with ONE JSON object and nothing else:
{{"<field>": {{"value": <string, number, JSON for nested fields, or null>, "confidence": <0..1>}}, ...}}
Rules:
- Copy values as printed in the document. Do not guess, infer or compute.
- If a field is not in the document, or you cannot read it, use null. A wrong value costs 100 times more than a null.
- Confidence is how sure you are the value is exactly right.

Fields (name: description):
{fields}
"""


def describe(schema: dict) -> str:
    out = []
    for name, spec in schema["properties"].items():
        desc = spec.get("description") or spec.get("title") or spec.get("type") or ""
        kind = spec.get("x-normalizer")
        out.append(f"- {name}" + (f" [{kind}]" if kind else "") + (f": {str(desc)[:300]}" if desc else ""))
    return "\n".join(out)


def pdf_text(path: Path) -> str | None:
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    try:
        from pypdf import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        return text if text.strip() else None
    except ImportError:
        return None


_cuad = None


def cuad_text(doc_id: str) -> str:
    global _cuad
    if _cuad is None:
        raw = json.loads((DATA / "raw/cuad/CUADv1.json").read_text(encoding="utf-8"))["data"]
        _cuad = {d["title"]: "\n".join(p["context"] for p in d["paragraphs"]) for d in raw}
    return _cuad[doc_id.split(":", 1)[1]]


def build_content(rec: dict) -> tuple[list, str | None]:
    prompt = PROMPT.format(fields=describe(rec["schema"]))
    page = DATA / rec["pages"][0]
    if rec["source_dataset"] == "cuad":
        text = cuad_text(rec["doc_id"])[:MAX_CHARS]
        return [{"type": "text", "text": f"{prompt}\n--- DOCUMENT ---\n{text}"}], text
    if page.suffix.lower() == ".pdf":
        text = pdf_text(page)
        if text:
            text = text[:MAX_CHARS]
            return [{"type": "text", "text": f"{prompt}\n--- DOCUMENT ---\n{text}"}], text
    # no text available: attach the file for a vision-capable model
    mime = mimetypes.guess_type(page.name)[0] or "application/octet-stream"
    url = f"data:{mime};base64,{base64.b64encode(page.read_bytes()).decode()}"
    part = ({"type": "image_url", "image_url": {"url": url}} if mime.startswith("image/")
            else {"type": "file", "file": {"filename": page.name, "file_data": url}})
    return [{"type": "text", "text": prompt}, part], None


def call_llm(content: list) -> str:
    base, key, model = (os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
    if not (base and key and model):
        sys.exit("Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (see the top of this file).")
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 16000,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse_json(reply: str) -> dict:
    s = reply.strip()
    if s.startswith("```"):
        s = s.strip("`").removeprefix("json").strip()
    try:
        return json.loads(s[s.index("{"): s.rindex("}") + 1])
    except ValueError:
        return {}


def grounded(value, text: str | None) -> bool:
    if text is None or isinstance(value, bool) or not isinstance(value, (str, int, float)) or len(str(value)) > 80:
        return True
    squash = lambda s: "".join(str(s).lower().split()).replace(",", "")
    return squash(value) in squash(text)


def doc_text(rec: dict) -> str | None:
    if rec["source_dataset"] == "cuad":
        return cuad_text(rec["doc_id"])[:MAX_CHARS]
    page = DATA / rec["pages"][0]
    return (pdf_text(page) or "")[:MAX_CHARS] or None if page.suffix.lower() == ".pdf" else None


PATTERNS = {
    "amount": r"-?\d[\d.,]*\d|\d",
    "date": r"\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|[A-Z][a-z]{2,8}\.? \d{1,2},? \d{4}|\d{1,2} [A-Z][a-z]{2,8}\.? \d{4}",
    "id": r"[A-Za-z0-9][A-Za-z0-9\-/]{3,}",
}


def offline_extract(rec: dict) -> dict:
    text = doc_text(rec)
    fields = {}
    for name, spec in rec["schema"]["properties"].items():
        kind = spec.get("x-normalizer")
        words = [w for w in re.split(r"[\W_]+", name.lower()) if len(w) > 2]
        value, conf = None, 0.0
        if text and words:
            if kind == "span":
                for sent in re.split(r"(?<=[.;])\s+", text):
                    if all(w in sent.lower() for w in words) and 20 < len(sent) < 600:
                        value, conf = " ".join(sent.split()), 0.3
                        break
            elif kind in PATTERNS:
                for line in text.splitlines():
                    low = line.lower()
                    if all(w in low for w in words):
                        tail = line[low.rindex(words[-1]) + len(words[-1]):]
                        m = re.search(PATTERNS[kind], tail)
                        if m:
                            value, conf = m.group(0).strip(), 0.5
                            break
        fields[name] = {"value": value, "confidence": conf}
    return {"doc_id": rec["doc_id"], "fields": fields}


def extract(rec: dict) -> dict:
    content, text = build_content(rec)
    answer = parse_json(call_llm(content))
    fields = {}
    for name in rec["schema"]["properties"]:
        a = answer.get(name)
        a = a if isinstance(a, dict) else {"value": a}
        value, conf = a.get("value"), a.get("confidence", 0.5)
        try:
            conf = min(max(float(conf), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.0
        if value is not None and not grounded(value, text):
            conf *= 0.5
        fields[name] = {"value": value, "confidence": round(conf, 3)}
    return {"doc_id": rec["doc_id"], "fields": fields}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="e.g. data/unified/dev.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="only the first N documents (start small: LLM calls cost money)")
    ap.add_argument("--mode", choices=["auto", "llm", "offline"], default="auto",
                    help="auto: LLM if LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are all set, else offline rules")
    ap.add_argument("--workers", type=int, default=4, help="documents processed in parallel (default 4)")
    ap.add_argument("--no-agent", action="store_true", help="CUAD: use the single-call extract() instead of the four-step agent")
    a = ap.parse_args()
    env = HERE / ".env"
    if env.exists():                                         # KEY=VALUE lines; variables already set win
        for line in env.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    if os.environ.get("TEXT_LLM_MODEL"):
        os.environ.setdefault("LLM_MODEL", os.environ["TEXT_LLM_MODEL"])
    has_key = all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))
    use_llm = a.mode == "llm" or (a.mode == "auto" and has_key)
    print(f"mode: {'LLM (' + os.environ.get('LLM_MODEL', '?') + ')' if use_llm else 'offline rules (no API key needed)'}"
          + ("" if not use_llm or a.no_agent else ", four-step agent for CUAD"), file=sys.stderr)

    def run_one(rec: dict) -> dict:
        if not use_llm:
            return offline_extract(rec)
        if rec["source_dataset"] == "cuad" and not a.no_agent:
            return agent.agent_confidence(rec)
        return extract(rec)
    recs = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["doc_id"] for line in open(a.out, encoding="utf-8") if line.strip()}
    todo = [rec for rec in recs[: a.limit] if rec["doc_id"] not in done]
    lock = threading.Lock()

    def work(rec: dict) -> tuple[dict, str | None]:
        try:
            return run_one(rec), None
        except Exception as e:                                # a failed document goes to a human: every field null
            return {"doc_id": rec["doc_id"], "fields": {f: {"value": None} for f in rec["schema"]["properties"]}}, f"{type(e).__name__}: {e}"

    with open(a.out, "a", encoding="utf-8") as fout, concurrent.futures.ThreadPoolExecutor(max(1, a.workers)) as pool:
        futures = {pool.submit(work, rec): rec for rec in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            row, error = fut.result()
            with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
            answered = sum(f.get("value") is not None for f in row["fields"].values())
            print(f"[{i}/{len(todo)}] {row['doc_id'][:80]}  {answered}/{len(row['fields'])} answered"
                  + (f"  FAILED: {error}" if error else ""), file=sys.stderr, flush=True)
    u = agent.USAGE
    if u["calls"]:
        print(f"agent: {u['calls']} calls, {u['prompt_tokens']:,} input + {u['completion_tokens']:,} output tokens, "
              f"${u['cost_usd']:.2f} as reported by the gateway", file=sys.stderr)


if __name__ == "__main__":
    main()
