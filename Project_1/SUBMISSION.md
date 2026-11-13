# Submission: Project 1

## Authors
**Team name:** Xplore

**Project:** Project 1: Teach a machine to say "I don't know" (structured extraction from documents)

## Problem and approach
For every field of a document we return a value and a confidence. Fields at or above a threshold are booked automatically,
the rest go to a human. A wrong booked value costs 100 reviews, so the confidence matters more than the extraction.

All results here are on **CUAD (contracts) only. ExtractBench is not included.**

### What the data told us (EDA)
The plots are in `slides/` (`01` to `03`), each drawn by the script next to it.

- **Where the answers sit** (381 CUAD dev contracts, 10,374 gold answer spans): 25% of all answers start in the first 5% of
  a contract. By quarter of the contract: 41%, 21%, 21%, 17%. So 59% of the answers lie beyond the first quarter and the
  whole contract has to be read.
- **Where the hallucinations sit** (`claude-sonnet-5.5`, one call per contract, 190 dev contracts, 286 answers given where
  gold has none): spread evenly, 27%, 27%, 25%, 21% by quarter, with no peak at the start. Only 4% start in the first 5%.
- **What the hallucinations are:** 277 of the 286 are whole clauses, not a date or a name. 65% mention a party by its
  contract role ("the Company", "Party A"), against 29% of the correct answers, and that share rises through the contract
  from 56% in the first quarter to 75% in the last.

#### NER tags in the hallucinated answers
Every answer was tagged with named entities by `claude-sonnet-5.5` (slide `03`, data in
`experiments/cuad_claude-sonnet-5.5_f0.5_s13_ner.jsonl` and `..._ner_report.json`): the 286 hallucinated answers and the
1,032 correct ones from the same run. Each answer gets one answer type (what the answer as a whole is) and a list of
entity tags found inside it.

**Answer type.** Hallucinations are almost never a single value.

| Answer type | Hallucinated (286) | Correct (1,032) |
| --- | --- | --- |
| CLAUSE (a passage to be read) | 277 (96.9%) | 603 (58.4%) |
| DATE | 9 (3.1%) | 217 (21.0%) |
| DOC (document name) | 0 | 171 (16.6%) |
| ORG | 0 | 40 (3.9%) |
| PERSON | 0 | 1 (0.1%) |

**Entity tags inside the answers.** Share of answers that contain at least one entity with the tag (an answer can carry
several tags, so the columns do not add up to 100%).

| NER tag | Hallucinated answers | Correct answers | Most frequent in hallucinations |
| --- | --- | --- | --- |
| ROLE (party by contract role) | 187 (65.4%) | 302 (29.3%) | "Company", "Party", "Party A", "Parties", "Party B", "Client" |
| ORG | 85 (29.7%) | 230 (22.3%) | "ICC", "TrueLink", "Excite" |
| DOC | 74 (25.9%) | 308 (29.8%) | "Agreement", "Exhibit A", "Documentation" |
| PRODUCT | 70 (24.5%) | 125 (12.1%) | "Product", "Hosting Services", "Products" |
| DURATION | 70 (24.5%) | 207 (20.1%) | "Term", "twelve (12) months", "Initial Term" |
| DATE | 20 (7.0%) | 245 (23.7%) | no entity repeats |
| GPE | 10 (3.5%) | 113 (10.9%) | "Kuwait", "Territory", "Netherlands" |
| MONEY | 7 (2.4%) | 19 (1.8%) | no entity repeats |
| QUANTITY | 7 (2.4%) | 10 (1.0%) | no entity repeats |
| LAW | 3 (1.0%) | 134 (13.0%) | no entity repeats |
| PERCENT | 3 (1.0%) | 13 (1.3%) | no entity repeats |
| PERSON | 2 (0.7%) | 10 (1.0%) | no entity repeats |

What this shows:

- **Over-represented in hallucinations:** ROLE (65% against 29%) and PRODUCT (25% against 12%). These are the defined
  terms of a contract ("the Company", "the Product", "the Term"), which occur in almost every operative clause, so they
  mark generic contract language rather than a specific fact.
- **Under-represented in hallucinations:** DATE (7% against 24%), LAW (1% against 13%) and GPE (4% against 11%). An
  answer anchored on a concrete date, a governing law or a place is rarely a hallucination.
- **Which clauses they come from:** Post-Termination Services (52), Third Party Beneficiary (33), Exclusivity (19),
  Price Restrictions (19), Non-Transferable License (18) and Renewal Term (15) account for 156 of the 286. All of them
  are clause-type questions; only Effective Date (10) produces hallucinated dates.

This is why step 4 of the method separates a named value from a clause to be read: values can be checked against the
contract by code, clauses cannot, and the clauses are where the hallucinations are.

### The ground truth has gaps
Many answers counted as hallucinations are in the contract: gold marks the clause as absent, but the context holds the answer.
Of the 286 hallucinated answers above:

- the quoted text is found in the contract for 284;
- `claude-opus-5.5`, given only the contract, the question and the answer, judged 280 of them to be in the contract and to
  answer the question.

Example: in the SightLife Surgical agreement, under the heading "Exclusivity:", the contract says "Surgical shall not promote,
market or sell any products for any third party during the Term, which directly or indirectly compete with the Product."
Gold has no Exclusivity answer for that contract. So part of the measured hallucination rate is disagreement with the labels,
which a better model does not remove.

### Approach: a four-step method
`agent_confidence.py`, four LLM calls per contract (slide `04`). Each step asks the model one narrow question and returns
its own confidence; code then checks the answer against the contract.


Dividing the document into steps to granularly verify it similar to rubrik technique.

| Step | The model answers | Code check |
| --- | --- | --- |
| 1 | Is the answer present? (all 41 clauses) | |
| 2 | Where is it? Quote the passages. | The quote is really in the contract. |
| 3 | In which quarter of the contract is it? | It matches where the quote actually sits. |
| 4 | Is it a value (date, name, amount) or a clause to be read? | A named value occurs in the contract. |

The final confidence combines the four step confidences with **equal weight** . The weights are a
design choice: depending on the application they can be varied for each step, for example for legal documents, where a quote that is not
in the contract should count for more than the other steps.

**Assumptions:** the contract text in `CUADv1.json` is the document (the gold answers are spans of it); gold is never read
when predicting.

## Models and why
**Declared confidence threshold: 0.95**

| Model and setup | Open-weight or closed | Where it runs | Main metrics (CUAD, public test) | Cost and time per item | Chosen? |
| --- | --- | --- | --- | --- | --- |
| No-AI baseline (most common value) | n/a | local | Precision 1.9%, recall 0.3% with every answer counted; accepts nothing at 0.95 | none | no, reference only |
| `claude-sonnet-5.5`, four-step method | closed | API (company gateway) | Precision 46.3%, recall 51.8% with every answer counted; 100.0% precision at 0.95 | 4 calls per contract; $0.18 to $0.21 per contract (estimated) | yes |

## Results


**What precision and recall measure here: acceptance.** Every precision, recall and F1 figure in the table  and
in the threshold table below measures the accept-or-send-to-a-human decision, not hallucination detection (that is the
last table of this section). 
Precision is the share of accepted answers that are
right; recall is the share of fields with a gold answer that were accepted and right; F1 is their
harmonic mean. Hallucinations are counted separately, as accepted answers for fields that gold marks absent.

The threshold was chosen by trading precision against recall:

| Threshold | Accepted | Precision | Recall | F1 | Net saving |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 1,891 | 46.3% | 51.8% | 0.489 | -1,885% |
| 0.60 | 1,792 | 48.2% | 51.2% | 0.497 | -1,721% |
| 0.70 | 1,590 | 51.9% | 48.9% | 0.504 | -1,414% |
| 0.75 | 1,434 | 55.0% | 46.7% | 0.505 | -1,194% |
| 0.80 | 1,269 | 58.9% | 44.3% | 0.505 | -963% |
| 0.85 | 1,041 | 63.4% | 39.1% | 0.484 | -701% |
| 0.90 | 717 | 70.7% | 30.0% | 0.422 | -383% |
| 0.93 | 253 | 86.6% | 13.0% | 0.226 | -60% |
| 0.95 | 26 | 100.0% | 1.5% | 0.030 | +0.5% |

Net saving is against reviewing every field by hand (review = 1, wrong booked value = 100).

**Hallucination detection.** The same confidence, read the other way: an answer is flagged when its confidence is below a
value. Here recall is the share of the 370 hallucinated answers that are flagged, and precision is the share of flagged
answers that are hallucinations.

| Flag if confidence is below | Flagged | Hallucinations caught | Recall | Precision |
| --- | --- | --- | --- | --- |
| 0.70 | 301 | 181 of 370 | 48.9% | 60.1% |
| 0.80 | 622 | 293 of 370 | 79.2% | 47.1% |
| 0.85 | 850 | 330 of 370 | 89.2% | 38.8% |
| 0.90 | 1,174 | 358 of 370 | 96.8% | 30.5% |
| 0.93 | 1,638 | 370 of 370 | 100.0% | 22.6% |

Flagging everything below 0.93 catches every hallucination, but it flags 1,638 of the 1,891 answers, so most of what is
flagged is not a hallucination.


**Token cost**

| | Single call | Four-step agent |
| --- | --- | --- |
| Calls per contract | 1 | 4 |
| Cost per contract | $0.066 (measured, 190 contracts) | $0.18 to $0.21 (estimated) |
| Cost per field | $0.0016 | $0.0044 to $0.0050 |



**Limits.**
- ExtractBench (forms and scans) is not part of these results.
- The equal weights were not tuned.
- The hallucination count includes answers that are in the contract but missing from gold (see above), so it overstates
  the invented answers.
- Four calls per contract make the method slow under a rate limit.

## How to run
Python 3.11+ (developed on 3.13). From a fresh clone:

```bash
cd Project_1
bash setup.sh --accept-licenses     # venv, requirements, downloads data/ (about 60 MB), builds data/unified/, runs the baseline
cp .env.example .env                # then fill in the variables below
. .venv/bin/activate

python 4_extract_example.py --data data/unified/test_public.jsonl --out preds_test_public_agent.jsonl
python 5_score.py --gold data/unified/test_public.jsonl --pred preds_test_public_agent.jsonl --threshold 0.80

bash run_hidden.sh <input.jsonl> <output.jsonl>     # hidden test: same extractor, needs data/raw/
```

Environment variables (names only): `LLM_BASE_URL`, `LLM_API_KEY`, `TEXT_LLM_MODEL` (we used `claude-sonnet-5.5`).
Any OpenAI-compatible endpoint works. A stopped run can be restarted with the same command; finished documents are skipped.

The slide plots are reproduced with `python slides/01_answer_position.py` (and `02` to `05`). Plots 2 and 3 read the
files in `experiments/`, made by `cuad_experiment.py`, `cuad_export_hallucinations.py` and `cuad_hallucination_ner.py`.

## Demo
`slides/` (five slides: `01_answer_position.png`, `02_hallucination_position.png`, `03_hallucination_ner.png`, `04_solution.png`, `05_results.png`)

## Next steps
- Vary the step weights by application and tune them on dev, instead of equal weights.
- Have the gold labels reviewed where a strong model finds the answer in the contract, and re-measure the hallucination
  rate against the corrected labels.
- Raise coverage at the safe threshold: today 0.95 accepts 1.5% of the fields.
- Extend the method to ExtractBench.
- For SIXT: book automatically only above the safe threshold, and send the rest to a reviewer with the model's quote as a suggestion.
