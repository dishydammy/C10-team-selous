# Team Selous — Health Information & Triage Assistant (SLM + RAG)

An evidence-grounded health information and basic triage assistant for African
communities, combining a small language model (SLM) with Retrieval-Augmented
Generation (RAG) over a curated public-health knowledge base.

**Disclaimer:** This is a research prototype. It does not diagnose disease,
prescribe medication, or replace professional medical care. It is designed to
provide concise, evidence-based information and to escalate to professional
or emergency care when appropriate.

## Overview

Many people in African communities — especially caregivers, pregnant
individuals, and people managing chronic or infectious disease — struggle to
access reliable, understandable health information. Our system retrieves
answers from trusted public-health sources and generates short, clinically
sensible responses, explicitly flagging when a user should seek in-person
care. It is guided by the values of safety, accuracy, fairness, accessibility,
transparency, empathy, and community well-being.

## Dataset

### Question–answer pairs

| File | Rows | Description |
|---|---|---|
| `data/raw/train_qa.csv` | 43 | Original benchmark training set (question, topic, care_setting, population, document_id, reference_answer, QuestionId) |
| `data/raw/test_questions.csv` | 11 | Test questions (no reference_answer column) |
| `data/raw/sample_submission.csv` | 11 | Example submission format |
| `data/processed/train_qa.csv` | 83 | Augmented training set (same schema, roughly 2× the original) |

### Knowledge-base documents

The knowledge base combines two collections of plain-text documents, organized
by topic in `data/processed/`:

- **`data_one/`** — 104 documents across 9 topics (child/infant health,
  chronic disease, emergency triage, infectious diseases, maternal health,
  medication safety, mental-health basics, nutrition, vaccination), collected
  from verified public-health sources (WHO, UNICEF, CDC, Nigerian health
  authorities).
- **`data_two/`** — 48 synthetic documents (6 per topic × 8 topics), written
  to target likely user questions rather than generic articles, improving
  practical coverage.

Each collection includes a `metadata.csv` with structured fields
(`Source_ID`, `File_Name`, `Topic`, `Population`, `Care_Setting`) used for
filtering, not embedded content.

`data/raw/documents.csv` contains 24 shorter document summaries (3 per
topic, all synthetic) — a separate, compact reference set.

## Pipeline

The `src/` directory implements a **retrieval-only** pipeline (no LLM
generation at inference time). The approach:

1. **Embed** training questions with `intfloat/e5-small-v2` (asymmetric
   `passage:`/`query:` prefixes, L2-normalized).
2. **Retrieve** — for each incoming question, find the most similar training
   question by cosine similarity over the embeddings.
3. **Copy** — return that neighbor's `reference_answer` verbatim as the
   prediction.
4. **Safety review** — scan high-risk topics (emergency triage, medication
   safety) for escalation keywords; flag answers missing safety-net language
   in a separate report.

A near-duplicate threshold (`cosine ≥ 0.92`) allows confident direct copies;
below that, the top-k nearest exemplars still provide the answer.

### Fine-tuning (notebooks only)

LoRA/PEFT fine-tuning with `unsloth/Qwen3-4B-unsloth-bnb-4bit` and a
ChromaDB-backed RAG pipeline were explored in the Colab notebooks under
`notebooks/`. These require a GPU and are **not** part of the local `src/`
pipeline. See the notebooks for details:

- `team_selous_finetune.ipynb` — retrieval-assisted LoRA fine-tuning with
  cross-validation.
- `team_selous_rag_finetune_v2.ipynb` — full RAG + ChromaDB + SFT training
  + inference pipeline (Colab/Kaggle GPU required).

## Evaluation

Primary metric: **mean Levenshtein distance** between predicted and hidden
reference answers (lower is better), which rewards concise, well-matched
phrasing. Because this metric alone doesn't capture clinical quality, we also
assess:
- **Retrieval quality** — whether relevant evidence was retrieved.
- **Grounding** — whether the answer reflects retrieved evidence.
- **Correctness** — medical consistency with evidence.
- **Safety** — avoidance of diagnosis/prescription, appropriate escalation.
- **Robustness** — behavior on ambiguous or out-of-scope questions.

We also plan structured human review: a healthcare-professional committee
rates high-risk questions (medication, emergency, pregnancy, infant-health)
as acceptable / needs revision / needs escalation / unsafe, and
participatory-design sessions with prospective users assess clarity, tone,
and cultural fit. Feedback from both feeds back into the dataset, escalation
rules, and test cases.

## Running locally

**Prerequisites:** Python 3.9+. No GPU required — the `src/` pipeline runs
on CPU (the E5-small-v2 embedding model is ~130 MB).

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd selous-health-qa

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full pipeline (validate → predict → write submission)
python -m src.run_pipeline
```

This will:
1. Load `data/raw/train_qa.csv` and split into train (85%) / validation (15%).
2. Build an embedding index on the train split.
3. Predict on the validation split and print the **mean Levenshtein distance**.
4. Rebuild the index on the full training set.
5. Predict on `data/raw/test_questions.csv`.
6. Write `submissions/submission.csv` and `submissions/safety_review.csv`.

### TF-IDF baseline (standalone)

```bash
# Run from the data/raw/ directory (the script uses relative CSV paths)
cd data/raw
python ../../src/tfidf_baseline.py
```

This produces a `submission.csv` in the current directory using TF-IDF
cosine similarity instead of learned embeddings.

## Repo structure

```
selous-health-qa/
├── data/
│   ├── raw/
│   │   ├── train_qa.csv              # 43 labelled QA pairs (benchmark)
│   │   ├── test_questions.csv        # 11 test questions (no answers)
│   │   ├── documents.csv             # 24 compact document summaries
│   │   └── sample_submission.csv     # submission format example
│   └── processed/
│       ├── data_one/                 # 104 original source documents (.txt), 9 topics
│       │   └── metadata.csv          # per-document metadata
│       ├── data_two/                 # 48 synthetic documents (.txt), 8 topics × 6
│       │   └── metadata.csv          # per-document metadata
│       └── train_qa.csv              # 83-row augmented training set
├── src/
│   ├── __init__.py                   # package marker
│   ├── config.py                     # paths and hyperparameters
│   ├── data.py                       # load CSVs, train/val split
│   ├── retrieval.py                  # E5 embeddings, StyleExemplarIndex
│   ├── inference.py                  # nearest-neighbor prediction, Levenshtein, safety review
│   ├── tfidf_baseline.py            # standalone TF-IDF baseline script
│   └── run_pipeline.py              # end-to-end entry point (validate → predict → submit)
├── notebooks/
│   ├── team_selous_finetune.ipynb             # LoRA fine-tuning (Colab, GPU)
│   └── team_selous_rag_finetune_v2.ipynb      # RAG + ChromaDB + SFT (Colab, GPU)
├── submissions/
│   ├── baseline_submission.csv       # pre-generated baseline output
│   └── tfidf_baseline_submission.csv # pre-generated TF-IDF output
├── docs/
│   ├── Team Selous Data Card.pdf
│   ├── Team Selous Impact Statement.pdf
│   ├── Team Selous Problem Statement.pdf
│   └── Team Selous Stakeholder Engagement Plan.pdf
├── requirements.txt                  # pandas, scikit-learn, numpy, sentence-transformers
└── README.md
```

## Appendix

**Team — Team Selous**
- Damola Adams
- Ejiabor Rita
- Adeniji Elijah
- Taofeek Kehinde
- Iheanacho Victor

**Mentors**
- Kosy Ashara