# Team Selous — Health QA / RAG Project

TRI Saturday AI Course project. Retrieval-based question answering over a
synthetic health knowledge base (maternal health, chronic disease, emergency
triage, vaccination, nutrition, mental health basics, medication safety,
infectious disease).

## Evaluation
Mean character-level Levenshtein distance between predicted `Answer` and the
hidden reference answer (lower is better). Reference answers are short, fixed
sentences pulled from a small pool — see `docs/benchmark_notes.md` for what
this implies about strategy (retrieval > free-form generation)

## Repo structure
```
data/
  raw/            # original provided files, never edited in place
  processed/      # cleaned/derived data (embeddings, cached vectors, etc.)
src/              # scripts and reusable pipeline code
notebooks/        # exploration, experiments, scratch work
submissions/      # generated submission.csv files, one per approach/version
docs/             # notes, write-ups, benchmark documentation
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Current status
- [x] Benchmark exploration — see `docs/benchmark_notes.md`
- [x] TF-IDF baseline reproduced — `src/tfidf_baseline.py` → `submissions/tfidf_baseline_submission.csv`
- [ ] Improved retrieval (embeddings / dense retrieval)
- [ ] RAG pipeline with generation
- [ ] Final eval + write-up

## Team
- Damola Adams
- 

## Running the baseline
```bash
cd src
python tfidf_baseline.py
```
Outputs to `submissions/tfidf_baseline_submission.csv`.
