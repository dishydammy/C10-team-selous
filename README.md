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

The knowledge base combines two sources:
- **Original documents** collected from verified public-health organizations
  (WHO, UNICEF, CDC, and relevant Nigerian health authorities), covering
  maternal health, chronic disease, infectious disease, medication safety,
  nutrition, vaccination, emergency triage, mental-health basics, and
  child/infant health.
- **Synthetic documents** (48 total, 6 per topic) written to target likely
  user questions rather than generic articles, improving practical coverage.

Testing showed the **combined** set outperforms either alone. Each document
carries simplified metadata (`Source_ID`, `File_Name`, `Topic`, `Population`,
`Care_Setting`) used for structured filtering, not embedded content. We
prioritized reusable, attributable sources, avoided any personally
identifiable health information, and included Nigerian/African guidance
alongside global sources to reduce Global-North-only cultural mismatch.

Reference answers for evaluation come from the provided benchmark dataset;
care was taken to avoid reference-answer leakage into the retrieval index.

## Training Pipeline

**Data collection & preprocessing:** Documents (DOCX/PDF/TXT) are text-
extracted, cleaned, and split using structure-aware chunking (~250–350
tokens/chunk, ~40–80 token overlap), preserving headings and paragraph
boundaries. Metadata is attached per chunk.

**Embedding & indexing:** Chunks are embedded with `intfloat/e5-small-v2`
using the asymmetric `passage:`/`query:` prefix convention, normalized, and
stored in a persisted ChromaDB collection (`kb_combined`). Queries are
embedded with the `query:` prefix; top-k (k=3) semantic retrieval is used,
optionally narrowed by `Topic` metadata.

**Model & fine-tuning:** We evaluated small, resource-efficient models
suited to constrained hardware — an earlier configuration used
`unsloth/Qwen3-4B-unsloth-bnb-4bit` (4-bit), with a later direction toward
Gemma variants. Fine-tuning uses LoRA/PEFT (rank 16, alpha 16, dropout 0.0,
lr 1e-4, batch size 2, grad accumulation 4, 4-bit loading), mapping
`question + retrieved evidence → short answer` rather than memorizing
documents. Validation split = 0.15, random_state = 3407. Epoch count was
reduced from an initial 14 (unnecessarily long) toward fewer epochs, since
the goal is reliable terse output, not training duration.

**Key design choices:** The system prompt frames the model as a "terse
clinical quick-reference," avoiding greetings and repetition. Output length
is controlled via `max_new_tokens` plus a post-generation character cap from
reference-answer length quantiles, preferring generation limits over
aggressive truncation (which risks cutting safety-critical instructions). A
lightweight safety review scans outputs for escalation terms (e.g.,
"hospital," "urgent," "dose," "pharmacist") for medication and emergency
questions.

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

## Reproduction

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run in order:
1. `src/build_knowledge_base.py` — chunk, embed, and load documents into
   ChromaDB (`kb_combined` collection).
2. `src/tfidf_baseline.py` — reproduce the TF-IDF retrieval baseline →
   `submissions/tfidf_baseline_submission.csv`.
3. `src/train_sft.py` — LoRA fine-tuning on question + retrieved-evidence →
   short-answer pairs.
4. `src/generate_answers.py` — run RAG inference (retrieval + generation +
   safety review) over `test_questions.csv` → `submissions/submission.csv`.
5. `src/evaluate.py` — compute mean Levenshtein distance and other metrics
   against reference answers.

Repo structure:
```
data/        raw/ (untouched originals), processed/ (chunks, embeddings)
src/         pipeline scripts (build KB, train, generate, evaluate)
notebooks/   exploration and experiments
submissions/ generated submission.csv files
docs/        benchmark notes and write-ups
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