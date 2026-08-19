# Team Selous — Week 1 Notes

## Benchmark structure
- `documents.csv`: 24 synthetic health articles, 8 topics × ~3 docs each. Fields: topic, care_setting, population, text, source_url, license (CC0-1.0).
- `train_qa.csv`: 43 Q/A pairs, each linked to a document_id, with a short single-sentence `reference_answer`.
- `test_questions.csv`: 11 questions, no answers, same metadata schema.
- `sample_submission.csv` / `baseline_submission.csv`: QuestionId, Answer format.

Topics: chronic_disease, infectious_disease, maternal_health, medication_safety,
nutrition, vaccination, emergency_triage, mental_health_basics.
Care settings: primary_care (dominant), community, home, hospital, school.
Population: child, adult, general, pregnant, infant.

## Evaluation metric
**Mean character-level Levenshtein distance** between predicted Answer and the
hidden reference (lower is better) — matches the Kaggle "legacy Levenshtein mean" metric.

Implication: this is an edit-distance/near-verbatim task, not semantic QA scoring.
Reference answers are drawn from a small fixed pool of canned sentences, so the
winning strategy is retrieving the closest existing sentence and copying it
verbatim — paraphrasing or generating free text will only add edit distance.
This favors retrieval-based baselines (TF-IDF nearest neighbor) over generative
approaches unless generation is constrained to reproduce the reference phrasing style.

## Baseline reproduction
Method: TF-IDF vectorize `question + topic + care_setting + population` for train
and test, cosine similarity, copy the reference_answer of the nearest train neighbor.

Result: mean Levenshtein of 5.9 vs `baseline_submission.csv` (10/11 questions exact
0-distance match; one miss on QuestionId 1010 due to thin lexical overlap with its
true nearest neighbor).
