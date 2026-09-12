"""Retrieval-only inference: nearest-neighbor lookup -> safety check, plus
the competition scoring metric and submission writing.

One function per question:
  1. Use the row's topic/population/care_setting if present, else derive_fields(...).
  2. Retrieve the nearest training question by embedding similarity.
  3. Return that neighbor's reference_answer directly — no generation step.
  4. Run a safety review flag for high-risk topics — reported separately,
     never appended to the scored answer.
"""
import pandas as pd

from .retrieval import derive_fields


def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


_SAFETY_KEYWORDS = {
    "emergency_triage": ["facility", "hospital", "urgent", "immediate", "emergency", "clinician", "seek care"],
    "medication_safety": ["clinician", "doctor", "label", "dose", "overdose", "pharmacist", "review"],
}


def safety_review(topic, answer_text):
    keywords = _SAFETY_KEYWORDS.get(topic)
    if not keywords:
        return None
    if not any(k in answer_text.lower() for k in keywords):
        return f"Review suggested: '{topic}' answer has no explicit safety-net language."
    return None


def retrieve_answer(row, style_index, train_split, config, exclude_row_id=None):
    """Retrieval-only answer for one row: find the nearest training question
    by embedding similarity and return its reference_answer directly — no
    model generation involved. topic/population/care_setting are used only
    for the safety-review flag (derived via nearest neighbors if missing)."""
    neighbour_row, sim = style_index.best_near_duplicate(row["question"])
    if exclude_row_id is not None and neighbour_row["QuestionId"] == exclude_row_id:
        # exclude the row's own match (validation), fall back to the next-nearest
        candidates = style_index.nearest(row["question"], k=2, exclude_row_id=exclude_row_id)
        neighbour_row, sim = candidates[0] if candidates else (neighbour_row, sim)

    answer = neighbour_row["reference_answer"]

    topic = row.get("topic")
    if pd.isna(topic):
        derived = derive_fields(style_index.embedder, row["question"], train_split)
        topic = derived["topic"]

    flag = safety_review(topic, answer)
    return answer, flag


def write_submission(test_df, predictions, out_path="submission.csv"):
    submission = pd.DataFrame({
        "QuestionId": test_df["QuestionId"],
        "Answer": predictions,
    })
    submission.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(submission)} rows")
    return submission


def write_safety_review(test_df, predictions, flags, out_path="safety_review.csv"):
    """Safety-review report, kept separate from submission.csv on purpose —
    never appended to the scored Answer column."""
    review_df = pd.DataFrame({
        "QuestionId": test_df["QuestionId"],
        "question": test_df["question"],
        "Answer": predictions,
        "safety_flag": flags,
    })
    review_df = review_df[review_df["safety_flag"].notna()]
    if len(review_df):
        review_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} with {len(review_df)} flagged row(s) for manual review.")
    else:
        print("No rows flagged for manual safety review.")
    return review_df
