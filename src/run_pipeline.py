#!/usr/bin/env python3
"""End-to-end retrieval-only pipeline for the Selous Health QA project.

Steps:
  1. Load configuration and training data.
  2. Split into train / validation.
  3. Build a StyleExemplarIndex on the train split.
  4. Validate: predict on the val split, compute mean Levenshtein distance.
  5. Rebuild the index on the full dataset (train + val).
  6. Load the test set, generate predictions, write submission.csv.
  7. Write a separate safety-review CSV for flagged rows.

Usage (from the project root):
    python -m src.run_pipeline
"""

from .config import CONFIG
from .data import load_train_data, split_train_val, load_test_data
from .retrieval import load_embedder, StyleExemplarIndex
from .inference import retrieve_answer, levenshtein, write_submission, write_safety_review


def main():
    # ── 1. Load config and data ──────────────────────────────────────────
    config = CONFIG
    qa = load_train_data(config)

    # ── 2. Train / validation split ──────────────────────────────────────
    train_split, val_split = split_train_val(qa, config)

    # ── 3. Build retrieval index on train split only ─────────────────────
    print("\nLoading embedding model…")
    embedder = load_embedder(config)
    print("Building style-exemplar index on train split…")
    train_index = StyleExemplarIndex(embedder, train_split)

    # ── 4. Validate on held-out split ────────────────────────────────────
    print(f"\nValidating on {len(val_split)} held-out rows…")
    val_distances = []
    for _, row in val_split.iterrows():
        pred, _flag = retrieve_answer(
            row, train_index, train_split, config,
            exclude_row_id=row["QuestionId"],
        )
        dist = levenshtein(pred, row["reference_answer"])
        val_distances.append(dist)

    mean_dist = sum(val_distances) / len(val_distances)
    print(f"  Mean Levenshtein distance on val split: {mean_dist:.2f}")

    # ── 5. Rebuild index on full labelled data ───────────────────────────
    print("\nRebuilding index on full training data (train + val)…")
    full_index = StyleExemplarIndex(embedder, qa)

    # ── 6. Load test set, predict, write submission ──────────────────────
    test_df = load_test_data(config)
    print(f"Generating predictions for {len(test_df)} test questions…")

    predictions = []
    flags = []
    for _, row in test_df.iterrows():
        pred, flag = retrieve_answer(row, full_index, qa, config)
        predictions.append(pred)
        flags.append(flag)

    write_submission(test_df, predictions, out_path=config["submission_csv"])

    # ── 7. Safety-review report ──────────────────────────────────────────
    write_safety_review(test_df, predictions, flags,
                        out_path=config["safety_review_csv"])

    print("\nDone.")


if __name__ == "__main__":
    main()
