"""Central configuration for the retrieval-only pipeline.

Every path/hyperparameter the pipeline uses lives here, in one place.
"""

from pathlib import Path

# All data paths are relative to the project root (one level above src/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # --- data ---
    "train_csv": str(_PROJECT_ROOT / "data" / "raw" / "train_qa.csv"),
    "test_csv": str(_PROJECT_ROOT / "data" / "raw" / "test_questions.csv"),
    "val_fraction": 0.15,
    "random_state": 3407,

    # --- output ---
    "submission_csv": str(_PROJECT_ROOT / "submissions" / "submission.csv"),
    "safety_review_csv": str(_PROJECT_ROOT / "submissions" / "safety_review.csv"),

    # --- embeddings / retrieval ---
    "embed_model_name": "intfloat/e5-small-v2",  # asymmetric: passage:/query: prefixes required
    "n_style_examples": 2,        # nearest short Q&A exemplars shown as a style guide
    "near_dup_threshold": 0.92,   # cosine sim above this -> just copy the neighbour's answer
}
