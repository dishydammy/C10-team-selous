"""Loading train_qa.csv / test_questions.csv and the train/validation split."""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_train_data(config):
    """train_qa.csv: one row per question, with topic/population/care_setting,
    the reference_answer we're trying to match, and document_id linking
    related questions back to the same source document."""
    qa = pd.read_csv(config["train_csv"])
    print("Shape:", qa.shape)
    print("Columns:", list(qa.columns))
    print("Topics:", sorted(qa["topic"].unique()))
    return qa


def split_train_val(qa, config):
    """Hold out a validation slice before touching retrieval or fine-tuning,
    so the competition metric can be estimated on data the model never sees.
    Stratifies by topic where possible so rare topics don't all land in one split."""
    try:
        train_split, val_split = train_test_split(
            qa,
            test_size=config["val_fraction"],
            random_state=config["random_state"],
            stratify=qa["topic"],
        )
    except ValueError:
        # Falls back to a plain random split if some topic has too few rows to stratify.
        train_split, val_split = train_test_split(
            qa, test_size=config["val_fraction"], random_state=config["random_state"]
        )
    print(f"Train: {len(train_split)}   Val: {len(val_split)}")
    return train_split, val_split


def load_test_data(config):
    test_df = pd.read_csv(config["test_csv"])
    print("Test shape:", test_df.shape)
    return test_df
