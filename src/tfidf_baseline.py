"""
Team Selous - Week 1: TF-IDF Baseline Reproduction
Reproduces the retrieval-based baseline for the health QA benchmark.

Method: concatenate question + topic + care_setting + population as the
matching text, TF-IDF vectorize train and test questions, cosine similarity,
and copy the nearest train neighbor's reference_answer as the prediction.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train = pd.read_csv("train_qa.csv")
test = pd.read_csv("test_questions.csv")

def build_text(df):
    return df["question"] + " " + df["topic"] + " " + df["care_setting"] + " " + df["population"]

train_text = build_text(train)
test_text = build_text(test)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

sims = cosine_similarity(X_test, X_train)
best_match_idx = sims.argmax(axis=1)

predictions = [train.iloc[idx]["reference_answer"] for idx in best_match_idx]

submission = pd.DataFrame({
    "QuestionId": test["QuestionId"],
    "Answer": predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(submission), "rows")
