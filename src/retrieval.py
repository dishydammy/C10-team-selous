"""Embeddings and style-exemplar retrieval.

Per the project's integration notes, the knowledge base is embedded with
intfloat/e5-small-v2, an *asymmetric* retrieval model: documents must be
embedded as "passage: " + text and queries as "query: " + text, both
L2-normalised. Getting this backwards (or dropping the prefixes) quietly
tanks retrieval quality, so it's wrapped in helper functions rather than
sprinkling string concatenation everywhere.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_embedder(config):
    return SentenceTransformer(config["embed_model_name"])


def embed_passages(embedder, texts):
    """Embed document/knowledge-base chunks (E5 'passage:' convention)."""
    prefixed = ["passage: " + t for t in texts]
    return embedder.encode(prefixed, normalize_embeddings=True).tolist()


def embed_query(embedder, text):
    """Embed a single user query (E5 'query:' convention)."""
    prefixed = "query: " + text
    return embedder.encode([prefixed], normalize_embeddings=True)[0].tolist()


def derive_fields(embedder, question, reference_df, k=5):
    """Fallback: guess topic/population/care_setting from the nearest training
    questions by embedding similarity (majority vote). Used only when a row
    is missing one of those fields."""
    q_emb = np.array(embed_query(embedder, question))
    ref_texts = reference_df["question"].tolist()
    ref_embs = np.array(embed_passages(embedder, ref_texts))
    sims = ref_embs @ q_emb
    top_idx = np.argsort(-sims)[:k]
    neighbours = reference_df.iloc[top_idx]
    return {
        "topic": neighbours["topic"].mode().iat[0],
        "population": neighbours["population"].mode().iat[0],
        "care_setting": neighbours["care_setting"].mode().iat[0],
    }


class StyleExemplarIndex:
    """Retrieves the most similar training QUESTIONS (not raw source
    paragraphs) and exposes their short reference_answer as a style template:
    "answer new questions the same clipped way these were answered".

    Feeding raw source-document paragraphs as "evidence" was tried first and
    scored worse (63 vs 54.5 mean Levenshtein) — it trained the model to
    paraphrase long clinical text, which character-level Levenshtein
    punishes hardest. This is still retrieval, it's just retrieving the
    right thing: style/format exemplars, not raw text.
    """

    def __init__(self, embedder, df):
        self.embedder = embedder
        self.df = df.reset_index(drop=True)
        self.embs = np.array(embed_passages(embedder, self.df["question"].tolist()))

    def nearest(self, question, k=2, exclude_row_id=None):
        q_emb = np.array(embed_query(self.embedder, question))
        sims = self.embs @ q_emb
        order = np.argsort(-sims)
        picked = []
        for idx in order:
            row = self.df.iloc[idx]
            if exclude_row_id is not None and row["QuestionId"] == exclude_row_id:
                continue
            picked.append((row, float(sims[idx])))
            if len(picked) >= k:
                break
        return picked

    def best_near_duplicate(self, question):
        """Return (row, similarity) for the single closest training question."""
        q_emb = np.array(embed_query(self.embedder, question))
        sims = self.embs @ q_emb
        idx = int(np.argmax(sims))
        return self.df.iloc[idx], float(sims[idx])
