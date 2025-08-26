#!/usr/bin/env python3
"""
train_shadow_insta.py

Example:
  python train_shadow_insta.py --csv "C:/Users/you/reels_log.csv" --out_dir "C:/Users/you/shadow_insta" --k 6 --horizon 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from scipy.sparse import hstack


# ---------------------------- utils ----------------------------

def topk_acc(y_true, prob, k=1):
    try:
        return float(top_k_accuracy_score(y_true, prob, k=k, labels=np.arange(prob.shape[1])))
    except Exception:
        return float("nan")


# ------------------------ category builder ---------------------

def build_categories(df, k=12, min_docs=500, random_state=42):
    """
    Assigns an unsupervised 'category' to each row.
    - Uses TF-IDF -> SVD -> KMeans on text (caption+hashtags or account)
    - Falls back to hashing account|content_id if text is unusable
    - Applies to reel/video/post (or entire df if too few)
    Returns: df_with_category, vec_cat, svd_cat, km_cat
    """
    import zlib

    short = df[df["content_type"].isin(["reel", "video", "post"])].copy()
    if len(short) < 50:
        short = df.copy()

    # Build robust text
    short["text"] = (
        short["caption"].fillna("").astype(str) + " " +
        short["hashtags"].fillna("").astype(str)
    ).str.strip()
    blank = short["text"].str.len().fillna(0).eq(0)
    short.loc[blank, "text"] = short.loc[blank, "account"].fillna("").astype(str)
    short["text"] = short["text"].replace(r"^\s*$", "placeholder", regex=True)

    vec_cat = svd_cat = km_cat = None

    try:
        vec = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1,
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b",
        )
        X = vec.fit_transform(short["text"])
        if X.shape[0] == 0 or X.nnz == 0:
            raise ValueError("degenerate tfidf matrix")

        n_comp = min(128, max(2, X.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        Xs = svd.fit_transform(X)

        k_eff = int(min(k, max(2, Xs.shape[0])))
        km = KMeans(n_clusters=k_eff, random_state=random_state, n_init="auto")
        labels = km.fit_predict(Xs)
        short["category"] = labels

        vec_cat, svd_cat, km_cat = vec, svd, km

    except Exception:
        # Fallback: deterministic hash buckets on account|content_id
        key = (short["account"].fillna("").astype(str) + "|" +
               short["content_id"].fillna("").astype(str))
        key = np.where(key == "|", short["content_id"].fillna("").astype(str), key)
        short["category"] = [int(zlib.adler32(s.encode()) % max(2, k)) for s in key]

    # Map back by (content_id, timestamp)
    key_cols = ["content_id", "timestamp"]
    out = df.copy()
    out = out.merge(short[key_cols + ["category"]], on=key_cols, how="left")
    return out, vec_cat, svd_cat, km_cat


# ------------------------- sequence builder --------------------

def build_sequence(df, horizon=1):
    """
    Build features X_t and labels y_{t+horizon} (time-ordered).
    Returns train/test split and fitted feature encoders.
    """
    from sklearn.preprocessing import OneHotEncoder
    import sklearn

    d = df.sort_values("timestamp").copy()
    d = d[d["category"].notna()].copy()
    d["category"] = d["category"].astype(int)

    # Robust text for features
    d["text"] = (
        d["caption"].fillna("").astype(str) + " " +
        d["hashtags"].fillna("").astype(str)
    ).str.strip()
    blank = d["text"].str.len().fillna(0).eq(0)
    d.loc[blank, "text"] = d.loc[blank, "account"].fillna("").astype(str)
    d["text"] = d["text"].replace(r"^\s*$", "placeholder", regex=True)

    vec_f = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        X_text = vec_f.fit_transform(d["text"])
        if X_text.nnz == 0:
            raise ValueError("degenerate tfidf")
    except Exception:
        X_text = vec_f.fit_transform(
            d["account"].fillna("").astype(str).replace(r"^\s*$", "placeholder", regex=True)
        )

    # Version-proof OneHotEncoder
    ver = tuple(map(int, sklearn.__version__.split(".")[:2]))
    if ver >= (1, 2):  # sklearn 1.2+ uses 'sparse_output'
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    else:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True)

    X_meta = enc.fit_transform(d[["interaction", "content_type"]])

    X_all = hstack([X_text, X_meta], format="csr")
    y = d["category"].values

    # Predict t+H
    H = max(1, int(horizon))
    X_seq = X_all[:-H]
    y_next = y[H:]
    if X_seq.shape[0] < 10:
        raise SystemExit(f"Not enough sequence length for horizon={H}. Try smaller horizon or more data.")

    # Time split
    n = X_seq.shape[0]
    split = int(np.floor(n * 0.85))
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_next[:split], y_next[split:]

    artifacts = {"vec_features": vec_f, "encoder": enc, "horizon": H}
    # d_aligned is the feature rows used to produce predictions; align to X_seq
    return (X_tr, y_tr, X_te, y_te, artifacts, d.iloc[:X_seq.shape[0]].copy())



# ------------------------- markov baseline ---------------------

def markov_baseline(categories):
    cats = np.array(categories, dtype=int)
    if cats.size == 0:
        return np.array([[1.0]], dtype=float)
    n_classes = int(cats.max()) + 1
    T = np.ones((n_classes, n_classes), dtype=float)  # Laplace smoothing
    for a, b in zip(cats[:-1], cats[1:]):
        T[a, b] += 1.0
    T = T / T.sum(axis=1, keepdims=True)
    return T


def predict_markov(T, current_cat):
    if int(current_cat) < T.shape[0]:
        return T[int(current_cat)]
    return np.ones(T.shape[1]) / T.shape[1]


# ------------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to reels_log.csv")
    ap.add_argument("--out_dir", default="shadow_insta", help="Output directory")
    ap.add_argument("--k", type=int, default=12, help="Number of unsupervised categories")
    ap.add_argument("--horizon", type=int, default=1, help="Predict t+H (default 1)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & sort
    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    # Build categories & save
    df_cat, vec_cat, svd_cat, km_cat = build_categories(df, k=args.k)
    df_cat[["timestamp", "content_type", "content_id", "account", "caption", "hashtags", "category", "source_file"]] \
        .to_csv(out_dir / "inferred_categories.csv", index=False)

    # Markov baseline (guard empties first)
    seq_cats = (
        df_cat[df_cat["category"].notna()]
        .sort_values("timestamp")["category"].astype(int).values
    )
    if seq_cats.size < 2:
        markov_T = np.array([[1.0]], dtype=float)
        pd.DataFrame(markov_T).to_csv(out_dir / "markov_transitions.csv", index=False)
        (out_dir / "metrics.json").write_text(
            json.dumps({"note": "insufficient sequence length for classifier", "samples_test": 0}),
            encoding="utf-8"
        )
        print("Not enough sequential data to train classifier. Wrote inferred_categories.csv and markov_transitions.csv.")
        return
    else:
        markov_T = markov_baseline(seq_cats)
        pd.DataFrame(markov_T).to_csv(out_dir / "markov_transitions.csv", index=False)

    # Build train/test with horizon
    X_tr, y_tr, X_te, y_te, feats_art, d_aligned = build_sequence(df_cat, horizon=args.horizon)

    # Classifier
    clf = LogisticRegression(max_iter=1000, multi_class="auto", class_weight="balanced")
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)
    top1 = topk_acc(y_te, prob, k=1)
    top3 = topk_acc(y_te, prob, k=3)

    # Markov eval aligned to same windows (use current category for each test row)
    curr = d_aligned["category"].values
    curr_te = curr[len(y_tr):]
    prob_mkv = np.vstack([predict_markov(markov_T, c) for c in curr_te])
    top1_mkv = topk_acc(y_te, prob_mkv, k=1)
    top3_mkv = topk_acc(y_te, prob_mkv, k=3)

    # Metrics
    metrics = {
        "samples_train": int(X_tr.shape[0]),
        "samples_test": int(X_te.shape[0]),
        "n_categories": int(markov_T.shape[0]),
        "horizon": int(feats_art.get("horizon", 1)),
        "logreg_top1": top1,
        "logreg_top3": top3,
        "markov_top1": top1_mkv,
        "markov_top3": top3_mkv,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Persist artifacts
    joblib.dump(clf, out_dir / "model.pkl")
    joblib.dump(feats_art["vec_features"], out_dir / "vectorizer.pkl")
    joblib.dump(feats_art["encoder"], out_dir / "encoder.pkl")
    joblib.dump({"vec_cat": vec_cat, "svd_cat": svd_cat, "kmeans_cat": km_cat}, out_dir / "category_builder.pkl")

    print("Done!")
    print("Outputs saved to:", out_dir)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
