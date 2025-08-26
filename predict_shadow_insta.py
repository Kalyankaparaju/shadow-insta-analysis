#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

def load_cluster_names(labels_cfg: Path|None, n_classes: int):
    if labels_cfg and labels_cfg.exists():
        cfg = json.loads(labels_cfg.read_text(encoding="utf-8"))
        # cluster_to_tags is like {"0": ["memes"], "1": ["tollywood", "news"], ...}
        m = {int(k): ", ".join(v) for k, v in cfg.get("cluster_to_tags", {}).items()}
        return [m.get(i, f"cat_{i}") for i in range(n_classes)]
    return [f"cat_{i}" for i in range(n_classes)]

def build_features(df: pd.DataFrame, vec, enc):
    # Defensive defaults
    for col in ["account","caption","hashtags","content_type"]:
        if col not in df.columns:
            df[col] = ""

    text = (df["caption"].fillna("").astype(str) + " " +
            df["hashtags"].fillna("").astype(str)).str.strip()
    # fallback for blanks: use account
    blank = text.str.len().fillna(0).eq(0)
    text.loc[blank] = df["account"].fillna("").astype(str)

    X_text = vec.transform(text)
    meta = pd.DataFrame({
        "interaction": df.get("interaction", pd.Series(["view"]*len(df))),
        "content_type": df.get("content_type", pd.Series(["post"]*len(df)))
    })
    X_meta = enc.transform(meta[["interaction","content_type"]])
    X = hstack([X_text, X_meta], format="csr")
    return X

def topk_from_proba(proba, k=3):
    idx = np.argsort(-proba, axis=1)[:, :k]
    vals = np.take_along_axis(proba, idx, axis=1)
    return idx, vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV to score (at least account, caption, hashtags, content_type)")
    ap.add_argument("--model_dir", required=True, help="Folder with model.pkl, vectorizer.pkl, encoder.pkl")
    ap.add_argument("--labels_config", default=None, help="Optional labels_config.json for nice names")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default=None, help="Output CSV (default: alongside input, *_preds.csv)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    clf = joblib.load(model_dir/"model.pkl")
    vec = joblib.load(model_dir/"vectorizer.pkl")
    enc = joblib.load(model_dir/"encoder.pkl")

    df = pd.read_csv(args.csv)
    X = build_features(df, vec, enc)

    proba = clf.predict_proba(X)
    n_classes = proba.shape[1]
    names = load_cluster_names(Path(args.labels_config) if args.labels_config else None, n_classes)
    top_idx, top_vals = topk_from_proba(proba, k=args.topk)

    # Attach predictions
    out = df.copy()
    out["pred_top1_id"]   = top_idx[:,0]
    out["pred_top1_name"] = [names[i] for i in top_idx[:,0]]
    out["pred_top1_p"]    = top_vals[:,0]
    if args.topk >= 3:
        out["pred_top2_id"]   = top_idx[:,1]
        out["pred_top2_name"] = [names[i] for i in top_idx[:,1]]
        out["pred_top2_p"]    = top_vals[:,1]
        out["pred_top3_id"]   = top_idx[:,2]
        out["pred_top3_name"] = [names[i] for i in top_idx[:,2]]
        out["pred_top3_p"]    = top_vals[:,2]

    out_path = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_preds.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
