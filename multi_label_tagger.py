#!/usr/bin/env python3
"""
multi_label_tagger.py
Assign MULTIPLE tags (labels) to every row using a rule config.
Works on the full inferred dataset.

Usage:
  python multi_label_tagger.py --inferred "C:/Users/you/shadow_insta_k6/inferred_categories.csv" --config "C:/Users/you/labels_config.json" --out_dir "C:/Users/you/shadow_insta_k6"
"""

import argparse, json, re
from pathlib import Path
import pandas as pd

def tokenize_hashtags(s: str):
    if not isinstance(s, str):
        return []
    # split on non-word, strip '#', lowercase
    toks = re.split(r"[^\w]+", s.lower())
    return [t.lstrip("#") for t in toks if t]

def load_config(p: Path):
    cfg = json.loads(p.read_text(encoding="utf-8"))
    # normalize keys for case-insensitive matching
    cfg["cluster_to_tags"] = {int(k): v for k, v in cfg.get("cluster_to_tags", {}).items()}
    cfg["account_to_tags"] = {k.lower(): v for k, v in cfg.get("account_to_tags", {}).items()}
    cfg["hashtag_to_tags"] = {k.lower(): v for k, v in cfg.get("hashtag_to_tags", {}).items()}
    cfg["keyword_to_tags"] = {k.lower(): v for k, v in cfg.get("keyword_to_tags", {}).items()}
    rx = []
    for item in cfg.get("regex_to_tags", []):
        rx.append((re.compile(item["pattern"], flags=re.I), item["tags"]))
    cfg["regex_to_tags"] = rx
    cfg["max_tags_per_post"] = int(cfg.get("max_tags_per_post", 8))
    return cfg

def tags_from_row(row, cfg):
    tags = set()

    # 1) cluster-based tags
    cat = row.get("category")
    if pd.notna(cat):
        cat_int = int(cat)
        tags.update(cfg["cluster_to_tags"].get(cat_int, []))

    # 2) account-based
    acct = str(row.get("account") or "").lower()
    if acct:
        tags.update(cfg["account_to_tags"].get(acct, []))

    # 3) hashtag-based
    hashtags = str(row.get("hashtags") or "")
    for h in tokenize_hashtags(hashtags):
        tags.update(cfg["hashtag_to_tags"].get(h, []))

    # 4) keyword-based (caption)
    caption = str(row.get("caption") or "").lower()
    for kw, tgs in cfg["keyword_to_tags"].items():
        if kw in caption:
            tags.update(tgs)

    # 5) regex-based (caption+hashtags)
    text = f"{caption} {hashtags}".strip()
    for rx, tgs in cfg["regex_to_tags"]:
        if rx.search(text):
            tags.update(tgs)

    # clean
    tags = [t for t in dict.fromkeys(tags)]  # dedupe, keep order
    if len(tags) > cfg["max_tags_per_post"]:
        tags = tags[:cfg["max_tags_per_post"]]
    return tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inferred", required=True, help="Path to inferred_categories.csv")
    ap.add_argument("--config", required=True, help="Path to labels_config.json")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: same as inferred)")
    args = ap.parse_args()

    inferred = Path(args.inferred)
    out_dir = Path(args.out_dir) if args.out_dir else inferred.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(args.config))

    df = pd.read_csv(inferred)
    # ensure columns exist
    for c in ["account","caption","hashtags","category"]:
        if c not in df.columns: df[c] = ""

    # assign tags to every row (FULL DATASET)
    df["tags_list"] = df.apply(lambda r: tags_from_row(r, cfg), axis=1)
    df["tags"] = df["tags_list"].apply(lambda xs: ", ".join(xs))

    # save full output
    out_full = out_dir / "inferred_with_tags.csv"
    df.to_csv(out_full, index=False, encoding="utf-8")

    # tag counts
    all_tags = []
    for xs in df["tags_list"]:
        all_tags.extend(xs)
    tag_counts = pd.Series(all_tags).value_counts().reset_index()
    tag_counts.columns = ["tag", "rows"]
    tag_counts.to_csv(out_dir / "tag_counts.csv", index=False, encoding="utf-8")

    # cluster ↔ tag co-occurrence (handy to refine rules)
    if "category" in df.columns and df["category"].notna().any():
        exploded = df.explode("tags_list")
        crosstab = (exploded.dropna(subset=["tags_list"])
                    .groupby(["category","tags_list"]).size()
                    .reset_index(name="rows")
                    .sort_values(["category","rows"], ascending=[True,False]))
        crosstab.to_csv(out_dir / "cluster_tag_crosstab.csv", index=False, encoding="utf-8")

    print("Saved:")
    print(" -", out_full)
    print(" -", out_dir / "tag_counts.csv")
    if (out_dir / "cluster_tag_crosstab.csv").exists():
        print(" -", out_dir / "cluster_tag_crosstab.csv")

if __name__ == "__main__":
    main()
