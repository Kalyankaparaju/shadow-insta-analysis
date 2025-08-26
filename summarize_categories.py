#!/usr/bin/env python3
"""
Summarize Shadow-Insta clusters.

Usage:
  python summarize_categories.py --inferred "C:/Users/kvvsa/shadow_insta_k6/inferred_categories.csv" --out_dir "C:/Users/kvvsa/shadow_insta_k6" --top_n 15
"""

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inferred", required=True, help="Path to inferred_categories.csv")
    ap.add_argument("--out_dir", default=None, help="Dir to write summaries (default = same folder as inferred)")
    ap.add_argument("--top_n", type=int, default=15, help="How many top accounts per category to show/save")
    args = ap.parse_args()

    inf_path = Path(args.inferred)
    out_dir = Path(args.out_dir) if args.out_dir else inf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load & normalize columns ---
    df = pd.read_csv(inf_path)
    df.columns = [c.strip() for c in df.columns]  # remove stray spaces

    for col in ["account", "content_id", "category", "content_type", "caption", "hashtags"]:
        if col not in df.columns:
            df[col] = ""

    df = df[df["category"].notna()].copy()
    df["category"] = df["category"].astype(int)
    df["account"] = df["account"].fillna("").astype(str).str.strip()
    df.loc[df["account"].eq(""), "account"] = "(unknown)"

    # --- High-level counts per category ---
    by_cat = (
        df.groupby("category")
          .agg(
              rows=("category", "size"),
              unique_accounts=("account", "nunique"),
              content_types=("content_type", lambda s: ", ".join(f"{k}:{v}" for k, v in s.value_counts().head(5).items()))
          )
          .reset_index()
          .sort_values("rows", ascending=False)
    )

    # --- Top accounts per category (robust across pandas versions) ---
    top_blocks = []
    for cat, g in df.groupby("category", sort=True):
        vc = g["account"].value_counts()
        top = vc.reset_index(name="count")                # pandas >= 1.1
        if "index" in top.columns:
            top = top.rename(columns={"index": "account"})  # older pandas
        # Keep only ["account","count"] in right order
        top = top[["account", "count"]]
        top.insert(0, "category", cat)                    # add category column at front
        top_blocks.append(top)

        # Save example rows per category (handy for manual labeling)
        g[["account", "content_id", "caption", "hashtags", "content_type"]].head(100) \
            .to_csv(out_dir / f"category_{cat:02d}_examples.csv", index=False)

    top_accounts_all = pd.concat(top_blocks, ignore_index=True)
    top_accounts_all = (
        top_accounts_all.sort_values(["category", "count"], ascending=[True, False])
                        .groupby("category", as_index=False)
                        .head(args.top_n)
    )

    # --- Draft labels from top accounts ---
    label_drafts = (
        top_accounts_all.groupby("category", as_index=False)
                        .apply(lambda x: "; ".join(f"{a}({c})" for a, c in zip(x["account"], x["count"])))
                        .rename(columns={0: "top_accounts"})
    )

    summary = by_cat.merge(label_drafts, on="category", how="left")

    # --- Save outputs ---
    summary_path = out_dir / "category_summary.csv"
    top_accounts_path = out_dir / "category_top_accounts.csv"
    summary.to_csv(summary_path, index=False)
    top_accounts_all.to_csv(top_accounts_path, index=False)

    # --- Pretty print to console ---
    print("\n=== Category summary (sorted by rows) ===")
    # limit very long strings in console
    with pd.option_context("display.max_colwidth", 60):
        print(summary.to_string(index=False))
    print(f"\nSaved:\n - {summary_path}\n - {top_accounts_path}\n - category_XX_examples.csv (per category)")

if __name__ == "__main__":
    main()
