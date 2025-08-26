#!/usr/bin/env python3
"""
auto_suggest_tags.py  (empty-safe)

Mine your Shadow-Insta dataset and propose multi-label tagging rules.
It inspects accounts, hashtags, and caption words per cluster and
emits a JSON file with suggested:
  - account_to_tags
  - hashtag_to_tags
  - keyword_to_tags

USAGE
-----
python auto_suggest_tags.py ^
  --inferred "C:/Users/you/shadow_insta_k6/inferred_with_tags.csv" ^
  --out_dir "C:/Users/you/shadow_insta_k6" ^
  --config "C:/Users/you/labels_config.json" ^
  --min_count 10
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- helpers ----------------

STOP_HASHTAGS = {
    "", "reels", "viral", "insta", "instagram", "reel", "explore", "fyp",
    "follow", "like", "love", "fun", "funny", "tiktok"
}
STOP_WORDS_EXTRA = {
    "http", "https", "www", "com", "amp", "insta", "instagram",
    "please", "thanks", "video", "reel", "link", "bio"
}

ACCOUNT_HARDMAP = [
    (re.compile(r"meme", re.I), ["memes"]),
    (re.compile(r"troll", re.I), ["trolls", "satire"]),
    (re.compile(r"news?", re.I), ["news"]),
    (re.compile(r"(film|movie|cinema|tollywood)", re.I), ["movies", "tollywood"]),
    (re.compile(r"music|song", re.I), ["music"]),
]

WORD_HARDMAP = [
    (re.compile(r"(film|movie|cinema|tollywood)s?", re.I), ["movies", "tollywood"]),
    (re.compile(r"trailer|teaser", re.I), ["film_promo"]),
    (re.compile(r"song|music", re.I), ["music"]),
    (re.compile(r"review", re.I), ["reviews"]),
    (re.compile(r"meme", re.I), ["memes"]),
    (re.compile(r"troll|roast|satire", re.I), ["trolls", "satire"]),
    (re.compile(r"nri|desi abroad", re.I), ["nri_relatable"]),
    (re.compile(r"cricket|ipl|wc|odi|t20", re.I), ["sports", "cricket"]),
]

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{2,}\b"

def norm_text(s: str) -> str:
    return (s or "").strip().lower()

def split_hashtags(s: str):
    toks = re.split(r"[^\w#]+", norm_text(s))
    out = []
    for t in toks:
        if not t:
            continue
        t = t.lstrip("#")
        if not t or t in STOP_HASHTAGS:
            continue
        out.append(t)
    return out

def load_existing_config(path: Path | None):
    if not path or not path.exists():
        return {
            "cluster_to_tags": {},
            "account_to_tags": {},
            "hashtag_to_tags": {},
            "keyword_to_tags": {},
            "regex_to_tags": [],
        }
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["account_to_tags"]  = {k.lower(): v for k, v in cfg.get("account_to_tags", {}).items()}
    cfg["hashtag_to_tags"]  = {k.lower(): v for k, v in cfg.get("hashtag_to_tags", {}).items()}
    cfg["keyword_to_tags"]  = {k.lower(): v for k, v in cfg.get("keyword_to_tags", {}).items()}
    return cfg

def suggest_from_account(name: str):
    sug = set()
    for rx, tags in ACCOUNT_HARDMAP:
        if rx.search(name):
            sug.update(tags)
    if "meme" in name:
        sug.add("memes")
    if "troll" in name:
        sug.update(["trolls", "satire"])
    if any(k in name for k in ["film", "movie", "cinema", "tollywood"]):
        sug.update(["movies", "tollywood"])
    return sorted(sug)

def suggest_from_word(word: str):
    out = set()
    for rx, tags in WORD_HARDMAP:
        if rx.search(word):
            out.update(tags)
    return sorted(out)

# ---------------- mining ----------------

def mine(
    in_csv: Path,
    out_dir: Path,
    cfg_path: Path | None,
    top_accounts: int = 25,
    top_hashtags: int = 40,
    top_words: int = 50,
    min_count: int = 1
):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    for c in ["account", "hashtags", "caption", "category"]:
        if c not in df.columns:
            df[c] = ""
    df["account"]  = df["account"].fillna("").astype(str).str.lower().str.strip()
    df["hashtags"] = df["hashtags"].fillna("").astype(str)
    df["caption"]  = df["caption"].fillna("").astype(str)
    if df["category"].isna().any():
        df = df[df["category"].notna()].copy()
    df["category"] = df["category"].astype(int)

    existing = load_existing_config(cfg_path)

    # 1) Top accounts per category
    rows_accounts = []
    for cat, g in df.groupby("category"):
        vc = g["account"].value_counts()
        for acct, cnt in vc.items():
            if not acct or acct == "(unknown)":
                continue
            rows_accounts.append({"category": cat, "account": acct, "count": int(cnt)})

    top_accounts_df = pd.DataFrame(rows_accounts, columns=["category", "account", "count"])
    if not top_accounts_df.empty:
        if min_count > 1:
            top_accounts_df = top_accounts_df[top_accounts_df["count"] >= min_count]
        top_accounts_df = top_accounts_df.sort_values(["category", "count"], ascending=[True, False])
        (out_dir / "top_accounts_by_cat.csv").write_text(
            top_accounts_df.to_csv(index=False), encoding="utf-8"
        )

    # 2) Top hashtags per category
    rows_hashtags = []
    for cat, g in df.groupby("category"):
        counter = Counter()
        for s in g["hashtags"].tolist():
            counter.update(split_hashtags(s))
        # take top N first, then apply min_count filter below
        for tag, cnt in counter.most_common(top_hashtags):
            rows_hashtags.append({"category": cat, "hashtag": tag, "count": int(cnt)})

    top_hashtags_df = pd.DataFrame(rows_hashtags, columns=["category", "hashtag", "count"])
    if not top_hashtags_df.empty:
        if min_count > 1:
            top_hashtags_df = top_hashtags_df[top_hashtags_df["count"] >= min_count]
        top_hashtags_df = top_hashtags_df.sort_values(["category", "count"], ascending=[True, False])
        top_hashtags_df.to_csv(out_dir / "top_hashtags_by_cat.csv", index=False)

    # 3) Top caption words per category
    rows_words = []
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, stop_words="english")
    for cat, g in df.groupby("category"):
        try:
            X = vec.fit_transform(g["caption"].astype(str))
        except ValueError:
            continue
        vocab  = vec.get_feature_names_out()
        counts = X.sum(axis=0).A1
        pairs = [(w, int(c)) for (w, c) in zip(vocab, counts) if w not in STOP_WORDS_EXTRA]
        pairs.sort(key=lambda t: t[1], reverse=True)
        for w, c in pairs[:top_words]:
            rows_words.append({"category": cat, "word": w, "count": int(c)})

    top_words_df = pd.DataFrame(rows_words, columns=["category", "word", "count"])
    if not top_words_df.empty:
        if min_count > 1:
            top_words_df = top_words_df[top_words_df["count"] >= min_count]
        top_words_df = top_words_df.sort_values(["category", "count"], ascending=[True, False])
        top_words_df.to_csv(out_dir / "top_words_by_cat.csv", index=False)

    # -------- Build suggestions JSON --------
    account_to_tags = defaultdict(set)
    hashtag_to_tags = defaultdict(set)
    keyword_to_tags = defaultdict(set)

    if not top_accounts_df.empty:
        for _, row in top_accounts_df.iterrows():
            acct = row["account"]
            if acct in existing["account_to_tags"]:
                continue
            sug = suggest_from_account(acct)
            if not sug:
                sug = [acct]  # fallback: keep recognizable
            for t in sug:
                account_to_tags[acct].add(t)

    if not top_hashtags_df.empty:
        for _, row in top_hashtags_df.iterrows():
            h = row["hashtag"]
            if h in existing["hashtag_to_tags"]:
                continue
            sugs = {h}
            if "meme" in h:
                sugs.add("memes")
            if any(k in h for k in ["film", "movie", "cinema", "tollywood"]):
                sugs.update(["movies", "tollywood"])
            if h in {"sad", "love", "emotional", "emotions"}:
                sugs.add("emotion")
            for t in sugs:
                hashtag_to_tags[h].add(t)

    if not top_words_df.empty:
        for _, row in top_words_df.iterrows():
            w = row["word"]
            if w in existing["keyword_to_tags"]:
                continue
            sugs = set(suggest_from_word(w))
            if not sugs and len(w) >= 4 and not w.isdigit():
                sugs.add(w)
            for t in sugs:
                keyword_to_tags[w].add(t)

    # sets → lists
    account_to_tags = {k: sorted(v) for k, v in account_to_tags.items()}
    hashtag_to_tags = {k: sorted(v) for k, v in hashtag_to_tags.items()}
    keyword_to_tags = {k: sorted(v) for k, v in keyword_to_tags.items()}

    suggestions = {
        "account_to_tags": account_to_tags,
        "hashtag_to_tags": hashtag_to_tags,
        "keyword_to_tags": keyword_to_tags
    }

    out_json = out_dir / "tag_suggestions.json"
    out_json.write_text(json.dumps(suggestions, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved (where available):")
    if not top_accounts_df.empty:
        print(" -", out_dir / "top_accounts_by_cat.csv")
    if not top_hashtags_df.empty:
        print(" -", out_dir / "top_hashtags_by_cat.csv")
    if not top_words_df.empty:
        print(" -", out_dir / "top_words_by_cat.csv")
    print(" -", out_json)
    print("\nNext: open tag_suggestions.json and merge what you like into labels_config.json")
    return suggestions

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inferred", required=True, help="Path to inferred_with_tags.csv OR inferred_categories.csv")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: alongside inferred)")
    ap.add_argument("--config", default=None, help="Existing labels_config.json (optional; used to skip duplicates)")
    ap.add_argument("--top_accounts", type=int, default=25)
    ap.add_argument("--top_hashtags", type=int, default=40)
    ap.add_argument("--top_words", type=int, default=50)
    ap.add_argument("--min_count", type=int, default=1,
                    help="Minimum frequency required for an account/hashtag/word to be suggested")
    args = ap.parse_args()

    inferred = Path(args.inferred)
    out_dir = Path(args.out_dir) if args.out_dir else inferred.parent
    cfg_path = Path(args.config) if args.config else None

    mine(
        inferred,
        out_dir,
        cfg_path,
        top_accounts=args.top_accounts,
        top_hashtags=args.top_hashtags,
        top_words=args.top_words,
        min_count=args.min_count,
    )

if __name__ == "__main__":
    main()
