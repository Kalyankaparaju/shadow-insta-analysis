#!/usr/bin/env python3
"""
Build a unified interaction log (CSV) from your Instagram export.

Usage:
  python build_reels_log.py <EXPORT_ROOT_DIR> --out reels_log.csv

Notes:
- Reads only file names you have (skips missing safely).
- Produces columns: timestamp, interaction, content_type, content_id,
  account, caption, hashtags, source_file
- We DO NOT try to hit 100% schema parity (IG changes formats). Instead,
  we defensively extract common fields if present.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import csv

def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

def to_iso(ts):
    # Many IG exports give epoch seconds or ISO strings; try both.
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            return None
    if isinstance(ts, str):
        try:
            # If it's already ISO-ish, return as-is
            # Otherwise try parse common formats
            return ts
        except Exception:
            return None
    return None

def row(
    timestamp=None,
    interaction=None,
    content_type=None,
    content_id=None,
    account=None,
    caption=None,
    hashtags=None,
    source_file=None,
):
    return {
        "timestamp": timestamp,
        "interaction": interaction,           # watched / liked / saved / commented / story_like / ad_view / ad_click / searched
        "content_type": content_type,         # reel / video / post / story / ad / search
        "content_id": content_id,
        "account": account,
        "caption": caption,
        "hashtags": hashtags,
        "source_file": source_file,
    }

def parse_videos_watched(root, out_rows):
    p = root / "ads_information" / "ads_and_topics" / "videos_watched.json"
    data = load_json(p)
    if not data: return
    # Expect a list of items with timestamp and maybe title/url/ad_id
    for item in data if isinstance(data, list) else data.get("videos_watched", []):
        ts = to_iso(item.get("timestamp") or item.get("time"))
        out_rows.append(row(
            timestamp=ts,
            interaction="watched",
            content_type="video",
            content_id=item.get("id") or item.get("ad_id") or item.get("url"),
            account=item.get("author") or item.get("page_name"),
            caption=item.get("title") or item.get("description"),
            source_file=str(p.relative_to(root))
        ))

def parse_subscriptions_reels(root, out_rows):
    p = root / "your_instagram_activity" / "subscriptions" / "reels.json"
    data = load_json(p)
    if not data: return
    items = data if isinstance(data, list) else data.get("reels", [])
    for item in items:
        ts = to_iso(item.get("timestamp") or item.get("time"))
        out_rows.append(row(
            timestamp=ts,
            interaction="watched",
            content_type="reel",
            content_id=item.get("media_id") or item.get("id") or item.get("url"),
            account=item.get("author") or item.get("username"),
            caption=item.get("caption"),
            hashtags=_extract_hashtags(item),
            source_file=str(p.relative_to(root))
        ))

def parse_liked_posts(root, out_rows):
    p = root / "your_instagram_activity" / "likes" / "liked_posts.json"
    data = load_json(p)
    if not data: return
    items = data if isinstance(data, list) else data.get("likes_media_likes", []) or data.get("likes", [])
    for item in items:
        ts = to_iso(item.get("timestamp") or item.get("string_list_data", [{}])[0].get("timestamp"))
        out_rows.append(row(
            timestamp=ts,
            interaction="liked",
            content_type=item.get("media_type") or "post",
            content_id=item.get("media_id") or item.get("id") or _from_string_list(item, "href"),
            account=item.get("owner") or item.get("title") or item.get("username"),
            caption=item.get("caption_text") or item.get("caption"),
            hashtags=_extract_hashtags(item),
            source_file=str(p.relative_to(root))
        ))

def parse_liked_comments(root, out_rows):
    p = root / "your_instagram_activity" / "likes" / "liked_comments.json"
    data = load_json(p)
    if not data: return
    items = data if isinstance(data, list) else data.get("likes_comment_likes", []) or data.get("likes", [])
    for item in items:
        ts = to_iso(item.get("timestamp"))
        out_rows.append(row(
            timestamp=ts,
            interaction="liked",
            content_type="comment",
            content_id=item.get("comment_id") or item.get("id"),
            account=item.get("author") or item.get("username"),
            caption=item.get("text"),
            source_file=str(p.relative_to(root))
        ))

def parse_saved_posts(root, out_rows):
    p = root / "your_instagram_activity" / "saved" / "saved_posts.json"
    data = load_json(p)
    if not data: return
    items = data if isinstance(data, list) else data.get("saved_saved_media", []) or data.get("saved", [])
    for item in items:
        ts = to_iso(item.get("timestamp"))
        out_rows.append(row(
            timestamp=ts,
            interaction="saved",
            content_type=item.get("media_type") or "post",
            content_id=item.get("media_id") or item.get("id") or _from_string_list(item, "href"),
            account=item.get("owner") or item.get("username"),
            caption=item.get("caption_text") or item.get("caption"),
            hashtags=_extract_hashtags(item),
            source_file=str(p.relative_to(root))
        ))

def parse_reels_comments(root, out_rows):
    p = root / "your_instagram_activity" / "comments" / "reels_comments.json"
    data = load_json(p)
    if not data: return
    items = data if isinstance(data, list) else data.get("reels_comments", []) or data.get("comments", [])
    for item in items:
        ts = to_iso(item.get("timestamp"))
        out_rows.append(row(
            timestamp=ts,
            interaction="commented",
            content_type="reel",
            content_id=item.get("media_id") or item.get("reel_id") or item.get("id"),
            account=item.get("author") or item.get("username"),
            caption=item.get("text"),
            source_file=str(p.relative_to(root))
        ))

def parse_story_likes(root, out_rows):
    p = root / "your_instagram_activity" / "story_interactions" / "story_likes.json"
    data = load_json(p)
    if not data: return
    items = data if isinstance(data, list) else data.get("story_likes", []) or data.get("likes", [])
    for item in items:
        ts = to_iso(item.get("timestamp"))
        out_rows.append(row(
            timestamp=ts,
            interaction="story_like",
            content_type="story",
            content_id=item.get("story_id") or item.get("id"),
            account=item.get("author") or item.get("username"),
            caption=item.get("text"),
            source_file=str(p.relative_to(root))
        ))

def parse_recent_searches(root, out_rows):
    base = root / "logged_information" / "recent_searches"
    for name in ("profile_searches.json", "tag_searches.json", "word_or_phrase_searches.json"):
        p = base / name
        data = load_json(p)
        if not data: continue
        items = data if isinstance(data, list) else data.get("searches", [])
        for item in items:
            ts = to_iso(item.get("timestamp") or item.get("time"))
            q = item.get("query") or item.get("search") or item.get("string_map_data", {}).get("Search", {}).get("value")
            out_rows.append(row(
                timestamp=ts,
                interaction="searched",
                content_type="search",
                content_id=None,
                account=None,
                caption=q,
                source_file=str(p.relative_to(root))
            ))

def _from_string_list(item, key):
    try:
        arr = item.get("string_list_data", [])
        if arr and isinstance(arr, list):
            return arr[0].get(key)
    except Exception:
        pass
    return None

def _extract_hashtags(item):
    # very light attempt: look for "hashtags" field or parse from caption_text
    tags = item.get("hashtags")
    if isinstance(tags, list):
        return " ".join([str(t) for t in tags])
    caption = item.get("caption_text") or item.get("caption")
    if isinstance(caption, str) and "#" in caption:
        # crude extraction
        parts = [w for w in caption.split() if w.startswith("#")]
        return " ".join(parts) if parts else None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Path to the unzipped Instagram export")
    ap.add_argument("--out", type=str, default="reels_log.csv", help="Output CSV")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    rows = []
    parse_videos_watched(root, rows)
    parse_subscriptions_reels(root, rows)
    parse_liked_posts(root, rows)
    parse_liked_comments(root, rows)
    parse_saved_posts(root, rows)
    parse_reels_comments(root, rows)
    parse_story_likes(root, rows)
    parse_recent_searches(root, rows)

    # Sort by timestamp (None at end)
    def ts_key(r):
        return r["timestamp"] or "9999-12-31T23:59:59Z"
    rows.sort(key=ts_key)

    # Write CSV
    out_path = Path(args.out).resolve()
    fieldnames = ["timestamp","interaction","content_type","content_id","account","caption","hashtags","source_file"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
