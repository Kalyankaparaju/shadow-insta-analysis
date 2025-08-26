#!/usr/bin/env python3
"""
Shadow-Insta — Beautiful multi-figure report (v2, extended)

Usage (Windows):
  python report_shadow_insta_v2.py ^
    --inferred "C:/Users/you/shadow_insta_k6/inferred_categories.csv" ^
    --with_tags "C:/Users/you/shadow_insta_k6/inferred_with_tags.csv" ^
    --preds     "C:/Users/you/shadow_insta_k6/reels_log_preds.csv" ^
    --markov    "C:/Users/you/shadow_insta_k6/markov_transitions.csv" ^
    --model_dir "C:/Users/you/shadow_insta_k6" ^
    --out       "C:/Users/you/shadow_insta_k6/report_v2"

Only --inferred and --out are required. Others are optional; skipped plots are handled gracefully.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

# -------------------- global style --------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})
sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("tab10")
CMAP = "Spectral_r"
DPI = 200

# -------------------- small utils --------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_fig(out_dir: Path, name: str) -> Path:
    ensure_dir(out_dir)
    return out_dir / f"{name}.png"

def save_and_close(fpath: Path):
    plt.tight_layout()
    plt.savefig(fpath, dpi=DPI, bbox_inches="tight")
    plt.close()

def _parse_dt_any(s: pd.Series) -> pd.Series:
    """Parse any datetime-like to tz-aware UTC then make tz-naive for clean merges/plots."""
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None)

def to_month_from_series(ts: pd.Series) -> pd.Series:
    ts = _parse_dt_any(ts)
    return ts.dt.to_period("M").astype(str)

def safe_read_csv(path: Path | None, parse_ts: bool = True) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        if parse_ts:
            return pd.read_csv(p, parse_dates=["timestamp"])
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p)
        except Exception:
            return None

# -------------------- I/O core --------------------
def load_core(inferred_csv: Path, with_tags_csv: Path | None, preds_csv: Path | None):
    df = safe_read_csv(inferred_csv, parse_ts=True)
    if df is None:
        raise FileNotFoundError(f"Could not read inferred categories CSV: {inferred_csv}")
    if "timestamp" in df.columns:
        df["timestamp"] = _parse_dt_any(df["timestamp"])

    wt = None
    if with_tags_csv:
        wt = safe_read_csv(with_tags_csv, parse_ts=True)
        if wt is not None and "timestamp" in wt.columns:
            wt["timestamp"] = _parse_dt_any(wt["timestamp"])
        if wt is not None and "tags" in wt.columns:
            wt_ = wt[["content_id", "timestamp", "tags"]].drop_duplicates(["content_id", "timestamp"])
            df = df.merge(wt_, on=["content_id", "timestamp"], how="left")

    preds = None
    if preds_csv:
        preds = safe_read_csv(preds_csv, parse_ts=True)
        if preds is not None and "timestamp" in preds.columns:
            preds["timestamp"] = _parse_dt_any(preds["timestamp"])

    return df, wt, preds

# -------------------- predictions: harmonize/join --------------------
def harmonize_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"ts", "time", "created_at", "datetime"}:
            ren[c] = "timestamp"
        if lc in {"pred", "predicted", "yhat", "y_hat", "top1_pred"}:
            ren[c] = "pred_top1"
    if ren:
        df = df.rename(columns=ren)
    if "timestamp" in df.columns:
        df["timestamp"] = _parse_dt_any(df["timestamp"])
    return df

def attach_true_category(preds: pd.DataFrame, inferred_df: pd.DataFrame) -> pd.DataFrame:
    if preds is None or preds.empty:
        return preds
    preds = preds.copy()
    inferred_df = inferred_df.copy()
    if "timestamp" in inferred_df.columns:
        inferred_df["timestamp"] = _parse_dt_any(inferred_df["timestamp"])
    if "timestamp" in preds.columns:
        preds["timestamp"] = _parse_dt_any(preds["timestamp"])

    join_keys = []
    if "content_id" in preds.columns and "content_id" in inferred_df.columns:
        join_keys.append("content_id")
    if "timestamp" in preds.columns and "timestamp" in inferred_df.columns:
        join_keys.append("timestamp")

    if join_keys:
        keep_cols = list(dict.fromkeys(join_keys + ["category"]))
        merged = preds.merge(inferred_df[keep_cols], on=join_keys, how="left", suffixes=("", "_true"))
    else:
        merged = preds.copy()

    need_asof = (
        ("category" not in merged.columns or merged["category"].isna().all())
        and {"content_id", "timestamp"}.issubset(preds.columns)
        and {"content_id", "timestamp"}.issubset(inferred_df.columns)
    )
    if need_asof:
        right = inferred_df.sort_values("timestamp")[["content_id", "timestamp", "category"]]
        parts = []
        for cid, left_grp in preds.sort_values("timestamp").groupby("content_id"):
            right_grp = right[right["content_id"] == cid]
            if right_grp.empty:
                l = left_grp.copy()
                l["category"] = np.nan
                parts.append(l)
                continue
            l = pd.merge_asof(
                left_grp.sort_values("timestamp"),
                right_grp.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("2H"),
            )
            parts.append(l)
        asof_merged = pd.concat(parts, axis=0).sort_index()
        if "category" not in merged.columns:
            merged["category"] = asof_merged["category"]
        else:
            mask = merged["category"].isna()
            merged.loc[mask, "category"] = asof_merged.loc[mask, "category"]
    return merged

# -------------------- core plots --------------------
def plot_category_hist(df, out_dir):
    ax = df["category"].value_counts().sort_index().plot(kind="bar", color=PALETTE)
    ax.set_title("Category distribution (all data)")
    ax.set_xlabel("Category"); ax.set_ylabel("Rows")
    f = make_fig(out_dir, "01_category_hist"); save_and_close(f)
    return ("Category distribution", f.name)

def plot_rows_over_time(df, out_dir):
    daily = df.set_index("timestamp").resample("D").size()
    ax = daily.plot()
    ax.set_title("Rows per day"); ax.set_ylabel("Rows"); ax.set_xlabel("Date")
    f = make_fig(out_dir, "02_rows_per_day"); save_and_close(f)
    return ("Rows per day", f.name)

def plot_category_share_by_month(df, out_dir):
    g = df.assign(month=to_month_from_series(df["timestamp"])) \
          .groupby(["month", "category"]).size().unstack(fill_value=0)
    share = g.div(g.sum(axis=1), axis=0)
    share.plot.area(colormap="tab20")
    plt.title("Category share over time (monthly)")
    plt.xlabel("Month"); plt.ylabel("Share")
    f = make_fig(out_dir, "03_category_share_month"); save_and_close(f)
    return ("Category share by month", f.name)

def plot_activity_heatmap(df, out_dir):
    d = df.copy()
    ts = _parse_dt_any(d["timestamp"])
    d["dow"] = ts.dt.dayofweek
    d["hod"] = ts.dt.hour
    z = d.pivot_table(index="dow", columns="hod", values="content_id", aggfunc="count", fill_value=0)
    sns.heatmap(z, cmap="Blues")
    plt.title("Activity heatmap (weekday × hour)")
    plt.xlabel("Hour"); plt.ylabel("Weekday (0=Mon)")
    f = make_fig(out_dir, "04_activity_heatmap"); save_and_close(f)
    return ("Activity heatmap", f.name)

def plot_top_accounts(df, out_dir, n=20):
    top = df["account"].value_counts().head(n)
    sns.barplot(x=top.values, y=top.index, orient="h")
    plt.title(f"Top {n} accounts"); plt.xlabel("Rows"); plt.ylabel("Account")
    f = make_fig(out_dir, "05_top_accounts"); save_and_close(f)
    return ("Top accounts", f.name)

def _tokenize_hashtags(series):
    toks = []
    for s in series.fillna("").astype(str):
        for t in s.replace("#", " ").split():
            t = t.strip().lower()
            if t and t not in {"reels", "viral", "insta", "instagram"}:
                toks.append(t)
    return toks

def plot_top_hashtags(df, out_dir, n=20):
    if "hashtags" not in df.columns:
        return None
    toks = _tokenize_hashtags(df["hashtags"])
    if not toks:
        return None
    top = pd.Series(toks).value_counts().head(n)
    sns.barplot(x=top.values, y=top.index, orient="h")
    plt.title(f"Top {n} hashtags"); plt.xlabel("Rows"); plt.ylabel("Hashtag")
    f = make_fig(out_dir, "06_top_hashtags"); save_and_close(f)
    return ("Top hashtags", f.name)

# -------------------- per-category expansions (adds many figs) --------------------
def plot_top_accounts_by_cat(df, out_dir, n=10):
    titles = []
    cats = sorted(pd.unique(df["category"].dropna().astype(int)))
    for c in cats:
        subset = df[df["category"] == c]
        vc = subset["account"].value_counts().head(n)
        if vc.empty: 
            continue
        sns.barplot(x=vc.values, y=vc.index, orient="h")
        plt.title(f"Top {n} accounts – cat {c}")
        plt.xlabel("Rows"); plt.ylabel("Account")
        f = make_fig(out_dir, f"acct_cat{c}")
        save_and_close(f)
        titles.append((f"Top accounts (cat {c})", f.name))
    return titles

def plot_top_hashtags_by_cat(df, out_dir, n=10):
    if "hashtags" not in df.columns:
        return []
    titles = []
    cats = sorted(pd.unique(df["category"].dropna().astype(int)))
    for c in cats:
        subset = df[df["category"] == c]
        toks = _tokenize_hashtags(subset["hashtags"])
        if not toks:
            continue
        vc = pd.Series(toks).value_counts().head(n)
        sns.barplot(x=vc.values, y=vc.index, orient="h")
        plt.title(f"Top {n} hashtags – cat {c}")
        plt.xlabel("Rows"); plt.ylabel("Hashtag")
        f = make_fig(out_dir, f"hash_cat{c}")
        save_and_close(f)
        titles.append((f"Top hashtags (cat {c})", f.name))
    return titles

def plot_hourly_activity_by_cat(df, out_dir):
    titles = []
    d = df.copy()
    d["hod"] = _parse_dt_any(d["timestamp"]).dt.hour
    cats = sorted(pd.unique(d["category"].dropna().astype(int)))
    for c in cats:
        g = d[d["category"] == c].groupby("hod").size()
        if g.empty:
            continue
        g.plot(marker="o")
        plt.title(f"Hourly activity – cat {c}")
        plt.xlabel("Hour"); plt.ylabel("Rows")
        plt.xticks(range(0, 24, 2))
        f = make_fig(out_dir, f"hour_cat{c}")
        save_and_close(f)
        titles.append((f"Hourly activity (cat {c})", f.name))
    return titles

# -------------------- Markov --------------------
def plot_markov_heatmap(markov_csv: Path | None, out_dir):
    if not markov_csv:
        return None
    p = Path(markov_csv)
    if not p.exists():
        return None
    try:
        T = pd.read_csv(p).values
    except Exception:
        return None
    sns.heatmap(T, cmap="viridis", annot=False)
    plt.title("Markov transition probabilities"); plt.xlabel("Next category"); plt.ylabel("Current category")
    f = make_fig(out_dir, "07_markov_heatmap"); save_and_close(f)
    return ("Markov transitions", f.name)

def plot_markov_ranklines(markov_csv: Path | None, out_dir):
    if not markov_csv:
        return None
    p = Path(markov_csv)
    if not p.exists():
        return None
    try:
        T = pd.read_csv(p).values
    except Exception:
        return None
    plt.figure()
    for i in range(T.shape[0]):
        nxt = pd.Series(T[i, :]).sort_values(ascending=False).reset_index(drop=True)
        sns.lineplot(data=nxt, label=f"from {i}")
    plt.title("Next-category distribution by current")
    plt.xlabel("Ranked next-categories"); plt.ylabel("Probability")
    plt.legend(title="Current cat", bbox_to_anchor=(1.02, 1), loc="upper left")
    f = make_fig(out_dir, "07b_markov_ranklines"); save_and_close(f)
    return ("Next-category rank lines", f.name)

# -------------------- predictions/metrics --------------------
def _has_truth(preds: pd.DataFrame) -> bool:
    return preds is not None and "category" in preds.columns and preds["category"].notna().any()

def plot_confusion(preds, out_dir):
    if not _has_truth(preds) or "pred_top1" not in preds.columns:
        return None
    cats = sorted(pd.unique(pd.concat([preds["category"], preds["pred_top1"]]).dropna().astype(int)))
    cm = pd.crosstab(preds["category"], preds["pred_top1"]).reindex(index=cats, columns=cats, fill_value=0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion matrix (t → t+H)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    f = make_fig(out_dir, "08_confusion"); save_and_close(f)
    return ("Confusion matrix", f.name)

def plot_class_prf1(preds, out_dir):
    if not _has_truth(preds) or "pred_top1" not in preds.columns:
        return None
    cats = sorted(pd.unique(pd.concat([preds["category"], preds["pred_top1"]]).dropna().astype(int)))
    p, r, f1, _ = precision_recall_fscore_support(
        preds["category"], preds["pred_top1"], labels=cats, zero_division=0
    )
    dfm = pd.DataFrame({"precision": p, "recall": r, "f1": f1}, index=cats)
    dfm.plot(kind="bar")
    plt.title("Per-class Precision / Recall / F1")
    plt.xlabel("Category"); plt.ylabel("Score")
    f = make_fig(out_dir, "09_prf1"); save_and_close(f)
    return ("Per-class PRF1", f.name)

def plot_confidence_hist(preds, out_dir):
    if "proba_top1" not in preds.columns:
        return None
    preds["proba_top1"].plot(kind="hist", bins=20)
    plt.title("Predicted top-1 probability (confidence)")
    plt.xlabel("proba_top1")
    f = make_fig(out_dir, "10_confidence_hist"); save_and_close(f)
    return ("Confidence histogram", f.name)

def plot_rolling_accuracy(preds, out_dir, window=30):
    if not _has_truth(preds) or "pred_top1" not in preds.columns or "timestamp" not in preds.columns:
        return None
    s = (preds["category"] == preds["pred_top1"]).astype(int)
    s_roll = s.rolling(window, min_periods=5).mean()
    plt.plot(preds["timestamp"], s_roll)
    plt.title(f"Rolling accuracy ({window}-day)")
    plt.ylabel("Accuracy"); plt.xlabel("Date")
    f = make_fig(out_dir, "11_rolling_acc"); save_and_close(f)
    return ("Rolling accuracy", f.name)

def plot_acc_by_month(preds, out_dir):
    if not _has_truth(preds) or "pred_top1" not in preds.columns:
        return None
    acc = (preds["category"] == preds["pred_top1"]).astype(int)
    m = preds.assign(month=to_month_from_series(preds["timestamp"]), acc=acc) \
             .groupby("month")["acc"].mean()
    m.plot(marker="o")
    plt.title("Accuracy by month"); plt.ylabel("Accuracy"); plt.xlabel("Month")
    f = make_fig(out_dir, "12_acc_by_month"); save_and_close(f)
    return ("Accuracy by month", f.name)

def plot_errors_by_category(preds, out_dir):
    if not _has_truth(preds) or "pred_top1" not in preds.columns:
        return None
    err = preds[preds["category"] != preds["pred_top1"]]
    if err.empty:
        return None
    vc = err["category"].value_counts().sort_index()
    sns.barplot(x=vc.index, y=vc.values, palette=PALETTE)
    plt.title("Misclassified counts by true category")
    plt.xlabel("True category"); plt.ylabel("Errors")
    f = make_fig(out_dir, "12b_errors_by_cat"); save_and_close(f)
    return ("Errors by true category", f.name)

def plot_error_pairs_heatmap(preds, out_dir):
    if not _has_truth(preds) or "pred_top1" not in preds.columns:
        return None
    err = preds[preds["category"] != preds["pred_top1"]]
    if err.empty:
        return None
    z = pd.crosstab(err["category"], err["pred_top1"])
    sns.heatmap(z, cmap="Reds")
    plt.title("Misclassification pairs (true → predicted)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    f = make_fig(out_dir, "12c_error_pairs_heatmap"); save_and_close(f)
    return ("Misclassification pairs heatmap", f.name)

def plot_error_hour_heatmap(preds, out_dir):
    if not _has_truth(preds) or "pred_top1" not in preds.columns or "timestamp" not in preds.columns:
        return None
    d = preds.copy()
    d["hod"] = _parse_dt_any(d["timestamp"]).dt.hour
    d["is_error"] = (d["category"] != d["pred_top1"]).astype(int)
    z = d.pivot_table(index="hod", columns="category", values="is_error", aggfunc="mean", fill_value=0)
    sns.heatmap(z, cmap="magma")
    plt.title("Error rate by hour × true category")
    plt.xlabel("True category"); plt.ylabel("Hour")
    f = make_fig(out_dir, "12d_error_hour_heatmap"); save_and_close(f)
    return ("Error rate by hour (by true cat)", f.name)

# -------------------- model features (per class: + and −) --------------------
def plot_top_features(model_dir: Path | None, out_dir, topn=12):
    if not model_dir:
        return []
    try:
        clf = joblib.load(Path(model_dir) / "model.pkl")
        vec = joblib.load(Path(model_dir) / "vectorizer.pkl")
    except Exception:
        return []
    if not hasattr(clf, "coef_"):
        return []
    vocab = np.array(getattr(vec, "get_feature_names_out")())
    titles = []
    for c in range(clf.coef_.shape[0]):
        w = clf.coef_[c]
        # positive
        pos_idx = np.argsort(w)[-topn:]
        sns.barplot(x=w[pos_idx], y=vocab[pos_idx], orient="h")
        plt.title(f"Top + features (cat {c})")
        f = make_fig(out_dir, f"feat_pos_cat{c}"); save_and_close(f)
        titles.append((f"Top + features (cat {c})", f.name))
        # negative
        neg_idx = np.argsort(w)[:topn]
        sns.barplot(x=w[neg_idx], y=vocab[neg_idx], orient="h")
        plt.title(f"Top − features (cat {c})")
        f = make_fig(out_dir, f"feat_neg_cat{c}"); save_and_close(f)
        titles.append((f"Top − features (cat {c})", f.name))
    return titles

# -------------------- tags --------------------
def plot_tag_frequency(df, out_dir, n=30):
    if "tags" not in df.columns:
        return None
    tags = []
    for s in df["tags"].fillna(""):
        for t in [x.strip() for x in s.split(",") if x.strip()]:
            tags.append(t)
    if not tags:
        return None
    vc = pd.Series(tags).value_counts().head(n)
    sns.barplot(x=vc.values, y=vc.index, orient="h")
    plt.title(f"Top {n} tags"); plt.xlabel("Rows"); plt.ylabel("Tag")
    f = make_fig(out_dir, "14_tag_frequency"); save_and_close(f)
    return ("Top tags", f.name)

def plot_tag_category_heatmap(df, out_dir, top=20):
    if "tags" not in df.columns:
        return None
    d = df.copy()
    d["tags_list"] = d["tags"].fillna("").apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    d = d.explode("tags_list")
    if d["tags_list"].dropna().empty:
        return None
    top_tags = d["tags_list"].value_counts().head(top).index
    d = d[d["tags_list"].isin(top_tags)]
    if d.empty:
        return None
    z = d.pivot_table(index="tags_list", columns="category", values="content_id", aggfunc="count", fill_value=0)
    sns.heatmap(z, cmap="mako")
    plt.title("Tag × Category (top tags)")
    plt.xlabel("Category"); plt.ylabel("Tag")
    f = make_fig(out_dir, "15_tag_category_heatmap"); save_and_close(f)
    return ("Tag × Category", f.name)

def plot_tag_share_over_time(df, out_dir, top=8):
    if "tags" not in df.columns:
        return None
    d = df.copy()
    d["tags_list"] = d["tags"].fillna("").apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    d = d.explode("tags_list")
    if d["tags_list"].dropna().empty:
        return None
    top_tags = d["tags_list"].value_counts().head(top).index
    d = d[d["tags_list"].isin(top_tags)]
    if d.empty:
        return None
    g = d.assign(month=to_month_from_series(d["timestamp"])) \
        .groupby(["month", "tags_list"]).size().unstack(fill_value=0)
    share = g.div(g.sum(axis=1), axis=0)
    share.plot.area()
    plt.title("Tag share over time (top tags)")
    plt.xlabel("Month"); plt.ylabel("Share")
    f = make_fig(out_dir, "16_tag_share_time"); save_and_close(f)
    return ("Tag share over time", f.name)

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

def plot_roc_per_class(preds, out_dir):
    if "proba_top1" not in preds or not _has_truth(preds):
        return None
    y_true = preds["category"].astype(int)
    y_score = preds.get("proba_all")  # requires storing full prob matrix!
    if y_score is None:
        return None

    n_classes = y_score.shape[1]
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve((y_true==c).astype(int), y_score[:, c])
        plt.plot(fpr, tpr, label=f"cat {c} (AUC={auc(fpr,tpr):.2f})")
    plt.plot([0,1],[0,1],"k--")
    plt.title("ROC curves per class")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend()
    f = make_fig(out_dir,"roc_per_class"); save_and_close(f)
    return ("ROC per class", f.name)

def plot_pr_per_class(preds, out_dir):
    if "proba_top1" not in preds or not _has_truth(preds):
        return None
    y_true = preds["category"].astype(int)
    y_score = preds.get("proba_all")
    if y_score is None:
        return None
    n_classes = y_score.shape[1]
    for c in range(n_classes):
        prec, rec, _ = precision_recall_curve((y_true==c).astype(int), y_score[:,c])
        ap = average_precision_score((y_true==c).astype(int), y_score[:,c])
        plt.plot(rec, prec, label=f"cat {c} (AP={ap:.2f})")
    plt.title("Precision–Recall per class")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend()
    f = make_fig(out_dir,"pr_per_class"); save_and_close(f)
    return ("PR per class", f.name)

def plot_calibration(preds, out_dir, n_bins=10):
    if "proba_top1" not in preds or not _has_truth(preds):
        return None
    y_true = (preds["category"] == preds["pred_top1"]).astype(int)
    prob_true, prob_pred = calibration_curve(y_true, preds["proba_top1"], n_bins=n_bins)
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1],"k--")
    plt.title("Calibration curve")
    plt.xlabel("Predicted prob")
    plt.ylabel("True freq")
    f = make_fig(out_dir,"calibration"); save_and_close(f)
    return ("Calibration curve", f.name)


def plot_tag_cooccurrence(df, out_dir, top_k=25):
    if "tags" not in df.columns:
        return None
    wt = df.copy()
    wt["tags"] = wt["tags"].fillna("").astype(str)
    rows = []
    for t in wt["tags"]:
        toks = [x.strip() for x in t.split(",") if x.strip()]
        rows.extend(toks)
    if not rows:
        return None
    top = pd.Series(rows).value_counts().head(top_k).index.tolist()
    mat = pd.DataFrame(0, index=top, columns=top, dtype=int)
    for t in wt["tags"]:
        toks = [x.strip() for x in t.split(",") if x.strip()]
        toks = [x for x in toks if x in top]
        for i in range(len(toks)):
            for j in range(i+1, len(toks)):
                a, b = toks[i], toks[j]
                mat.loc[a, b] += 1
                mat.loc[b, a] += 1
    if mat.values.sum() == 0:
        return None
    sns.heatmap(mat, cmap=CMAP)
    plt.title("Tag co-occurrence (top tags)")
    plt.xlabel("Tag"); plt.ylabel("Tag")
    f = make_fig(out_dir, "16b_tag_cooccurrence"); save_and_close(f)
    return ("Tag co-occurrence", f.name)

# -------------------- HTML --------------------
HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Shadow-Insta Report</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color:#1b1b1b; }
  h1 { margin-bottom: 0.2rem; }
  .sub { color:#666; margin-bottom: 1.2rem; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(420px,1fr)); gap:22px; }
  figure { margin:0; background:#fff; border:1px solid #e8e8e8; border-radius:12px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,.04); }
  figure img { width:100%; display:block; }
  figcaption { padding:10px 14px; font-size:14px; color:#444; border-top:1px solid #eee; background:#fafafa; }
</style>
</head>
<body>
<h1>Shadow-Insta Report</h1>
<div class="sub">Auto-generated visual analysis • __NPLOTS__ figures</div>
<div class="grid">
__FIGS__
</div>
</body>
</html>
"""

def write_index(out_dir: Path, items: list[tuple[str, str]]):
    cards = []
    for title, fn in items:
        if not fn:
            continue
        cards.append(f'<figure><img src="figs/{fn}" alt="{title}"/><figcaption>{title}</figcaption></figure>')
    html = HTML_TEMPLATE.replace("__NPLOTS__", str(len(cards))).replace("__FIGS__", "\n".join(cards))
    (out_dir / "index.html").write_text(html, encoding="utf-8")

# -------------------- main --------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_date", default=None, help="Restrict evaluation to a single YYYY-MM-DD date (optional)")
    ap.add_argument("--inferred", required=True, help="inferred_categories.csv")
    ap.add_argument("--with_tags", default=None, help="inferred_with_tags.csv (optional)")
    ap.add_argument("--preds", default=None, help="model predictions CSV (optional)")
    ap.add_argument("--model_dir", default=None, help="folder with model.pkl/vectorizer.pkl (optional)")
    ap.add_argument("--markov", default=None, help="markov_transitions.csv (optional)")
    ap.add_argument("--out", required=True, help="output report folder")
    ap.add_argument("--per_cat", type=int, default=1, help="Include per-category expansions (1=yes,0=no)")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out))
    figs_dir = ensure_dir(out_dir / "figs")

    # --- Read once ---
    df, wt, preds = load_core(Path(args.inferred), args.with_tags, args.preds)

    # --- Harmonize / attach truth for preds if present ---
    if preds is not None and not preds.empty:
        preds = harmonize_pred_columns(preds)
        preds = attach_true_category(preds, df)
        preds = preds.sort_values(by=[c for c in ["timestamp"] if c in preds.columns])

    # --- Build figure list ---
    items: list[tuple[str, str]] = []

    # Core data
    items += [plot_category_hist(df, figs_dir)]
    items += [plot_rows_over_time(df, figs_dir)]
    items += [plot_category_share_by_month(df, figs_dir)]
    items += [plot_activity_heatmap(df, figs_dir)]
    items += [plot_top_accounts(df, figs_dir)]
    t = plot_top_hashtags(df, figs_dir)
    if t: items += [t]

    # Per-category expansions (adds many figures)
    if args.per_cat:
        items += plot_top_accounts_by_cat(df, figs_dir, n=10)
        items += plot_top_hashtags_by_cat(df, figs_dir, n=10)
        items += plot_hourly_activity_by_cat(df, figs_dir)

    # Markov (optional)
    t = plot_markov_heatmap(args.markov, figs_dir)
    if t: items += [t]
    t = plot_markov_ranklines(args.markov, figs_dir)
    if t: items += [t]

    # Tags (optional; only if 'tags' is present in df via with_tags merge)
    t = plot_tag_frequency(df, figs_dir)
    if t: items += [t]
    t = plot_tag_category_heatmap(df, figs_dir)
    if t: items += [t]
    t = plot_tag_share_over_time(df, figs_dir)
    if t: items += [t]
    t = plot_tag_cooccurrence(df, figs_dir)
    if t: items += [t]

    # Predictions (optional)
    if preds is not None and not preds.empty:
        t = plot_confusion(preds, figs_dir)
        if t: items += [t]
        t = plot_class_prf1(preds, figs_dir)
        if t: items += [t]
        t = plot_confidence_hist(preds, figs_dir)
        if t: items += [t]
        t = plot_rolling_accuracy(preds, figs_dir)
        if t: items += [t]
        t = plot_acc_by_month(preds, figs_dir)
        if t: items += [t]
        t = plot_errors_by_category(preds, figs_dir)
        if t: items += [t]
        t = plot_error_pairs_heatmap(preds, figs_dir)
        if t: items += [t]
        t = plot_error_hour_heatmap(preds, figs_dir)
        if t: items += [t]

        # --- Optional: filter to a single test date ---
        if args.test_date:
            dt = pd.to_datetime(args.test_date).date()
            preds = preds[_parse_dt_any(preds["timestamp"]).dt.date == dt]
            print(f"Filtered predictions to {len(preds)} rows on {dt}")


    # Feature importance per class (positive & negative)
    ft = plot_top_features(Path(args.model_dir) if args.model_dir else None, figs_dir)
    if ft:
        items += ft

    # Write index
    write_index(out_dir, [x for x in items if x])

    print(f"Report written to: {out_dir / 'index.html'}")
    print(f"Figures: {len(items)} saved under: {figs_dir}")

if __name__ == "__main__":
    main()
