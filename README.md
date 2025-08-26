# 📊 Shadow Insta Analysis

This project analyzes Instagram activity and builds an **end-to-end ML + visualization pipeline**.  
It clusters your Instagram interactions, predicts future “vibes,” and produces **rich visual reports**.

---

## 📂 Project Structure
- `auto_suggest_tags.py` → Suggests relevant hashtags  
- `build_reels_log.py` → Builds engagement log  
- `multi_label_tagger.py` → Multi-label classification for tags  
- `train_shadow_insta.py` → Model training & evaluation  
- `report_v2/figs/` → Graph outputs (36+ visualizations)  

---

## 📈 Analysis Graphs

### 🔹 Core Trends
| Category Histogram | Rows per Day | Category Share (Monthly) |
|--------------------|--------------|---------------------------|
| ![](report_v2/figs/01_category_hist.png) | ![](report_v2/figs/02_rows_per_day.png) | ![](report_v2/figs/03_category_share_month.png) |

| Activity Heatmap | Top Accounts | Top Hashtags |
|------------------|--------------|--------------|
| ![](report_v2/figs/04_activity_heatmap.png) | ![](report_v2/figs/05_top_accounts.png) | ![](report_v2/figs/06_top_hashtags.png) |

---

### 🔹 Sequence Dynamics
| Markov Heatmap | Markov Rank Lines |
|----------------|-------------------|
| ![](report_v2/figs/07_markov_heatmap.png) | ![](report_v2/figs/07b_markov_ranklines.png) |

---

### 🔹 Tags & Categories
| Tag Frequency | Tag–Category Heatmap | Tag Share Over Time |
|---------------|----------------------|----------------------|
| ![](report_v2/figs/14_tag_frequency.png) | ![](report_v2/figs/15_tag_category_heatmap.png) | ![](report_v2/figs/16_tag_share_time.png) |

| Tag Co-Occurrence |
|-------------------|
| ![](report_v2/figs/16b_tag_cooccurrence.png) |

---

### 🔹 Accounts by Cluster
| Cat 0 | Cat 1 | Cat 2 |
|-------|-------|-------|
| ![](report_v2/figs/acct_cat0.png) | ![](report_v2/figs/acct_cat1.png) | ![](report_v2/figs/acct_cat2.png) |

| Cat 3 | Cat 4 | Cat 5 |
|-------|-------|-------|
| ![](report_v2/figs/acct_cat3.png) | ![](report_v2/figs/acct_cat4.png) | ![](report_v2/figs/acct_cat5.png) |

---

### 🔹 Feature Importance (Negative Correlations)
| Cat 0 | Cat 1 | Cat 2 |
|-------|-------|-------|
| ![](report_v2/figs/feat_neg_cat0.png) | ![](report_v2/figs/feat_neg_cat1.png) | ![](report_v2/figs/feat_neg_cat2.png) |

| Cat 3 | Cat 4 | Cat 5 |
|-------|-------|-------|
| ![](report_v2/figs/feat_neg_cat3.png) | ![](report_v2/figs/feat_neg_cat4.png) | ![](report_v2/figs/feat_neg_cat5.png) |

---

### 🔹 Feature Importance (Positive Correlations)
| Cat 0 | Cat 1 | Cat 2 |
|-------|-------|-------|
| ![](report_v2/figs/feat_pos_cat0.png) | ![](report_v2/figs/feat_pos_cat1.png) | ![](report_v2/figs/feat_pos_cat2.png) |

| Cat 3 | Cat 4 | Cat 5 |
|-------|-------|-------|
| ![](report_v2/figs/feat_pos_cat3.png) | ![](report_v2/figs/feat_pos_cat4.png) | ![](report_v2/figs/feat_pos_cat5.png) |

---

### 🔹 Hourly Activity by Category
| Cat 0 | Cat 1 | Cat 2 |
|-------|-------|-------|
| ![](report_v2/figs/hour_cat0.png) | ![](report_v2/figs/hour_cat1.png) | ![](report_v2/figs/hour_cat2.png) |

| Cat 3 | Cat 4 | Cat 5 |
|-------|-------|-------|
| ![](report_v2/figs/hour_cat3.png) | ![](report_v2/figs/hour_cat4.png) | ![](report_v2/figs/hour_cat5.png) |

---

## 🚀 How to Run

```bash
# Clone this repo
git clone https://github.com/Kalyankaparaju/shadow-insta-analysis.git

# Navigate inside
cd shadow-insta-analysis

# Run training (example)
python train_shadow_insta.py
