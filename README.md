# Shadow-Insta 📊
End-to-end ML pipeline that converts raw Instagram reels/posts logs into **multi-label predictions** and **beautiful auto-generated reports**.

## Features
- Category inference & tagging (memes, movies, trolls, Tollywood…)
- Markov chain baseline + Logistic Regression classifier
- Multi-label tagger with JSON configs
- Auto-suggest for new tags
- Evaluation metrics: Accuracy, PRF1, Confusion Matrix, ROC, Calibration
- 20+ visualizations auto-exported into an HTML dashboard (`index.html`)
- Date-specific evaluation (e.g., `--test_date 2025-08-25`)

## Example Visuals
- Category distribution
- Markov transitions heatmap
- Top hashtags & accounts
- Tag × Category heatmap
- Accuracy by month
- Rolling accuracy curve
- Per-class PRF1 barplot

