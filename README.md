# Music Recommendation System (Top-10) with Implicit Feedback

## Overview
This project builds an end-to-end **music recommender system** that generates **top-10 song recommendations per user** using **implicit feedback** (listens, likes, skips, playlist additions).

The project intentionally starts from **messy, realistic interaction logs** to demonstrate the full data science workflow:
data auditing -> cleaning -> feature construction -> modeling -> evaluation.

---

## Objective
Given a user’s historical interactions with songs, the system ranks unseen songs and returns the **top 10 recommendations** most likely to result in positive engagement.

---

## Dataset
**File:** `music_interactions_messy.csv`

The dataset represents user–song interaction logs and is intentionally noisy.
Each row corresponds to a single interaction event.

### Key characteristics
- Duplicate rows
- Missing values
- Inconsistent categorical labels
- Invalid timestamps
- Mixed data types
- Outliers in numeric fields

### Core columns
- `user_id`, `song_id`, `artist_id`
- `event_type` (play, like, skip, add_to_playlist, etc.)
- `play_count`, `listen_seconds`
- `liked`, `added_to_playlist`, `skipped`
- `timestamp`
- `device`, `country`, `genre`

This structure reflects real-world logging data used in recommender systems.

---

## Project Structure
```
├── music_interactions_messy.csv
├── final_recommendations.csv
├── music_recommender.ipynb
└── README.md
```

---

## Pipeline Summary
- Environment setup and reproducibility configuration
- Dataset loading, inspection, and data quality audit
- Cleaning and standardization of interaction logs
- Construction of implicit interaction strength
- Filtering sparse users and songs
- User/song ID encoding and sparse user-to-item matrix build
- Time-based train/test split (leave-last-1 per user)
- Popularity baseline evaluation
- ALS model training with confidence scaling
- Offline evaluation at top-10

---

## Modeling
- **Feedback type:** Implicit
- **Primary model:** Matrix Factorization (ALS)
- **Library:** `implicit`
- **Recommendation task:** Top-N ranking (N = 10)

A popularity-based baseline is used for comparison.

---

## Evaluation
Offline evaluation uses ranking-based metrics:
- Recall@10
- Precision@10
- NDCG@10

A time-based train/test split simulates real-world recommendation scenarios.

---

## Tools & Libraries
- Python
- Pandas, NumPy
- SciPy (sparse matrices)
- implicit
- Matplotlib / Seaborn
- Jupyter Notebook

---

## Notes
This project is designed as a **portfolio-quality recommender system** demonstrating both data engineering and modeling considerations commonly encountered in production systems.
