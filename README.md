# Music Recommendation System (Top-3) with Implicit Feedback

## Overview
This project builds an end-to-end **music recommender system** that generates **top-3 song recommendations per user** using **implicit feedback** (listens, likes, skips, playlist additions).

The project intentionally starts from **messy, realistic interaction logs** to demonstrate the full data science workflow:
data auditing → cleaning → feature construction → modeling → evaluation.

---

## Objective
Given a user’s historical interactions with songs, the system ranks unseen songs and returns the **top 3 recommendations** most likely to result in positive engagement.

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
├── music_interactions_messy.csv
├── music_recommender.ipynb
└── README.md


---

## Current Progress
✔ Environment setup and reproducibility configuration  
✔ Dataset loading and inspection  
✔ Systematic data audit and quality assessment  
✔ Documentation of data issues and modeling implications  

The notebook currently focuses on **understanding and validating the data** before any cleaning or modeling is performed.

---

## Modeling Plan
- **Feedback type:** Implicit
- **Primary model:** Matrix Factorization (ALS)
- **Library:** `implicit`
- **Recommendation task:** Top-N ranking (N = 3)

A popularity-based baseline will be used for comparison.

---

## Evaluation
Offline evaluation will use ranking-based metrics:
- Recall@3
- Precision@3
- NDCG@3

A time-based train/test split will be used to simulate real-world recommendation scenarios.

---

## Tools & Libraries
- Python
- Pandas, NumPy
- SciPy (sparse matrices)
- implicit
- Matplotlib / Seaborn
- Jupyter Notebook

---

## Next Steps
- Clean and standardize interaction data
- Construct implicit interaction strength
- Build user–item sparse matrix
- Train ALS recommender
- Evaluate Top-3 recommendations
- Export final recommendations

---

## Notes
This project is designed as a **portfolio-quality recommender system** demonstrating both data engineering and modeling considerations commonly encountered in production systems.
