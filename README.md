# Music Recommendation System (Top-10) with Implicit Feedback

## Overview
This project builds an end-to-end **music recommender system** designed to generate **top-10 personalized song recommendations**. Unlike traditional systems that rely on explicit ratings (e.g., 1-5 stars), this project leverages **implicit feedback**,signals like plays, likes, and skips, to infer user preferences.

A core focus of this project is the **full data science lifecycle**, starting from an intentionally messy dataset to demonstrate robust data engineering and auditing before modeling.

---

## The "Messy" Dataset
**File:** `music_interactions_messy.csv`

The project begins with a **Data Quality Audit**, which identifies and fixes real-world logging issues:
- **Inconsistent Categorical Labels:** Normalizing event types (e.g., 'PLAY' vs 'play').
- **Data Type Issues:** Handling non-numeric strings in `play_count` and `listen_seconds`.
- **Invalid Timestamps:** Cleaning unparseable date formats.
- **ID Corruption:** Removing "guest" or null `user_id` and `song_id` entries.

---

## Technical Pipeline

### 1. Data Auditing & Cleaning
Standardized the dataset by normalizing identifiers, mapping inconsistent labels, and handling outliers in listening duration and play counts.

### 2. Interaction Strength Construction
Converted raw events into a numeric **interaction strength** score:
- **Positive Weight:** High weight for `Like` and `Add to Playlist`.
- **Base Weight:** Derived from `play_count`.
- **Negative Feedback:** `Skip` events result in a zeroed-out strength to ensure the model avoids those items.

### 3. Sparse Matrix & Filtering
To handle high sparsity:
- **Filtering:** Retained only users and songs with at least **5 interactions**.
- **Matrix Representation:** Data is converted into a **Compressed Sparse Row (CSR) matrix** to optimize memory usage for the recommendation engine.

### 4. Time-Based Evaluation
Used a **Leave-Last-1** split strategy. For every user, the most recent interaction is held out for testing, while all prior history is used for training. This accurately simulates a real-world scenario of predicting the next song a user will enjoy.

---

## Modeling: Matrix Factorization (ALS)
The system uses **Alternating Least Squares (ALS)** for collaborative filtering via the `implicit` library. 
- **Personalization:** Learns latent factors for users and songs to find patterns beyond simple popularity.
- **Confidence Scaling:** Uses `alpha` scaling to treat the interaction strengths as confidence levels for user preferences.

---

## Evaluation Results
The system was evaluated using ranking-based metrics at $K=10$, comparing the personalized ALS model against a global popularity baseline.

| Metric | Popularity Baseline | **Personalized ALS** |
| :--- | :--- | :--- |
| **Recall@10** | ~0.177 | **~0.206** |
| **Precision@10** | ~0.017 | **~0.020** |
| **NDCG@10** | N/A | **~0.128** |

*The improvement over the baseline confirms that the model successfully learned individualized user tastes.*

---

## Project Structure
```text
├── music_interactions_messy.csv   # Raw, messy input data
├── music_recommender.ipynb        # Main analysis and modeling notebook
├── final_recommendations.csv      # Exported Top-10 list for all users
└── README.md                      # Project documentation
```

---

Tools & Requirements
- Core: Python, Pandas, NumPy
- ML/Math: SciPy (Sparse Matrices), implicit (ALS)
- Viz: Matplotlib, Seaborn

To run the project, ensure you have the implicit library installed:

pip install implicit

Conclusion
This project demonstrates a complete recommendation pipeline, from raw, noisy data to a deployable recommendation list, highlighting the importance of data quality and the effectiveness of matrix factorization for implicit feedback.
