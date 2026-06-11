<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDd6bGJoZHZ2ZDNrcWVneHZ6aGw3NzBheTQwdDFhaTl3MHNnNXV1eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPEqDGUULpEU0aQ/giphy.gif" width="100%" height="200" style="object-fit:cover;border-radius:12px" alt="data flow animation"/>

<br/><br/>

# 🎬 Scalable Movie Recommendation Engine
### A 10-Million-Row Matrix Factorization Case Study

<br/>

![Data Scale](https://img.shields.io/badge/Data_Scale-10M_Ratings-7F77DD?style=for-the-badge&logo=apache-spark&logoColor=white)
![Algorithm](https://img.shields.io/badge/Algorithm-SVD_%2B_ALS_Ensemble-534AB7?style=for-the-badge&logo=scikit-learn&logoColor=white)
![RMSE](https://img.shields.io/badge/Leaderboard_RMSE-0.82-1D9E75?style=for-the-badge&logo=target&logoColor=white)
![Inference](https://img.shields.io/badge/Inference-5M_rows_%2F_2_min-D85A30?style=for-the-badge&logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-0F6E56?style=for-the-badge)

<br/>

> 🔒 **Academic compliance note:** In strict accordance with the platform's honor code and competition terms, the source code is securely hosted in a private repository. This document serves as an architectural and engineering case study only.

</div>

---

## 📌 Executive Summary

How do you predict what movie someone wants to watch next — out of 48,000 options — when you only know what they've rated before? This project answers that question at scale.

Working with **10 million user ratings** from the MovieLens dataset, I engineered a recommendation pipeline that learns hidden "taste profiles" for 162,541 users and 48,213 movies, then predicts unseen ratings with a final leaderboard **RMSE of 0.82** — placing in the top positions of the ExploreAI Academy 2026 hackathon.

The real challenge wasn't the algorithm. It was the **99.87% matrix sparsity**, the **5,044 cold-start movies** with zero training history, and the need to run **5 million inferences** without grinding to a halt.

---

## 📊 Dataset at a Glance

| Metric | Value |
|--------|-------|
| 🎯 Training ratings | **10,000,038** |
| 👥 Unique users | **162,541** |
| 🎬 Unique movies (train) | **48,213** |
| 🧊 Matrix sparsity | **99.87%** |
| 📉 Avg ratings / user | **61.5** (max: 12,952) |
| ❄️ Cold-start movies | **5,044** (in test, unseen in train) |
| 📅 Date range | **1996 – 2019** |

---

## 🔬 Key Data Findings

### Rating Distribution
Users skew overwhelmingly positive. **4.0★** is the most common rating (26.5% of all ratings), and the distribution trails off sharply below 2.0★ — people tend to watch movies they expect to enjoy.

```
0.5★  ████░░░░░░░░░░░░░░░░  158K
1.0★  ████████░░░░░░░░░░░░  311K
2.0★  ████████████████░░░░  657K
3.0★  ████████████████████  1.96M  ← 2nd most common
4.0★  ████████████████████  2.65M  ← most common
5.0★  █████████████████░░░  1.45M
```

### Genre Breakdown (by movie count)
```
Drama       ████████████████████  25,606
Comedy      █████████████░░░░░░░  16,870
Thriller    ██████░░░░░░░░░░░░░░   8,654
Romance     █████░░░░░░░░░░░░░░░   7,719
Action      █████░░░░░░░░░░░░░░░   7,348
Horror      ████░░░░░░░░░░░░░░░░   5,989
```

### Most-Rated Films
| Rank | Title | Ratings | Avg ★ |
|------|-------|---------|-------|
| 1 | The Shawshank Redemption (1994) | 32,831 | 4.42 |
| 2 | Forrest Gump (1994) | 32,383 | 4.05 |
| 3 | Pulp Fiction (1994) | 31,697 | 4.20 |
| 4 | The Silence of the Lambs (1991) | 29,444 | 4.14 |
| 5 | The Matrix (1999) | 29,014 | 4.15 |
| 6 | Star Wars: Episode IV (1977) | 27,560 | 4.11 |
| 7 | Schindler's List (1993) | 24,004 | 4.25 |
| 8 | Fight Club (1999) | 23,536 | 4.23 |

> **Insight:** All top-rated films are from the 90s — this period dominates because early adopters of the platform were 90s film enthusiasts who rated prolifically.

---

## 🧠 The Math Behind the Magic

Every predicted rating is computed as:

```
r̂(u,i) = μ + b_u + b_i + qᵢᵀ pᵤ
```

| Symbol | Component | What it captures | Range |
|--------|-----------|-----------------|-------|
| **μ** | Global mean | Average rating across everyone | ≈ 3.53★ |
| **b_u** | User bias | Is this user a harsh critic or easy to please? | −0.8 to +0.8 |
| **b_i** | Movie bias | Is this film universally loved or disliked? | −1.2 to +0.9 |
| **qᵢᵀpᵤ** | Latent dot product | How perfectly does your taste match this film's profile? | 200 dimensions |

The SVD model learns 200-dimensional embeddings for every user and movie — these are the hidden "taste dimensions" (action vs drama, mainstream vs arthouse, etc.) that no human ever explicitly labelled.

---

## ⚡ Engineering Challenges

### ❄️ Cold-Start: 5,044 movies with zero training history

**The problem:** EDA revealed 5,044 movies in the evaluation set with *zero ratings* in training. Collaborative filtering has nothing to learn from — the matrix dot product returns noise for these items.

**The solution:** A three-tier routing system based on movie density:
- `count = 0` → Pure ALS bias fallback (global mean + refined user bias)
- `count < 10` → 10% SVD + 90% ALS bias
- `count ≥ 500` → 92% SVD + 8% ALS bias

### 🚀 Inference bottleneck: 5 million test rows

**The problem:** Row-by-row `.iterrows()` over 5M rows takes 4+ hours on standard hardware.

**The solution:** Refactored to Surprise's batch testset format — the entire evaluation block is passed as a structured array at once.

```
iterrows loop:  ████████████████████  ~4 hours
batch inference: ██░░░░░░░░░░░░░░░░░  ~2 minutes
```

---

## 🛠️ ML Pipeline

```
10M Train Dataset
       │
       ▼
EDA & Cold-Start Audit ──── flagged 5,044 unseen items
       │
       ▼
ALS Iterative Bias Model ── 10 rounds, reg_movie=10, reg_user=15
       │
       ▼
5-Fold Cross-Validation ─── on 20% sample for RMSE estimation
       │
       ▼
SVD Full Production Train ── 200 factors · 40 epochs · lr=0.005
       │
       ▼
Weighted Ensemble ────────── SVD weight scaled by movie density
       │
       ▼
Batch Inference + Clip ───── 5M predictions · per-user soft clipping
       │
       ▼
submission.csv
```

---

## 📈 Results

| Strategy | RMSE | Notes |
|----------|------|-------|
| Global baseline | ~1.05 | Predict global mean only |
| Vanilla SVD | 0.84 | 150 factors · 30 epochs · simple bias fallback |
| **SVD + ALS Ensemble** | **0.82** | 200 factors · weighted blend · per-user clipping |

The two-step improvement (0.84 → 0.82) came from replacing the simple mean fallback with an ALS iterative bias model, and routing predictions to the appropriate model tier based on how much data exists per movie.

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-surprise](https://img.shields.io/badge/scikit--surprise-SVD-FF6B6B?style=flat-square)
![pandas](https://img.shields.io/badge/pandas-2.0-150458?style=flat-square&logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-1.24-013243?style=flat-square&logo=numpy&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

---

## 💡 Key Takeaways

**Data-first mindset wins.** Solving the cold-start problem during EDA saved the submission from producing garbage predictions on 10% of test rows. Data auditing before modelling every time.

**ALS beats simple means.** Iterating between user and movie bias estimation — instead of computing them independently — produces significantly better predictions for sparse movies. The biases "correct each other" across 10 rounds until they stabilise.

**Routing > one-size-fits-all.** A movie with 2 ratings needs a completely different prediction strategy than a movie with 2,000. A weighted routing system that sends sparse items to the bias model and dense items to SVD consistently outperforms any single-model approach.

---

<div align="center">

*Built for ExploreAI Academy Hackathon 2026 · MovieLens dataset (CC BY-NC 4.0)*

</div>