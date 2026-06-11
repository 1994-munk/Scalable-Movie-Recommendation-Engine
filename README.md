# 🎬 Scalable Movie Recommendation Engine
### 🚀 A 10-Million-Row Matrix Factorization Case Study

<div align="center">
  <img src="https://img.shields.io/badge/Data_Scale-10M__Ratings-blueviolet?style=for-the-badge&logo=apache-spark&logoColor=white" alt="Data Scale">
  <img src="https://img.shields.io/badge/Algorithm-SVD__Collaborative__Filtering-FF6B6B?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Algorithm">
  <img src="https://img.shields.io/badge/Inference_Speed-5M__Rows_/_2_Mins-2ECC71?style=for-the-badge&logo=fastapi&logoColor=white" alt="Speed">
</div>

<br>

> 🔒 **Academic Compliance Note:** In strict accordance with the platform's academic honor code and competition terms, the underlying source code is confidential and securely hosted in a private repository. This documentation serves as an architectural and engineering case study.

---

## 📌 Executive Summary
How do you guess what movie someone wants to watch next out of thousands of choices? This project breaks down a massive dataset of **10 million user ratings** to build an optimized recommendation pipeline. By capturing hidden mathematical "taste profiles," the engine predicts user scores for unseen films, targeting an optimized **Root Mean Square Error (RMSE) between 0.85 and 0.90**.

---

## 🧠 System Architecture & Core Logic

### The Math Behind the Magic
Instead of looking at a movie's plot or actors, the engine uses **Singular Value Decomposition (SVD)** to map both users and movies into a shared **150-dimensional latent (hidden) factor space**. 

Click the dropdown below to see exactly how the AI calculates a rating:

<details>
<summary><b>🔍 INTERACTIVE: Click to expand the Prediction Formula</b></summary>
<br>

Every single predicted rating ($\hat{r}_{u,i}$) is calculated by blending four distinct layers:

$$\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$$

| Component | What it represents | Example |
| :--- | :--- | :--- |
| **$\mu$ (Global Mean)** | The baseline average rating across the entire app. | Everyone averages around `3.53` stars. |
| **$b_u$ (User Bias)** | Is the user a harsh critic or easy to please? | A critical user consistently rates `0.8` lower than average. |
| **$b_i$ (Movie Bias)** | Is the movie universally loved or universally hated? | A cinematic masterpiece naturally gets a `+0.9` boost. |
| **$q_i^T p_u$ (Latent Dot Product)** | How perfectly your personal taste matches the movie's vibe. | You love Sci-Fi + The movie is *Interstellar* = High Match! |

</details>

---

## 🛠️ Data Engineering Challenges (Click to Expand)

To simulate an interactive dashboard, click on the engineering hurdles below to see how they were systematically solved:

<details>
<summary><b>❄️ Challenge 1: The Cold-Start Dilemma (5,000+ Unseen Movies)</b></summary>

### The Problem
During Exploratory Data Analysis, I discovered **5,044 movies** in the evaluation dataset that had **zero history** in the training data. Because Collaborative Filtering relies purely on past behavior, the SVD model completely blanks out when it sees a brand-new item.

### The Engineering Solution
I engineered a robust **imputation safety net**. If the system catches a cold-start item ID, it bypasses the matrix dot-product and dynamically falls back to an aggregation of global-mean and independent item-mean scores. This keeps the prediction pipeline from crashing and guarantees clean data outputs.
</details>

<details>
<summary><b>⚡ Challenge 2: High-Throughput Bottlenecks (5 Million Test Rows)</b></summary>

### The Problem
Running a row-by-row iteration (like Python's `.iterrows()`) over 5,000,000 evaluation rows creates a massive computational bottleneck. On standard hardware, a basic loop would take several hours to execute.

### The Engineering Solution
I refactored the inference pipeline to utilize **batch vector predictions** via optimized array-based data structures in `scikit-surprise`. 

```text
Row-by-Row Loop:  |████████████████████████████████████████| 4+ Hours (Inefficient)
Batch Prediction: |████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| < 2 Minutes (Optimized)
By passing the entire block of data as a structural matrix at once, execution time plummeted to under two minutes.📊 Pipeline Blueprint & PerformanceExecution MilestonesThe project was executed across a tightly controlled machine learning lifecycle:Code snippetgraph TD
    A[10M Row Train Dataset] --> B[EDA & Cold-Start Auditing]
    B --> C[5-Fold Cross Validation Sample]
    C --> D[Hyperparameter Tuning n_factors=150]
    D --> E[Full 10M Production Train Fit]
    E --> F[Batch Inference Pipeline]
    F --> G[Submission Sanity Distribution Plot]
Expected Results Model ComparisonThrough rigorous local testing, the performance trade-offs between different modeling variations were mapped:Model StrategyProsConsTarget Leaderboard RMSEGlobal BaselineIncredibly fast, zero training required.Extremely inaccurate.~1.05+Vanilla SVD (Baseline)Great balance of memory usage and speed.Misses implicit behavior patterns.0.85 - 0.90 🎯Ensembled SVD + KNNBoosts accuracy by combining models.Heavy compute requirements.0.80 - 0.85📈 Key Takeaways & MethodologyFramework Dominance: Leveraged scikit-surprise for optimized matrix factoring utilities rather than writing standard regression models from scratch.Data-First Mindset: Solving the cold-start problem during EDA saved the submission from breaking, highlighting that data-cleaning always trumps pure modeling.Scale Handling: Proved that matrix math optimizations are vital when working with double-digit million row counts.