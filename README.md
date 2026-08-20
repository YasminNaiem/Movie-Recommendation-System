# 🎬 Hybrid Movie Recommendation System

A machine learning recommendation engine that combines **Content-Based Filtering** and **Collaborative Filtering (SVD)** into a hybrid model, trained and evaluated on the MovieLens dataset.

---

## 📌 Overview

Most recommendation systems rely on a single approach — but each has blind spots:
- **Content-based filtering** ignores what similar users actually enjoyed.
- **Collaborative filtering** struggles with new or sparsely-rated movies (the "cold start" problem).

This project builds both approaches independently, then blends them into a **hybrid engine** with a tunable weighting parameter (`alpha`), and rigorously benchmarks all three against each other using standard ranking and error metrics.

---

## 🗂️ Dataset

- **Source:** MovieLens dataset
- **Raw size:** 100,836 ratings · 9,742 movies
- **After cleaning & filtering** (users/movies with fewer than 5 ratings removed):
  - **610** users
  - **3,650** movies
  - **90,274** ratings
  - Rating scale: 0.5 – 5.0 (average: 3.54)

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Cleaned missing values, removed duplicates, normalized movie titles
- Parsed genre strings into vectorizable text
- Filtered users and movies below a minimum rating threshold
- Split data: 80% train / 20% test (stratified by user)

### 2. Content-Based Filtering
- Built a **TF-IDF matrix** (3,650 × 159) on movie genres
- Computed **cosine similarity** between all movie pairs
- Generates recommendations based on genre similarity to a given movie

### 3. Collaborative Filtering (SVD)
- Trained a **Singular Value Decomposition (SVD)** model using the `surprise` library
- Parameters: 100 latent factors, 30 epochs, learning rate 0.005
- Trained on a **97.4% sparse** user-item matrix

### 4. Hybrid Recommendation Engine
- Combined content-based and collaborative scores using a weighted formula:
  `hybrid_score = alpha * collaborative_score + (1 - alpha) * content_score`
- Ran a grid search across `alpha` values (0.0 → 1.0) to find the optimal blend

### 5. Evaluation
Benchmarked all models on a held-out test set using:
- **RMSE** / **MAE** (rating prediction accuracy)
- **Precision** / **Recall** / **F1-Score** (relevance at a 4-star threshold)

---

## 📊 Results

| Metric | Collaborative (SVD) | Hybrid (alpha=0.7) |
|---|---|---|
| RMSE | **0.868** | 1.044 |
| MAE | **0.665** | 0.874 |
| Precision | **0.808** | 0.956 |
| Recall | 0.367 | 0.007 |
| F1-Score | **0.505** | 0.014 |

**Key finding:** The standalone SVD collaborative filtering model outperformed the hybrid model on ranking accuracy and error metrics, while the hybrid model showed different trade-offs on precision. Grid search across `alpha` confirmed **alpha = 1.0** (pure collaborative filtering) minimized RMSE and maximized F1-Score for this dataset — a data-backed conclusion rather than an assumed "hybrid is always better" result.

---

## 🏗️ Project Structure

```
├── app/                  # Application layer
├── data/                 # Raw and processed datasets (ratings.csv, movies.csv)
├── models/               # Saved trained models (svd_model.pkl)
├── outputs/              # Exported CSV results (recommendations, evaluation metrics)
├── visualizations/       # EDA and model comparison plots
└── Movie_recommendation.ipynb   # Main analysis notebook
```

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy
- **scikit-learn** — TF-IDF, cosine similarity, train/test split, evaluation metrics
- **Surprise** — SVD collaborative filtering
- **Matplotlib / Seaborn** — visualizations
- **joblib** — model persistence

---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn joblib
   ```

3. Place `ratings.csv` and `movies.csv` inside the `data/` folder

4. Open and run the notebook
   ```bash
   jupyter notebook Movie_recommendation.ipynb
   ```

---

## 📈 Example Output

**Content-Based Recommendations (similar to "Toy Story"):**
- Antz
- Toy Story 2
- Adventures of Rocky and Bullwinkle
- Emperor's New Groove
- Monsters, Inc.

**Hybrid Recommendations (User 1 + "Toy Story", alpha=0.7):**
- Toy Story 2
- Monsters, Inc.
- Shrek
- Finding Nemo
- Ice Age

---

## 📄 License

This project is for educational purposes.
