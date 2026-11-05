# Benchmark Classification Models for Music Genre Prediction

## Project Overview

Welcome to the **Music Genre Classification & Model Benchmarking** project. Automated music genre classification is a core capability powering modern music streaming and discovery platforms like Spotify and Apple Music. By analyzing audio characteristics such as spectral features, rhythm, and Mel-Frequency Cepstral Coefficients (MFCCs), machine learning algorithms can categorize songs into genres without human intervention.

This project delivers an end-to-end, production-grade machine learning pipeline that:
- Ingests and inspects the 30-second audio feature extractions from the famous **GTZAN Genre Collection dataset**.
- Implements exploratory data analysis (EDA) in an interactive Jupyter notebook.
- Formulates a clean data preprocessing workflow adhering to best practices (handling missing values, encoding categorical labels with `LabelEncoder`, splitting data into 80/20 train/test partitions, and scaling features with `StandardScaler` fitted strictly on training data to prevent data leakage).
- Benchmarks multiple classification algorithms:
  1. **Logistic Regression** (Regularized linear model)
  2. **Support Vector Machine (SVM)** with Radial Basis Function (RBF) kernel
  3. **Random Forest Classifier** (Ensemble tree-based model)
- Evaluates models using comprehensive multiclass classification metrics (Accuracy, Macro-averaged Precision, Macro-averaged Recall, and Macro-averaged F1-Score).
- Serializes the champion model (`output/best_model.joblib`), produces comparison visualization (`output/benchmark_comparison.png`), and exports structured benchmark reports (`output/benchmark_results.csv`, `output/best_model_metrics.json`).

---

## Project Structure

The project follows a standard, reproducible directory hierarchy:

```text
/
├── data/                      # Raw and processed datasets
│   └── features_30_sec.csv    # GTZAN 30-second audio features (1,000 tracks, 10 genres)
├── notebooks/                 # Jupyter Notebooks for exploration
│   └── eda.ipynb              # Exploratory Data Analysis & visual inspections
├── output/                    # Generated benchmark artifacts
│   ├── benchmark_results.csv  # Performance metrics across all benchmarked models
│   ├── benchmark_comparison.png # Comparison bar chart (Macro F1-Score)
│   ├── best_model_metrics.json # Performance metrics of the champion model
│   └── best_model.joblib      # Serialized scikit-learn best model object
├── src/                       # Production Python source code
│   ├── __init__.py            # Package initialization
│   ├── preprocess.py          # Data ingestion, label encoding, scaling, and splitting
│   └── train.py               # Model training, evaluation, comparison, and serialization
├── Dockerfile                 # Fast, lightweight containerization definition
├── docker-compose.yml         # Container orchestration configuration
├── .dockerignore              # Excludes local artifacts from container context
├── .gitignore                 # Excludes environments, caches, and large CSV files
├── requirements.txt           # Python dependency specifications
└── README.md                  # Comprehensive project documentation
```

---

## Setup and Installation

Follow the instructions below to configure the environment and run the pipeline locally or using Docker.

### Local Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Pleasoutory
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Using Python 3.10, 3.11, or 3.12
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Docker Setup (Fast Compose)

The project includes an optimized Docker setup based on `python:3.12-slim` designed for minimal build and compose times:

```bash
# Build and run the pipeline using Docker Compose
docker-compose up --build
```

Or using standalone Docker:
```bash
docker build -t music-genre-benchmark .
docker run --rm -v $(pwd)/output:/app/output music-genre-benchmark
```

---

## How to Run

### 1. Exploratory Data Analysis (EDA)
To interactively review the exploratory analysis and feature visualizations:
```bash
jupyter notebook notebooks/eda.ipynb
```
The notebook contains:
- Dataset shape, schema, and column summary.
- Class distribution verification (100 samples per genre).
- Missing values verification (0 missing values).
- Visualizations including histograms, KDE plots, genre boxplots, and correlation heatmaps.

### 2. Data Preprocessing
To execute the data preprocessing pipeline independently:
```bash
python src/preprocess.py
```

### 3. Model Training & Benchmarking
To execute model training, evaluation, plot generation, and serialization:
```bash
python src/train.py
```

---

## Exploratory Data Analysis Summary

Key takeaways from the exploratory data analysis on the GTZAN dataset:
- **Class Balance**: The dataset contains 1,000 audio tracks evenly split across 10 genres (100 tracks each): `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, and `rock`. Because classes are balanced, both accuracy and macro-averaged metrics offer reliable performance indications.
- **Completeness**: No missing values were detected in any feature column.
- **Scale Variance**: Audio features span distinct orders of magnitude (e.g., `spectral_centroid_mean` in thousands of Hz vs. `zero_crossing_rate_mean` between $0.01$ and $0.15$). Standardization with `StandardScaler` ensures features with wider ranges do not distort distance-based and regularized models.
- **Data Leakage Safeguards**: Preprocessing strictly fits `StandardScaler` only on `X_train` and transforms both `X_train` and `X_test`.

---

## Results and Findings

### Benchmark Results

All models were evaluated on the held-out test split (200 tracks, $20\%$ of dataset, stratified across all 10 genres). Evaluation metrics were computed using macro-averaging to weigh all genre classes equally:

| Model Name | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **0.7350** | **0.7378** | **0.7350** | **0.7340** |
| **Logistic Regression** | 0.7100 | 0.7229 | 0.7100 | 0.7107 |
| **Random Forest Classifier** | 0.6950 | 0.6874 | 0.6950 | 0.6878 |

*Detailed benchmark metrics are saved to [output/benchmark_results.csv](output/benchmark_results.csv).*

### Key Findings

1. **Top Performer**: The **Support Vector Machine (SVM)** with an RBF kernel achieved the highest performance across all evaluation metrics, reaching a **Macro F1-Score of 0.7340** and an **Accuracy of 73.50%**.
2. **Feature Linearity & Boundaries**: Music genre feature spaces exhibit non-linear decision boundaries where kernel-based mapping (SVM RBF) successfully separates acoustic features across genres like classical, metal, and jazz.
3. **Linear Baseline**: Logistic Regression performed commendably (Macro F1: $0.7107$), demonstrating that standardized spectral features have strong predictive signal even under linear decision boundaries.
4. **Ensemble Trees**: Random Forest delivered solid performance ($0.6878$ F1-Score) without requiring extensive hyperparameter tuning, although SVM remained superior on this standardized feature set.

### Model Serialization and Output Verification

The champion model (`SVM`) is persisted in the repository:
- **Trained Model Object**: [`output/best_model.joblib`](output/best_model.joblib)
- **Champion Metrics JSON**: [`output/best_model_metrics.json`](output/best_model_metrics.json)
  ```json
  {
      "model_name": "SVM",
      "accuracy": 0.735,
      "precision": 0.7378000040595949,
      "recall": 0.7350000000000001,
      "f1_score": 0.7340186077025197
  }
  ```
- **Comparison Visualization**: [`output/benchmark_comparison.png`](output/benchmark_comparison.png)
