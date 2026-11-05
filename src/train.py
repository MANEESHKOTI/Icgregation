"""
Model Training, Benchmarking, and Serialization Module.

This script:
1. Obtains preprocessed GTZAN training and test sets from src.preprocess.
2. Trains three scikit-learn classification models:
   - Logistic Regression
   - Support Vector Classifier (SVM)
   - Random Forest Classifier
3. Evaluates all models on test set using:
   - Accuracy
   - Macro-averaged Precision
   - Macro-averaged Recall
   - Macro-averaged F1-Score
4. Saves benchmark results to 'output/benchmark_results.csv'.
5. Generates comparison bar plot to 'output/benchmark_comparison.png'.
6. Selects best model based on macro-averaged F1-Score.
7. Saves best model metrics to 'output/best_model_metrics.json'.
8. Serializes best model using joblib to 'output/best_model.joblib'.
"""

import json
import os
import sys
from typing import Dict, List, Any, Tuple

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

try:
    from src.preprocess import preprocess_pipeline
except ImportError:
    from preprocess import preprocess_pipeline



def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Define classifiers to benchmark.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "SVM": SVC(
            kernel="rbf",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=random_state
        )
    }


def train_and_evaluate_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Train each model, compute evaluation metrics, and collect results.
    """
    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        results.append({
            "model_name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

    return results, trained_models


def save_benchmark_csv(
    results: List[Dict[str, Any]],
    output_path: str = "output/benchmark_results.csv"
) -> pd.DataFrame:
    """
    Save benchmark metrics into a CSV file with required columns:
    model_name, accuracy, precision, recall, f1_score.
    """
    df = pd.DataFrame(results)
    # Ensure exact column ordering
    columns = ["model_name", "accuracy", "precision", "recall", "f1_score"]
    df = df[columns]
    df.to_csv(output_path, index=False)
    print(f"Saved benchmark results to {output_path}")
    return df


def generate_benchmark_plot(
    results_df: pd.DataFrame,
    output_path: str = "output/benchmark_comparison.png"
) -> None:
    """
    Generate and save a bar plot comparing models by key metrics (F1-Score focus).
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create bar plot for F1-score
    ax = sns.barplot(
        data=results_df,
        x="model_name",
        y="f1_score",
        palette="viridis",
        hue="model_name",
        legend=False
    )
    
    plt.title("Model Benchmarking Comparison (Macro F1-Score)", fontsize=16, weight="bold", pad=15)
    plt.xlabel("Model", fontsize=12, labelpad=10)
    plt.ylabel("Macro F1-Score", fontsize=12, labelpad=10)
    plt.ylim(0, 1.05)
    
    # Annotate bar values
    for p in ax.patches:
        val = p.get_height()
        ax.annotate(
            f"{val:.4f}",
            (p.get_x() + p.get_width() / 2., val),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontweight="bold",
            fontsize=11
        )
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved benchmark comparison plot to {output_path}")


def save_best_model_artifacts(
    results: List[Dict[str, Any]],
    trained_models: Dict[str, Any],
    json_path: str = "output/best_model_metrics.json",
    joblib_path: str = "output/best_model.joblib"
) -> Dict[str, Any]:
    """
    Identify best model based on macro F1-score, serialize model and save its metrics.
    """
    # Select best model based on macro F1-score
    best_result = max(results, key=lambda x: x["f1_score"])
    best_model_name = best_result["model_name"]
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with F1-Score: {best_result['f1_score']:.4f}")
    
    # Save best model metrics JSON with required exact keys:
    # model_name, accuracy, precision, recall, f1_score
    metrics_to_save = {
        "model_name": str(best_result["model_name"]),
        "accuracy": float(best_result["accuracy"]),
        "precision": float(best_result["precision"]),
        "recall": float(best_result["recall"]),
        "f1_score": float(best_result["f1_score"])
    }
    
    with open(json_path, "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Saved best model metrics to {json_path}")
    
    # Serialize best model
    joblib.dump(best_model, joblib_path)
    print(f"Serialized best model to {joblib_path}")
    
    return metrics_to_save


def main():
    os.makedirs("output", exist_ok=True)
    
    # Step 1: Preprocess data
    print("Preprocessing audio feature data...")
    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = preprocess_pipeline()
    
    # Step 2: Initialize models
    models = get_models(random_state=42)
    
    # Step 3: Train & evaluate
    results, trained_models = train_and_evaluate_models(
        models, X_train, X_test, y_train, y_test
    )
    
    # Step 4: Save benchmark CSV
    results_df = save_benchmark_csv(results, "output/benchmark_results.csv")
    
    # Step 5: Generate plot
    generate_benchmark_plot(results_df, "output/benchmark_comparison.png")
    
    # Step 6: Serialize best model & metrics JSON
    save_best_model_artifacts(
        results,
        trained_models,
        json_path="output/best_model_metrics.json",
        joblib_path="output/best_model.joblib"
    )
    print("\nBenchmarking and serialization workflow complete!")


if __name__ == "__main__":
    main()
