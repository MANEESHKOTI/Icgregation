"""
Data Preprocessing Module for Music Genre Classification.

This module handles:
1. Loading the GTZAN audio features dataset.
2. Splitting features and target label.
3. Encoding categorical labels using LabelEncoder.
4. Performing train-test split (80/20) with a fixed random_state.
5. Scaling features using StandardScaler (fit ONLY on training data to prevent data leakage).
"""

import os
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str = "data/features_30_sec.csv") -> pd.DataFrame:
    """
    Load the GTZAN features dataset from CSV file.
    
    Args:
        filepath: Path to the dataset CSV file.
        
    Returns:
        pd.DataFrame containing the raw dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def prepare_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Separate technical audio features from metadata and target label.
    
    Drops non-feature metadata columns ('filename', 'length') if present,
    and isolates the 'label' target column.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple of (feature DataFrame X, target Series y, list of feature column names).
    """
    if "label" not in df.columns:
        raise KeyError("Expected 'label' column in dataframe")
    
    # Exclude non-feature columns
    cols_to_drop = [col for col in ["filename", "length", "label"] if col in df.columns]
    feature_cols = [col for col in df.columns if col not in cols_to_drop]
    
    X = df[feature_cols].copy()
    y = df["label"].copy()
    
    return X, y, feature_cols


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string target labels into integers using LabelEncoder.
    
    Args:
        y: Series of text labels.
        
    Returns:
        Tuple of (encoded labels array, fitted LabelEncoder instance).
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing partitions using a fixed random state.
    
    Args:
        X: Feature DataFrame.
        y: Target label array.
        test_size: Proportion of dataset for test set (default 0.2 = 20%).
        random_state: Fixed seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale numerical features to zero mean and unit variance.
    
    IMPORTANT: Fits ONLY on training features and transforms both
    training and testing data to strictly prevent data leakage.
    
    Args:
        X_train: Training feature DataFrame.
        X_test: Testing feature DataFrame.
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted StandardScaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    filepath: str = "data/features_30_sec.csv",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder, List[str]]:
    """
    Run full preprocessing pipeline.
    
    Args:
        filepath: Path to dataset CSV.
        test_size: Test partition fraction (default: 0.2).
        random_state: Random state seed (default: 42).
        
    Returns:
        Tuple of:
            - X_train_scaled (np.ndarray)
            - X_test_scaled (np.ndarray)
            - y_train (np.ndarray)
            - y_test (np.ndarray)
            - scaler (StandardScaler)
            - label_encoder (LabelEncoder)
            - feature_names (List[str])
    """
    df = load_data(filepath)
    X, y, feature_names = prepare_features_and_target(df)
    y_encoded, label_encoder = encode_labels(y)
    
    X_train, X_test, y_train, y_test = split_data(
        X, y_encoded, test_size=test_size, random_state=random_state
    )
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_names


if __name__ == "__main__":
    print("Running data preprocessing pipeline...")
    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = preprocess_pipeline()
    print(f"Features: {len(feature_names)} features extracted")
    print(f"Classes: {list(encoder.classes_)}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print("Preprocessing completed successfully.")

