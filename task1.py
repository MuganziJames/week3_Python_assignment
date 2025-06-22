# iris_decision_tree.py
"""
Train a decision-tree classifier on the classic Iris dataset
and evaluate it with accuracy, precision, and recall.

Usage:
    python iris_decision_tree.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading Iris dataset...")
    df = pd.read_csv("Iris.csv")       # Assumes Iris.csv is in the same folder
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Species distribution:\n{df['Species'].value_counts()}")
    
    df = df.drop(columns=["Id"])       # Row ID isn't a signal feature
    print(f"Dataset shape after dropping Id column: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Handle missing values (if any) – numeric mean imputation
    # ------------------------------------------------------------------
    print("\nChecking for missing values...")
    X = df.drop(columns=["Species"])
    y = df["Species"]
    
    print(f"Missing values in features:\n{X.isnull().sum()}")
    print(f"Missing values in target: {y.isnull().sum()}")

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    print("Applied mean imputation to handle any missing values (future-proofing)")

    # ------------------------------------------------------------------
    # 3. Encode labels
    # ------------------------------------------------------------------
    print("\nEncoding target labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Original classes: {label_encoder.classes_}")
    print(f"Encoded classes: {sorted(set(y_encoded))}")

    # ------------------------------------------------------------------
    # 4. Train / test split
    # ------------------------------------------------------------------
    print("\nSplitting data into train/test sets (80%/20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,  # keep class balance
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # ------------------------------------------------------------------
    # 5. Train decision tree
    # ------------------------------------------------------------------
    print("\nTraining Decision Tree Classifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print("Decision tree training completed")

    # ------------------------------------------------------------------
    # 6. Evaluate on the hold-out set
    # ------------------------------------------------------------------
    print("\nEvaluating model performance...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # ------------------------------------------------------------------
    # 7. Report metrics
    # ------------------------------------------------------------------
    print("\n" + "="*50)
    print("Decision Tree performance on Iris test set")
    print("="*50)
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print("="*50)
    
    # Additional information about the decision tree
    print(f"\nDecision Tree Details:")
    print(f"Tree depth: {clf.tree_.max_depth}")
    print(f"Number of leaves: {clf.tree_.n_leaves}")
    print(f"Feature importances:")
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    for i, importance in enumerate(clf.feature_importances_):
        print(f"  {feature_names[i]}: {importance:.3f}")


if __name__ == "__main__":
    main() 