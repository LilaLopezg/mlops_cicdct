#!/usr/bin/env python3
"""
Script de entrenamiento optimizado:
 - SMOTEENN
 - GridSearch
 - RandomForest & XGBoost
 - Evaluación con THRESHOLD (0.50 y 0.70)
 - Selección del mejor modelo
 - Guardado con metadatos
"""

import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.combine import SMOTEENN


# --------------------------------------------------------------------
# 1. Cargar datos
# --------------------------------------------------------------------
def load_data(filepath="/home/ubuntu/mlops_cicdct/data/train.csv"):
    print("Cargando dataset...")

    df = pd.read_csv(filepath, sep=";")

    X = df.drop("Consumo", axis=1).select_dtypes(include=["number"])
    y = df["Consumo"]

    feature_names = list(X.columns)
    target_names = sorted(y.unique().astype(str))

    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases: {target_names}")

    return X, y, feature_names, target_names


# --------------------------------------------------------------------
# 2. Preprocesamiento
# --------------------------------------------------------------------
def preprocess_data(X, y, random_state=42):
    print("Preprocesando datos...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Aplicando SMOTEENN...")
    smote_enn = SMOTEENN(random_state=random_state)
    X_train_bal, y_train_bal = smote_enn.fit_resample(X_train_s, y_train)

    print(f"Train original: {len(X_train)}, Balanceado: {len(X_train_bal)}")

    return X_train_bal, X_test_s, y_train_bal, y_test, scaler


# --------------------------------------------------------------------
# 3. Modelos + GridSearch
# --------------------------------------------------------------------
def train_models(X_train, y_train, random_state=42):

    results = {}

    # ----------------------
    # RANDOM FOREST
    # ----------------------
    print("\n→ Optimizando RandomForest...")

    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    rf_params = {
        "n_estimators": [200, 300],
        "max_depth": [5, 7],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf_grid = GridSearchCV(
        rf, rf_params, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train, y_train)
    results["RF"] = rf_grid

    # ----------------------
    # XGBOOST
    # ----------------------
    print("\n→ Optimizando XGBoost...")

    class_ratio = y_train.value_counts()
    scale_pos = class_ratio.min() / class_ratio.max()

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        random_state=random_state
    )

    xgb_params = {
        "n_estimators": [200, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    }

    xgb_grid = GridSearchCV(
        xgb, xgb_params, cv=3, scoring="f1_macro",
        n_jobs=-1, verbose=1
    )
    xgb_grid.fit(X_train, y_train)
    results["XGB"] = xgb_grid

    return results


# --------------------------------------------------------------------
# THRESHOLD customizado
# --------------------------------------------------------------------
def predict_with_threshold(model, X, thr):
    """Devuelve etiquetas aplicando threshold."""
    proba = model.predict_proba(X)[:, 1]
    return (proba >= thr).astype(int)


# --------------------------------------------------------------------
# 4. Evaluación de modelos + threshold
# --------------------------------------------------------------------
def evaluate_models(model_dict, X_test, y_test, target_names):

    print("\n=== EVALUACIÓN FINAL CON THRESHOLD ===")

    thresholds = [0.50, 0.70]
    best_model = None
    best_acc = 0
    best_thr = 0
    best_name = ""

    for name, grid in model_dict.items():

        model = grid.best_estimator_

        for thr in thresholds:

            print(f"\n>>> Modelo: {name} con threshold={thr}")

            y_pred = predict_with_threshold(model, X_test, thr)
            acc = accuracy_score(y_test, y_pred)

            print(f"Accuracy = {acc:.4f}")
            print(classification_report(y_test, y_pred, target_names=target_names))
            print("Matriz de confusión:")
            print(confusion_matrix(y_test, y_pred))

            # Seleccionar el mejor modelo + threshold
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_thr = thr
                best_name = name

    print(f"\n🏆 Mejor modelo: {best_name} con threshold={best_thr} accuracy={best_acc:.4f}")

    return best_model, best_acc, best_thr


# --------------------------------------------------------------------
# 5. Guardar modelo final
# --------------------------------------------------------------------
def save_model(model, scaler, feature_names, target_names, accuracy, threshold):

    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "target_names": target_names,
        "accuracy": accuracy,
        "threshold": threshold,
        "timestamp": timestamp
    }

    path = f"models/best_consumo_model_{timestamp}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"✓ Modelo guardado en: {path}")
    return path


# --------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------
def main():

    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    grids = train_models(X_train, y_train)

    best_model, best_acc, best_thr = evaluate_models(
        grids, X_test, y_test, target_names
    )

    save_model(best_model, scaler, feature_names, target_names, best_acc, best_thr)


if __name__ == "__main__":
    main()
