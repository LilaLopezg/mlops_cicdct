#!/usr/bin/env python3
"""
Script de entrenamiento para clasificación del consumo de alcohol
Genera un modelo entrenado y lo guarda en formato pickle
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
import argparse



# ============================================================
# 1. CARGA DE DATOS
# ============================================================
def load_data(filepath=None):
    """Carga el dataset de consumo de alcohol"""

    if filepath is None:
        filepath = "/app/data/train.csv"

    print(f"Cargando dataset desde: {filepath}")

    df = pd.read_csv(filepath, sep=";")


    X = df.drop("Consumo", axis=1).select_dtypes(include=["number"])
    y = df["Consumo"]

    feature_names = list(X.columns)
    target_names = sorted(y.unique().astype(str))

    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases: {target_names}")

    return X, y, feature_names, target_names


# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================
def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Preprocesa los datos"""

    print("Preprocesando datos...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de prueba: {X_test.shape[0]} muestras")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================
# 3. ENTRENAMIENTO
# ============================================================
def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entrena el modelo Random Forest"""

    print("Entrenando modelo Random Forest...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("✓ Modelo entrenado exitosamente")
    return model


# ============================================================
# 4. MÉTRICAS NO PARAMÉTRICAS
# ============================================================
def bootstrap_ci_accuracy(model, X_test, y_test, n_bootstrap=1000, alpha=0.05):

    accuracies = []

    for _ in range(n_bootstrap):
        X_sample, y_sample = resample(X_test, y_test)
        y_pred_sample = model.predict(X_sample)
        accuracies.append(accuracy_score(y_sample, y_pred_sample))

    lower = np.percentile(accuracies, 100 * alpha / 2)
    upper = np.percentile(accuracies, 100 * (1 - alpha / 2))

    return lower, upper, np.mean(accuracies)


def mcnemar_test(y_true, pred_A, pred_B):

    b = np.sum((pred_A == y_true) & (pred_B != y_true))
    c = np.sum((pred_A != y_true) & (pred_B == y_true))

    table = [[0, b],
             [c, 0]]

    result = mcnemar(table, exact=True if (b + c) < 25 else False)
    return result.statistic, result.pvalue


def wilcoxon_test(scores_A, scores_B):
    stat, p = wilcoxon(scores_A, scores_B)
    return stat, p


# ============================================================
# 5. EVALUACIÓN
# ============================================================
def evaluate_model(model, X_test, y_test, target_names, model_B=None, threshold=0.5):
    """Evalúa el modelo con métricas clásicas + no paramétricas."""

    print("Evaluando modelo...")

   # --- Aplicar threshold ---
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        print(f"\nUsando threshold = {threshold}")
    else:
        # Modelos que no entregan probabilidades
        print("\nEl modelo no soporta predict_proba(). Usando predicciones directas.")

    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Reporte clásico
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Bootstrap CI
    ci_low, ci_high, acc_mean = bootstrap_ci_accuracy(model, X_test, y_test)
    print(f"\nBootstrap CI 95%: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Bootstrap Mean: {acc_mean:.4f}")

    # Test McNemar
    if model_B is not None:
        y_pred_B = model_B.predict(X_test)
        stat, p = mcnemar_test(y_test, y_pred, y_pred_B)
        print("\nTest de McNemar:")
        print(f"Estadístico = {stat:.4f}")
        print(f"p-value = {p:.4f}")

    return accuracy, y_pred, (ci_low, ci_high)


# ============================================================
# 6. GUARDAR MODELO
# ============================================================
def save_model(model, scaler, feature_names, target_names, accuracy, output_dir="models"):

    print("Guardando modelo...")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "target_names": target_names,
        "accuracy": accuracy,
        "timestamp": timestamp,
        "model_type": "RandomForestClassifier",
        "dataset": "ConsumoAlcohol"
    }

    model_path = os.path.join(output_dir, f"consumo_model_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    latest_path = os.path.join(output_dir, "consumo_model_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"✓ Modelo guardado en: {model_path}")
    return model_path


# ============================================================
# 7. MAIN
# ============================================================
def main():
    print("=" * 50)
    print("SCRIPT DE ENTRENAMIENTO - CONSUMO DE ALCOHOL")
    print("=" * 50)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/app/data/train.csv")
    args = parser.parse_args()

    try:
        X, y, feature_names, target_names = load_data(filepath=args.data)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        model = train_model(X_train, y_train)

        accuracy, y_pred, ci = evaluate_model(model, X_test, y_test, target_names)
        print(f"CI Bootstrap 95%: {ci}")

        model_path = save_model(model, scaler, feature_names, target_names, accuracy)

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise


if __name__ == "__main__":
    main()

