#!/usr/bin/env python3
"""
train_noparame_models.py

Entrenamiento de modelos (RandomForest, XGBoost) con:
 - SMOTEENN
 - GridSearchCV
 - Evaluación con thresholds (0.50, 0.70, 0.80)
 - Bootstrap CI para accuracy (no paramétrica)
 - Test de McNemar entre los dos modelos
 - Registro completo en MLflow (métricas, parámetros, conf matrix, artefactos, modelo)

Ejemplo:
  MLflow server debe estar corriendo (si usas local):
    mlflow server --host 127.0.0.1 --port 8080 --default-artifact-root ./mlruns

  Ejecutar:
    python training/train_noparame_models.py --data ./data/train.csv --out training/models --mlflow_uri http://127.0.0.1:8080
"""
import os
import json
import tempfile
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.combine import SMOTEENN
from sklearn.utils import resample

# Non-parametric / statistical tests
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon

# MLflow
import mlflow
import mlflow.sklearn

# ------------------------------
# Utilidades
# ------------------------------
def bootstrap_ci_accuracy(model, X_test, y_test, n_bootstrap=1000, alpha=0.05, random_state=42):
    """Bootstrap no paramétrico para intervalo de confianza del accuracy."""
    rng = np.random.RandomState(random_state)
    accuracies = []
    n = len(y_test)
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)  # sample with replacement
        Xs = X_test_arr[idx]
        ys = y_test_arr[idx]
        yhat = model.predict(Xs)
        accuracies.append(accuracy_score(ys, yhat))
    lower = np.percentile(accuracies, 100 * (alpha / 2.0))
    upper = np.percentile(accuracies, 100 * (1 - alpha / 2.0))
    return float(lower), float(upper), float(np.mean(accuracies)), accuracies  # devuelvo lista accuracies por si la queremos usar

def mcnemar_test(y_true, pred_A, pred_B):
    """Calcula McNemar statistic y p-value (exacto o aproximado según b+c)."""
    b = np.sum((pred_A == y_true) & (pred_B != y_true))
    c = np.sum((pred_A != y_true) & (pred_B == y_true))
    table = [[0, b],
             [c, 0]]
    result = mcnemar(table, exact=(b + c) < 25)
    return float(result.statistic), float(result.pvalue)

def plot_and_log_confusion(y_true, y_pred, labels, mlflow_run, thr_str):
    """Genera y guarda matriz de confusión como artefacto en mlflow."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (thr={thr_str})")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, f"confusion_thr_{thr_str}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="confusion_matrices")
    return cm

def save_model_local(model_data, out_dir, name_prefix="consumo_model"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{name_prefix}_{ts}.pkl"
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        import pickle
        pickle.dump(model_data, f)
    # latest
    latest = os.path.join(out_dir, f"{name_prefix}_latest.pkl")
    with open(latest, "wb") as f:
        pickle.dump(model_data, f)
    return path, latest

# ------------------------------
# Pipeline principal
# ------------------------------
def load_data(filepath):
    print(f"Cargando dataset desde: {filepath}")
    df = pd.read_csv(filepath, sep=";")
    X = df.drop("Consumo", axis=1).select_dtypes(include=["number"])
    y = df["Consumo"]
    feature_names = list(X.columns)
    target_names = sorted(y.unique().astype(str))
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases: {target_names}")
    return X, y, feature_names, target_names

def preprocess_data(X, y, test_size=0.2, random_state=42):
    print("Preprocesando datos: train/test + StandardScaler + SMOTEENN...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote_enn = SMOTEENN(random_state=random_state)
    X_train_bal, y_train_bal = smote_enn.fit_resample(X_train_s, y_train)

    print(f"Train original: {len(X_train)} , Balanceado: {len(X_train_bal)}")
    return X_train_bal, X_test_s, y_train_bal, y_test, scaler

def train_models(X_train, y_train, random_state=42, n_jobs=-1):
    results = {}

    # RandomForest
    print("→ GridSearch RandomForest")
    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    rf_params = {
        "n_estimators": [200, 300],
        "max_depth": [5, 7],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="f1_macro", n_jobs=n_jobs, verbose=1)
    rf_grid.fit(X_train, y_train)
    results["RF"] = rf_grid

    # XGBoost
    print("→ GridSearch XGBoost")
    class_ratio = pd.Series(y_train).value_counts()
    # avoid division by zero
    scale_pos = float(class_ratio.min() / class_ratio.max()) if class_ratio.max() > 0 else 1.0

    xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                        scale_pos_weight=scale_pos, use_label_encoder=False, random_state=random_state)
    xgb_params = {
        "n_estimators": [200, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    }
    xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring="f1_macro", n_jobs=n_jobs, verbose=1)
    xgb_grid.fit(X_train, y_train)
    results["XGB"] = xgb_grid

    return results

def predict_with_threshold(model, X, thr):
    """Produce etiquetas binaras aplicando threshold sobre la probabilidad de la clase '1'."""
    # model must support predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # fallback: decision -> sigmoid
        df = model.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-df))
    else:
        raise RuntimeError("El modelo no soporta predict_proba ni decision_function")
    return (proba >= thr).astype(int), proba

def evaluate_models_and_log(grids, X_test, y_test, target_names, scaler, feature_names, out_dir, mlflow_experiment, thresholds=[0.50, 0.70, 0.80], bootstrap_n=1000):
    best = {"name": None, "model": None, "thr": None, "acc": -1}
    mlflow.set_experiment(mlflow_experiment)

    for name, grid in grids.items():
        best_est = grid.best_estimator_
        best_params = grid.best_params_
        # Log model hyperparams in mlflow for trace
        mlflow.log_params({f"{name}_{k}": v for k, v in best_params.items()})
        for thr in thresholds:
            thr_str = f"{thr:.2f}"
            with mlflow.start_run(run_name=f"{name}_thr_{thr_str}"):
                mlflow.log_param("model_name", name)
                mlflow.log_param("threshold", thr)
                mlflow.log_param("dataset_rows", len(X_test))
                mlflow.log_param("feature_count", len(feature_names))
                mlflow.log_param("best_params", json.dumps(best_params))
                # predictions
                y_pred, proba = predict_with_threshold(best_est, X_test, thr)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                # log metrics
                mlflow.log_metric("accuracy", float(acc))
                mlflow.log_metric("precision", float(prec))
                mlflow.log_metric("recall", float(rec))
                mlflow.log_metric("f1_score", float(f1))
                # classic report
                cr = classification_report(y_test, y_pred, output_dict=True)
                mlflow.log_dict(cr, "classification_report.json")
                # confusion matrix (artifact)
                cm = plot_and_log_confusion(y_test, y_pred, target_names, mlflow.active_run(), thr_str)
                mlflow.log_metric("cm_00", int(cm[0,0])); mlflow.log_metric("cm_01", int(cm[0,1]))
                mlflow.log_metric("cm_10", int(cm[1,0])); mlflow.log_metric("cm_11", int(cm[1,1]))
                # bootstrap CI
                ci_low, ci_high, acc_mean, acc_samples = bootstrap_ci_accuracy(best_est, X_test, y_test, n_bootstrap=bootstrap_n)
                mlflow.log_metric("bootstrap_ci_low", ci_low)
                mlflow.log_metric("bootstrap_ci_high", ci_high)
                mlflow.log_metric("bootstrap_accuracy_mean", acc_mean)
                # save proba histogram as artifact
                fig, ax = plt.subplots(figsize=(5,3))
                ax.hist(proba, bins=25)
                ax.set_title("Probabilidades clase=1 (test)")
                tmp = tempfile.mkdtemp()
                ph = os.path.join(tmp, f"proba_hist_{name}_thr_{thr_str}.png")
                fig.tight_layout(); fig.savefig(ph); plt.close(fig)
                mlflow.log_artifact(ph, artifact_path="proba_histograms")
                # register best
                if acc > best["acc"]:
                    best.update({"name": name, "model": best_est, "thr": thr, "acc": acc,
                                 "ci": (ci_low, ci_high), "acc_samples": acc_samples})
                print(f"{name} (thr={thr_str}) acc={acc:.4f} f1={f1:.4f}")

    # Optionally compare RF vs XGB on test set with McNemar (if both exist)
    if "RF" in grids and "XGB" in grids:
        mA = grids["RF"].best_estimator_.predict(X_test)
        mB = grids["XGB"].best_estimator_.predict(X_test)
        stat, p = mcnemar_test(y_test, mA, mB)
        # log mc nemar
        with mlflow.start_run(run_name="mcnemar_test", nested=True):
            mlflow.log_metric("mcnemar_stat", stat)
            mlflow.log_metric("mcnemar_pvalue", p)
        print(f"McNemar: stat={stat:.4f}, p={p:.4f}")

    # Save best model and artifacts locally + log model to mlflow
    if best["model"] is not None:
        print(f"\nMejor modelo: {best['name']} (thr={best['thr']}) acc={best['acc']:.4f}")
        model_data = {
            "model": best["model"],
            "scaler": scaler,
            "feature_names": feature_names,
            "target_names": target_names,
            "accuracy": best["acc"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "threshold": best["thr"],
            "model_name": best["name"]
        }
        local_path, latest_path = save_model_local(model_data, out_dir)
        print(f"✓ Modelo guardado localmente en: {local_path}")
        # Log model to mlflow
        with mlflow.start_run(run_name=f"{best['name']}_final", nested=True):
            mlflow.sklearn.log_model(best["model"], artifact_path="model")
            mlflow.log_artifact(local_path, artifact_path="models_pickle")
            mlflow.log_param("threshold_best", best["thr"])
            mlflow.log_metric("accuracy_best", best["acc"])
            mlflow.log_metric("ci_low", best["ci"][0])
            mlflow.log_metric("ci_high", best["ci"][1])
    else:
        print("No se encontró un mejor modelo (best==None)")

    return best

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/train.csv", help="CSV de entrenamiento (separated by ; )")
    p.add_argument("--out", type=str, default="training/models", help="Directorio para guardar modelos")
    p.add_argument("--mlflow_uri", type=str, default="http://127.0.0.1:8080", help="URI de tracking MLflow")
    p.add_argument("--experiment", type=str, default="ConsumoAlcohol_NoParam", help="Nombre del experimento MLflow")
    p.add_argument("--n_jobs", type=int, default=-1, help="n_jobs para GridSearchCV")
    p.add_argument("--bootstrap_n", type=int, default=1000, help="Número de iteraciones bootstrap")
    return p.parse_args()

def main():
    args = parse_args()

    # Configurar mlflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    print(f"MLflow URI: {args.mlflow_uri}  Experiment: {args.experiment}")

    X, y, feature_names, target_names = load_data(args.data)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    grids = train_models(X_train, y_train, n_jobs=args.n_jobs)

    best = evaluate_models_and_log(grids, X_test, y_test, target_names, scaler, feature_names, args.out, args.experiment, thresholds=[0.50, 0.70, 0.80], bootstrap_n=args.bootstrap_n)

    print("Proceso finalizado.")

if __name__ == "__main__":
    main()
