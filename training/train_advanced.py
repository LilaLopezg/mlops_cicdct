#!/usr/bin/env python3
"""
train_advanced.py

Pipeline avanzado para:
 - Fase 1: SMOTEENN, BalancedRandomForest, class weights optimizados
 - Fase 2: XGBoost, LightGBM, CatBoost
 - Fase 3: RandomizedSearchCV + Optuna (bayesian)
 - Fase 4: Curva Precision-Recall y SHAP
 - Fase 5: ajuste de threshold (max F1, PR-based)
 - Fase 6: guardar el mejor pipeline (scaler + modelo + threshold + metadata)

Usage:
    python train_advanced.py --data /mnt/data/train.csv --out models/
"""

import argparse
import json
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_curve, auc, f1_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.ensemble import BalancedRandomForestClassifier

# Model libraries
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# Optuna
import optuna

# SHAP
import shap

import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Utilities
# -------------------------
def load_data(path):
    print("Cargando datos desde:", path)
    df = pd.read_csv(path, sep=";")
    if "Consumo" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'Consumo' como target.")
    X = df.drop(columns=["Consumo"]).select_dtypes(include=["number"])
    y = df["Consumo"]
    return X, y

def train_test_split_strat(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# -------------------------
# Resampling / Preprocess
# -------------------------
def build_preprocessor(scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    return scaler

def apply_smoteenn(X_train, y_train, random_state=42):
    print("Aplicando SMOTEENN...")
    sm = SMOTEENN(random_state=random_state)
    Xr, yr = sm.fit_resample(X_train, y_train)
    print("Antes:", len(y_train), "Despues:", len(yr))
    print(pd.Series(yr).value_counts())
    return Xr, yr

# -------------------------
# Search spaces
# -------------------------
import scipy.stats as ss

RF_PARAM_DIST = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

LGB_PARAM_DIST = {
    "num_leaves": [31, 63, 127],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0]
}

CAT_PARAM_DIST = {
    "iterations": [200, 400],
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1]
}

# -------------------------
# RandomizedSearch wrapper
# -------------------------
def randomized_search(model, param_dist, X, y, cv=3, n_iter=20, scoring="f1_macro", n_jobs=1, random_state=42):
    print(f"Running RandomizedSearchCV on {model.__class__.__name__}...")
    rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs, random_state=random_state, verbose=1)
    rs.fit(X, y)
    print("Best score:", rs.best_score_)
    print("Best params:", rs.best_params_)
    return rs

# -------------------------
# Optuna objective (example for XGBoost)
# -------------------------
def optuna_objective_xgb(trial, X, y, cv_splits=3):
    param = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100,200,300]),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.01, 1.0),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        Xtr, Xv = X.iloc[train_idx], X.iloc[valid_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[valid_idx]
        model = XGBClassifier(**param)
        model.fit(Xtr, ytr)
        preds = model.predict(Xv)
        scores.append(f1_score(yv, preds, average="macro"))
    return np.mean(scores)

def run_optuna_xgb(X, y, n_trials=25):
    study = optuna.create_study(direction="maximize")
    func = lambda trial: optuna_objective_xgb(trial, X, y)
    study.optimize(func, n_trials=n_trials, show_progress_bar=True)
    print("Optuna best params:", study.best_params)
    return study.best_params

# -------------------------
# Evaluate at different thresholds and metrics
# -------------------------
def evaluate_thresholds(model, X_test, y_test, thresholds=[0.5, 0.6, 0.7]):
    proba = model.predict_proba(X_test)[:,1]
    results = {}
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        results[thr] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }
    return results, proba

def find_best_threshold_by_f1(proba, y_true):
    precisions, recalls, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1s)
    return thr[best_idx], precisions[best_idx], recalls[best_idx], f1s[best_idx]

def find_threshold_by_pr_curve(proba, y_true, target_precision=None, target_recall=None):
    precisions, recalls, thr = precision_recall_curve(y_true, proba)
    if target_precision is not None:
        # choose threshold achieving at least target_precision while maximizing recall
        mask = precisions >= target_precision
        if mask.any():
            idx = np.argmax(recalls[mask])
            chosen_idx = np.where(mask)[0][idx]
            return thr[chosen_idx], precisions[chosen_idx], recalls[chosen_idx]
    if target_recall is not None:
        mask = recalls >= target_recall
        if mask.any():
            idx = np.argmax(precisions[mask])
            chosen_idx = np.where(mask)[0][idx]
            return thr[chosen_idx], precisions[chosen_idx], recalls[chosen_idx]
    # fallback: return 0.5
    return 0.5, precisions[-1], recalls[-1]

# -------------------------
# SHAP plotting helper
# -------------------------
def compute_shap_and_save(model, X_sample, out_dir, model_name):
    print("Calculando SHAP (puede tardar)...")
    explainer = shap.Explainer(model.predict_proba, X_sample)  # uses model's predict_proba
    shap_values = explainer(X_sample)
    plt.figure(figsize=(8,6))
    shap.plots.beeswarm(shap_values)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"shap_beeswarm_{model_name}.png")
    plt.savefig(out_png)
    plt.close()
    return out_png

# -------------------------
# Main pipeline
# -------------------------
def main(args):
    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split_strat(X, y, test_size=0.2, random_state=42)

    scaler = build_preprocessor()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Phase 1: SMOTEENN resample
    X_bal, y_bal = apply_smoteenn(pd.DataFrame(X_train_s, columns=X_train.columns), y_train)

    # Phase 1 alternative: BalancedRandomForest baseline
    brf = BalancedRandomForestClassifier(n_estimators=200, random_state=42)
    brf.fit(X_bal, y_bal)
    print("BalancedRandomForest - evaluación (default threshold 0.5):")
    preds_brf = brf.predict(X_test_s)
    print(classification_report(y_test, preds_brf))

    # Phase 2 + 3: try RandomizedSearch for RF, XGB, LGB, CatBoost
    results = {}

    # RandomForest (use balanced RF or classic RF)
    rf_base = RandomForestClassifier(random_state=42)
    rf_rs = randomized_search(rf_base, RF_PARAM_DIST, X_bal, y_bal, cv=3, n_iter=10, n_jobs=args.n_jobs)
    results['RandomForest'] = rf_rs

    # XGBoost randomized
    xgb_base = XGBClassifier(objective="binary:logistic", use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb_rs = randomized_search(xgb_base, XGB_PARAM_DIST, X_bal, y_bal, cv=3, n_iter=10, n_jobs=args.n_jobs)
    results['XGBoost'] = xgb_rs

    # LightGBM randomized (use sklearn wrapper)
    lgb_base = lgb.LGBMClassifier(random_state=42)
    lgb_rs = randomized_search(lgb_base, LGB_PARAM_DIST, X_bal, y_bal, cv=3, n_iter=10, n_jobs=args.n_jobs)
    results['LightGBM'] = lgb_rs

    # CatBoost randomized (silent)
    cat_base = CatBoostClassifier(silent=True, random_state=42)
    cat_rs = randomized_search(cat_base, CAT_PARAM_DIST, X_bal, y_bal, cv=3, n_iter=6, n_jobs=args.n_jobs)
    results['CatBoost'] = cat_rs

    # Phase 3: Optuna for XGBoost (optional)
    if args.optuna_trials and args.optuna_trials > 0:
        print("Running Optuna for XGBoost...")
        best_params_xgb = run_optuna_xgb(pd.DataFrame(X_bal, columns=X_train.columns), y_bal, n_trials=args.optuna_trials)
        # train final xgb with best params
        xgb_final = XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric="logloss", random_state=42)
        xgb_final.fit(X_bal, y_bal)
        results['XGBoost_Optuna'] = xgb_final

    # Phase 4 & 5: Evaluate models, thresholds, PR curves
    os.makedirs(args.out, exist_ok=True)
    evaluation = {}
    candidate_models = {}

    # helpers to retrieve models (handle Grid/Randomized wrappers)
    def get_model(est):
        if hasattr(est, "best_estimator_"):
            return est.best_estimator_
        return est

    # collect trained models
    for name, est in results.items():
        candidate_models[name] = get_model(est)

    if 'XGBoost_Optuna' in results:
        candidate_models['XGBoost_Optuna'] = results['XGBoost_Optuna']

    # Evaluate each candidate
    for name, model in candidate_models.items():
        print("\nEvaluando candidato:", name)
        # Some models (CatBoost, XGBoost) might need DMatrix/array, but sklearn API works with wrappers
        if not hasattr(model, "predict_proba"):
            print("El modelo no tiene predict_proba, omitiendo.")
            continue

        thr_results, proba = evaluate_thresholds(model, X_test_s, y_test, thresholds=[0.5, 0.6, 0.7])
        best_thr_f1, p, r, f1v = find_best_threshold_by_f1(proba, y_test)
        pr_thr, pr_p, pr_r = find_threshold_by_pr_curve(proba, y_test, target_precision=None, target_recall=None)

        evaluation[name] = {
            "thresholds": thr_results,
            "best_thr_by_f1": {"thr": float(best_thr_f1), "precision": float(p), "recall": float(r), "f1": float(f1v)},
            "pr_thr": {"thr": float(pr_thr), "precision": float(pr_p), "recall": float(pr_r)}
        }

        # save PR curve
        precisions, recalls, th = precision_recall_curve(y_test, proba)
        pr_auc = auc(recalls, precisions)
        plt.figure()
        plt.plot(recalls, precisions, label=f"{name} (PR AUC={pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall curve - {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"pr_curve_{name}.png"))
        plt.close()

        # SHAP (use small subset for speed)
        try:
            sample_X = pd.DataFrame(X_test_s, columns=X_test.columns).sample(n=min(200, len(X_test)), random_state=42)
            shap_png = compute_shap_and_save(model, sample_X, args.out, name)
            evaluation[name]["shap"] = shap_png
        except Exception as e:
            print("SHAP failed for", name, ":", e)
            evaluation[name]["shap"] = None

    # Phase 6: select best model by F1 on chosen threshold or by accuracy
    # Here we will pick model with highest f1 at its best_thr_by_f1
    best_score = -1
    best_model_name = None
    best_model_obj = None
    best_threshold = 0.5

    for name, meta in evaluation.items():
        score = meta["best_thr_by_f1"]["f1"]
        if score > best_score:
            best_score = score
            best_model_name = name
            best_model_obj = candidate_models[name]
            best_threshold = meta["best_thr_by_f1"]["thr"]

    print(f"\nMejor modelo seleccionado: {best_model_name} con F1={best_score:.4f} @ thr={best_threshold:.3f}")

    # Save final pipeline: scaler + model + threshold + metadata
    final_pkg = {
        "model": best_model_obj,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "threshold": float(best_threshold),
        "train_time": datetime.now().isoformat(),
        "model_name": best_model_name,
        "metrics": evaluation
    }

    out_path = os.path.join(args.out, f"best_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    joblib.dump(final_pkg, out_path)
    print("Guardado pipeline en:", out_path)

    # Save evaluation summary
    with open(os.path.join(args.out, "evaluation_summary.json"), "w") as f:
        json.dump(evaluation, f, indent=2)

    print("FIN.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="/app/data/train.csv", help="Ruta al csv de entrenamiento (sep=';').")
    ap.add_argument("--out", type=str, default="models", help="Directorio para guardar modelos y artefactos.")
    ap.add_argument("--n_jobs", type=int, default=2, help="n_jobs para RandomizedSearchCV.")
    ap.add_argument("--optuna_trials", type=int, default=0, help="Número de trials Optuna (0 para skip).")
    args = ap.parse_args()
    main(args)
