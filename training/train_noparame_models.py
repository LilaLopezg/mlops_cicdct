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

# ============================
# MÉTRICAS NO PARAMÉTRICAS
# ============================
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample


def bootstrap_ci_accuracy(model, X_test, y_test, n_bootstrap=1000, alpha=0.05):
    """Bootstrap no paramétrico para intervalos de confianza de accuracy."""
    boot_accuracies = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_b = X_test[idx]
        y_b = y_test.iloc[idx]

        if hasattr(model, "predict_proba"):
            y_pred_b = (model.predict_proba(X_b)[:, 1] >= 0.5).astype(int)
        else:
            y_pred_b = model.predict(X_b)

        boot_accuracies.append(accuracy_score(y_b, y_pred_b))

    low = np.percentile(boot_accuracies, alpha/2*100)
    high = np.percentile(boot_accuracies, (1-alpha/2)*100)
    mean = np.mean(boot_accuracies)

    return low, high, mean


def mcnemar_test(y_true, pred_A, pred_B):
    """Aplica test de McNemar entre dos modelos."""
    table = confusion_matrix(pred_A == y_true, pred_B == y_true)
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue


# ============================
# 1. CARGA DE DATOS
# ============================
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


# ============================
# 2. PREPROCESAMIENTO
# ============================
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


# ============================
# 3. ENTRENAMIENTO
# ============================
def train_models(X_train, y_train, random_state=42):

    results = {}

    # RANDOM FOREST
    print("\n→ Optimizando RandomForest...")
    rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
    rf_params = {
        "n_estimators": [200, 300],
        "max_depth": [5, 7],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    results["RF"] = rf_grid

    # XGBOOST
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

    xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring="f1_macro",
                            n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, y_train)
    results["XGB"] = xgb_grid

    return results


# ============================
# THRESHOLD
# ============================
def predict_with_threshold(model, X, thr):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= thr).astype(int)


# ============================
# 4. EVALUACIÓN + THRESHOLD
# ============================
def evaluate_models(model_dict, X_test, y_test, target_names, model_B=None):

    print("\n=== EVALUACIÓN FINAL CON THRESHOLD ===")

    thresholds = [0.50, 0.70, 0.80]
    best_model = None
    best_acc = 0
    best_thr = 0
    best_name = ""
    best_y_pred = None

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

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_thr = thr
                best_name = name
                best_y_pred = y_pred

    print(f"\nMejor modelo: {best_name} con threshold={best_thr} accuracy={best_acc:.4f}")

    # =============================
    # MÉTRICAS NO PARAMÉTRICAS
    # =============================

    print("\n=== Bootstrap Accuracy 95% CI ===")
    ci_low, ci_high, acc_mean = bootstrap_ci_accuracy(best_model, X_test, y_test)
    print(f"Bootstrap CI 95%: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Bootstrap Mean: {acc_mean:.4f}")

    if model_B is not None:
        print("\n=== Test de McNemar ===")

        if hasattr(model_B, "predict_proba"):
            y_pred_B = (model_B.predict_proba(X_test)[:, 1] >= best_thr).astype(int)
        else:
            y_pred_B = model_B.predict(X_test)

        stat, p = mcnemar_test(y_test, best_y_pred, y_pred_B)

        print(f"Estadístico = {stat:.4f}")
        print(f"p-value = {p:.4f}")

    return best_model, best_acc, best_thr


# ============================
# GUARDAR MODELO
# ============================
def save_model(model, scaler, feature_names, target_names, accuracy, threshold):

    output_path = "models"
    os.makedirs(output_path, exist_ok=True)

    filename = f"{output_path}/consumo_model_{datetime.now():%Y%m%d_%H%M%S}.pkl"

    with open(filename, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "features": feature_names,
            "targets": target_names,
            "accuracy": accuracy,
            "threshold": threshold
        }, f)

    print(f"\n✓ Modelo guardado en: {filename}")


# ============================
# MAIN
# ============================
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
