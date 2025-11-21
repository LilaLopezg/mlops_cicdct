#!/usr/bin/env python3
"""
Script de entrenamiento para clasificación del consumo de alcohol
Incluye diferentes técnicas de balanceo (SMOTE, SMOTETomek, SMOTEENN, TomekLinks).
Genera modelos entrenados y los guarda en formato pickle.
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
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks


def load_data(filepath="/Users/lila/Documents/mlops_cicdct/data/train.csv"):
    """Carga el dataset de consumo de alcohol"""
    print("Cargando dataset de consumo de alcohol...")
    
    df = pd.read_csv(filepath, sep=";")
    
    # Separar variables predictoras y target
    X = df.drop("Consumo", axis=1).select_dtypes(include=["number"])
    y = df["Consumo"]
    
    feature_names = list(X.columns)
    target_names = sorted(y.unique().astype(str))
    
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases: {target_names}")
    
    return X, y, feature_names, target_names


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Preprocesa los datos y los divide en conjuntos de entrenamiento y prueba"""
    print("Preprocesando datos...")
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de prueba: {X_test.shape[0]} muestras")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_resampling(X_train, y_train, method="none"):
    """Aplica diferentes técnicas de balanceo"""
    print(f"\nAplicando técnica de balanceo: {method}")
    
    if method == "smote":
        sampler = SMOTE(random_state=42)
    elif method == "smotetomek":
        sampler = SMOTETomek(random_state=42)
    elif method == "smoteenn":
        sampler = SMOTEENN(random_state=42)
    elif method == "tomek":
        sampler = TomekLinks()
    else:
        return X_train, y_train  # sin balanceo
    
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"Dataset balanceado: {X_res.shape[0]} muestras")
    return X_res, y_res


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entrena el modelo Random Forest"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evalúa el modelo entrenado"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, y_pred


def save_model(model, scaler, feature_names, target_names, accuracy, method, output_dir="models"):
    """Guarda el modelo y metadatos en formato pickle"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'target_names': target_names,
        'accuracy': accuracy,
        'timestamp': timestamp,
        'model_type': 'RandomForestClassifier',
        'dataset': 'ConsumoAlcohol',
        'resampling': method
    }
    
    model_path = os.path.join(output_dir, f"consumo_model_{method}_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    latest_path = os.path.join(output_dir, f"consumo_model_{method}_latest.pkl")
    with open(latest_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Modelo ({method}) guardado en: {model_path}")
    return model_path


def main():
    print("=" * 50)
    print("SCRIPT DE ENTRENAMIENTO - CONSUMO DE ALCOHOL")
    print("=" * 50)
    
    try:
        # 1. Cargar datos
        X, y, feature_names, target_names = load_data()
        
        # 2. Preprocesar datos
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        # 3. Probar diferentes métodos de balanceo
        methods = ["none", "smote", "smotetomek", "smoteenn", "tomek"]
        
        for method in methods:
            print("\n" + "-" * 50)
            print(f"Entrenando con método: {method.upper()}")
            
            # Aplicar balanceo
            X_train_res, y_train_res = apply_resampling(X_train, y_train, method)
            
            # Entrenar modelo
            model = train_model(X_train_res, y_train_res)
            
            # Evaluar modelo
            accuracy, _ = evaluate_model(model, X_test, y_test, target_names)
            
            # Guardar modelo
            save_model(model, scaler, feature_names, target_names, accuracy, method)
        
        print("\n" + "=" * 50)
        print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise


if __name__ == "__main__":
    main()
