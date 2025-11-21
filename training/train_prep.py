#!/usr/bin/env python3
"""
Script de entrenamiento para clasificación de enfermedades cardíacas
Genera un modelo entrenado y lo guarda en formato pickle
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
from datetime import datetime


def load_data(filepath="/Users/lila/Documents/mlops_cicdct/data/datos_preprocesados.csv"):
    """Carga el dataset de salud"""
    print("Cargando dataset de salud...")

    df = pd.read_csv(filepath)

    # Variable objetivo (Heart_Disease)
    y = df["Heart_Disease"]

    # Variables predictoras (todas menos Heart_Disease)
    X = df.drop("Heart_Disease", axis=1)

    # Separar categóricas y numéricas
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    feature_names = categorical_features + numeric_features
    target_names = sorted(y.unique().astype(str))  # ["No", "Yes"]

    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Clases: {target_names}")

    return X, y, categorical_features, numeric_features, feature_names, target_names


def preprocess_data(X, y, categorical_features, numeric_features, test_size=0.2, random_state=42):
    """Preprocesa datos con OneHot para categóricas y StandardScaler para numéricas"""
    print("Preprocesando datos...")

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Transformadores
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    # ColumnTransformer para combinar
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features)
        ]
    )

    # Ajustar en train y transformar ambos
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de prueba: {X_test.shape[0]} muestras")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def train_model(X_train, y_train, n_estimators=200, random_state=42):
    """Entrena el modelo Random Forest"""
    print("Entrenando modelo Random Forest...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1
    )

    model.fit(X_train, y_train)

    print("✓ Modelo entrenado exitosamente")
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evalúa el modelo entrenado"""
    print("Evaluando modelo...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy, y_pred


def save_model(model, preprocessor, feature_names, target_names, accuracy, output_dir="models"):
    """Guarda el modelo y metadatos en formato pickle"""
    print("Guardando modelo...")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'target_names': target_names,
        'accuracy': accuracy,
        'timestamp': timestamp,
        'model_type': 'RandomForestClassifier',
        'dataset': 'HeartDisease'
    }

    model_path = os.path.join(output_dir, f"heart_disease_model_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    latest_path = os.path.join(output_dir, "heart_disease_model_latest.pkl")
    with open(latest_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Modelo guardado en: {model_path}")
    print(f"✓ Modelo latest guardado en: {latest_path}")

    return model_path


def load_saved_model(model_path):
    """Función auxiliar para cargar un modelo guardado"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def main():
    print("=" * 50)
    print("SCRIPT DE ENTRENAMIENTO - ENFERMEDAD CARDÍACA")
    print("=" * 50)

    try:
        # 1. Cargar datos
        X, y, categorical_features, numeric_features, feature_names, target_names = load_data()

        # 2. Preprocesar datos
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(X, y, categorical_features, numeric_features)

        # 3. Entrenar modelo
        model = train_model(X_train, y_train)

        # 4. Evaluar modelo
        accuracy, y_pred = evaluate_model(model, X_test, y_test, target_names)

        # 5. Guardar modelo
        model_path = save_model(model, preprocessor, feature_names, target_names, accuracy)

        print("\n" + "=" * 50)
        print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"✓ Accuracy final: {accuracy:.4f}")
        print(f"✓ Modelo guardado en: {model_path}")
        print("=" * 50)

        # Ejemplo de predicción
        print("\nEjemplo de predicción con modelo guardado:")
        model_data = load_saved_model(model_path)
        loaded_model = model_data['model']
        loaded_preprocessor = model_data['preprocessor']

        sample = X.iloc[[0]]  # primera fila como ejemplo
        sample_processed = loaded_preprocessor.transform(sample)
        pred = loaded_model.predict(sample_processed)
        proba = loaded_model.predict_proba(sample_processed)

        print(f"Muestra: {sample.values[0]}")
        print(f"Predicción: {pred[0]}")
        print(f"Probabilidades: {proba[0]}")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise


if __name__ == "__main__":
    main()
