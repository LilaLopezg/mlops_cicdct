#!/usr/bin/env python3
"""
Script de entrenamiento para clasificación del dataset Iris
Genera un modelo entrenado y lo guarda en formato pickle
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime


def load_data():
    """Carga el dataset consumo de alcohol"""
    print("Cargando dataset consumo de alcohol...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
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


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entrena el modelo Random Forest"""
    print("Entrenando modelo Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Modelo entrenado exitosamente")
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evalúa el modelo entrenado"""
    print("Evaluando modelo...")
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, y_pred


def save_model(model, scaler, feature_names, target_names, accuracy, output_dir="models"):
    """Guarda el modelo y metadatos en formato pickle"""
    print("Guardando modelo...")
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear timestamp para el archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Preparar datos del modelo
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'target_names': target_names,
        'accuracy': accuracy,
        'timestamp': timestamp,
        'model_type': 'RandomForestClassifier',
        'dataset': 'Iris'
    }
    
    # Guardar modelo
    model_path = os.path.join(output_dir, f"iris_model_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # También guardar una versión "latest"
    latest_path = os.path.join(output_dir, "iris_model_latest.pkl")
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
    """Función principal del script de entrenamiento"""
    print("=" * 50)
    print("SCRIPT DE ENTRENAMIENTO - DATASET IRIS")
    print("=" * 50)
    
    try:
        # 1. Cargar datos
        X, y, feature_names, target_names = load_data()
        
        # 2. Preprocesar datos
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        # 3. Entrenar modelo
        model = train_model(X_train, y_train)
        
        # 4. Evaluar modelo
        accuracy, y_pred = evaluate_model(model, X_test, y_test, target_names)
        
        # 5. Guardar modelo
        model_path = save_model(model, scaler, feature_names, target_names, accuracy)
        
        print("\n" + "=" * 50)
        print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"✓ Accuracy final: {accuracy:.4f}")
        print(f"✓ Modelo guardado en: {model_path}")
        print("=" * 50)
        
        # Ejemplo de uso del modelo guardado
        print("\nEjemplo de uso del modelo guardado:")
        model_data = load_saved_model(model_path)
        loaded_model = model_data['model']
        loaded_scaler = model_data['scaler']
        
        # Hacer una predicción de ejemplo
        sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Ejemplo de flor Iris
        sample_scaled = loaded_scaler.transform(sample_data)
        prediction = loaded_model.predict(sample_scaled)
        probability = loaded_model.predict_proba(sample_scaled)
        
        print(f"Muestra: {sample_data[0]}")
        print(f"Predicción: {target_names[prediction[0]]}")
        print(f"Probabilidades: {probability[0]}")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise


if __name__ == "__main__":
    main()