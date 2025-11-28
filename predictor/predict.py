#!/usr/bin/env python3
"""
Script para hacer predicciones usando el modelo de consumo de alcohol entrenado.
"""

import pickle
import numpy as np
import os
import argparse
import pandas as pd


def load_model(model_path="models/consumo_model_latest.pkl"):
    """Carga el modelo entrenado desde el archivo pickle"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def predict_consumo(age, gender, parent_alcohol, academic_semester,
                    model_path="models/consumo_model_latest.pkl"):
    """
    Realiza una predicción del riesgo/nivel de consumo de alcohol.
    """

    # Cargar el modelo
    model_data = load_model(model_path)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]  # Las 71 columnas originales
    target_names = model_data.get("target_names", ["No consumo", "Consumo"])

    # Crear un diccionario con todas las features
    input_dict = {feature: 0 for feature in feature_names}

    # Sobrescribir solo las que vienen por CLI
    # Los nombres deben coincidir exactamente con el dataset original
    replacements = {
        "edad": age,
        "sexo": gender,
        "EC_Padres": parent_alcohol,
        "semestre_academico_actual": academic_semester
    }

    for key, val in replacements.items():
        if key in input_dict:
            input_dict[key] = val
        else:
            print(f"⚠ ADVERTENCIA: La columna '{key}' NO existe en las features del modelo.")

    # Convertir a DataFrame con las 71 columnas
    X = pd.DataFrame([input_dict])

    # Escalar
    X_scaled = scaler.transform(X)

    # Predicción
    pred = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    return {
        "input": replacements,
        "prediction": target_names[pred],
        "prediction_index": int(pred),
        "probabilities": {
            target_names[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        "confidence": float(max(probabilities))
    }


def main():
    parser = argparse.ArgumentParser(description="Predicción de consumo de alcohol")

    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--gender", type=int, required=True)
    parser.add_argument("--parent_alcohol", type=int, required=True)
    parser.add_argument("--academic_semester", type=int, required=True)
    parser.add_argument("--model_path", type=str, default="models/consumo_model_latest.pkl")

    args = parser.parse_args()

    try:
        result = predict_consumo(
            args.age,
            args.gender,
            args.parent_alcohol,
            args.academic_semester,
            args.model_path
        )

        print("=" * 50)
        print("PREDICCIÓN DE CONSUMO DE ALCOHOL")
        print("=" * 50)
        print(f"Edad:                {args.age}")
        print(f"Género:              {args.gender}")
        print(f"Padres con alcohol:  {args.parent_alcohol}")
        print(f"Semestre académico:  {args.academic_semester}")
        print()
        print(f"Predicción: {result['prediction']}")
        print(f"Confianza: {result['confidence']:.4f}")
        print("\nProbabilidades:")
        for clase, prob in result['probabilities'].items():
            print(f"  {clase}: {prob:.4f}")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

