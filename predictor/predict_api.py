#!/usr/bin/env python3
"""
API de predicción para el modelo de consumo de alcohol.
Permite hacer predicciones desde variables de entorno, argumentos o ejemplos.
"""

import pickle
import numpy as np
import os
import argparse
import json
import sys


def load_model(model_path="models/consumo_model_smoteenn_latest.pkl"):
    """Carga el modelo entrenado"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_consumo(age, gender, parent_alcohol, academic_semester,
                    model_path="models/consumo_model_smoteenn_latest.pkl"):

    model_data = load_model(model_path)
    model = model_data["model"]
    scaler = model_data["scaler"]
    target_names = model_data.get("target_names", ["No consumo", "Consumo"])

    X = np.array([[age, gender, parent_alcohol, academic_semester]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]

    return {
        "input": {
            "age": age,
            "gender": gender,
            "parent_alcohol": parent_alcohol,
            "academic_semester": academic_semester,
        },
        "prediction": target_names[pred],
        "prediction_index": int(pred),
        "probabilities": {target_names[i]: float(p) for i, p in enumerate(probs)},
        "confidence": float(max(probs)),
    }


def get_env_params():
    """Lee datos desde variables de entorno"""
    return (
        float(os.getenv("AGE", 18)),
        int(os.getenv("GENDER", 0)),
        int(os.getenv("PARENT_ALCOHOL", 0)),
        int(os.getenv("ACADEMIC_SEMESTER", 1)),
    )


def run_examples():
    """Ejemplos fijos"""
    print("\n================ EJEMPLOS ================\n")
    ejemplos = [
        ("Estudiante sin riesgo", (18, 0, 0, 1)),
        ("Estudiante con posible riesgo", (22, 1, 1, 6))
    ]
    for nombre, params in ejemplos:
        print(f"📌 {nombre}")
        res = predict_consumo(*params)
        print(json.dumps(res, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Predictor consumo de alcohol")
    parser.add_argument("--age", type=float)
    parser.add_argument("--gender", type=int)
    parser.add_argument("--parent_alcohol", type=int)
    parser.add_argument("--academic_semester", type=int)
    parser.add_argument("--env", action="store_true")
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--model_path", type=str,
                        default="models/consumo_model_smoteenn_latest.pkl")

    args = parser.parse_args()

    if args.examples:
        run_examples()
        return

    if args.env:
        params = get_env_params()
    else:
        params = (
            args.age or 18,
            args.gender or 0,
            args.parent_alcohol or 0,
            args.academic_semester or 1,
        )

    result = predict_consumo(*params, model_path=args.model_path)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n========== Predicción Consumo ==========")
        print(f"Entrada: {result['input']}")
        print(f"Predicción: {result['prediction']}")
        print(f"Confianza: {result['confidence']:.4f}")
        print("Probabilidades:")
        for k, v in result["probabilities"].items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
