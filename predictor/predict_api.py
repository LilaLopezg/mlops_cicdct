#!/usr/bin/env python3
"""
API de predicción para el modelo de consumo de alcohol.
Ahora soporta el modelo real con 71 variables.
"""

import pickle
import numpy as np
import os
import argparse
import json
import sys


FEATURE_NAMES = [
    "edad","sexo","semestre_academico_actual","EC_Padres","Estado_Civil",
    "nivel_escolar_m","nivel_escolar_p","n_hermanos","DummC","DumCRI",
    "DumNPNR","DummO","DummSP","DummPH","DummSA","DummHF","DummCS",
    "DummVS","DummyO","DumS","DumC","DumV","DumD","DumUL","SMMLV_familiar",
    "Estrato_social_vi","Estrato","trabaja_estudia","DumVS","DumVMP",
    "DumVCF","DumVP","DuoO","costo_estudios","DummyC","DuDu","DuuV","DuUs",
    "DusS","DunNA","antecedentes_escolar","satisfecho_est","reprobado_mat",
    "desertar_est","estudios_graduarse","futuro_prof","familiares_borrac",
    "amigos_borrac","Du_Al","DuMi","DuTi","DuOl","DuNi","DuAl","DuuM","DuT",
    "DuO","DuN","DuA","DuF","DuP","DuCT","DummS","DuNCA","DuCA","DuCF",
    "DuFD","DuSB","DumNCA","Religion","edad_consumo"
]

N_FEATURES = len(FEATURE_NAMES)


def load_model(model_path="models/consumo_model_latest.pkl"):
    """Carga el modelo entrenado"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_consumo(age, gender, parent_alcohol, academic_semester,
                    model_path="models/consumo_model_latest.pkl"):

    model_data = load_model(model_path)
    model = model_data["model"]
    scaler = model_data["scaler"]
    target_names = model_data.get("target_names", ["No consumo", "Consumo"])

    # Creamos vector de 71 features lleno de 0
    X = np.zeros((1, N_FEATURES))

    # Asignamos las variables entregadas
    X[0, 0] = age                      # edad
    X[0, 1] = gender                   # sexo
    X[0, 2] = academic_semester        # semestre_academico_actual
    X[0, 46] = parent_alcohol          # familiares_borrac  (posición correcta)

    # Escalamos
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]

    return {
        "input_used": {
            "edad": age,
            "sexo": gender,
            "semestre_academico_actual": academic_semester,
            "familiares_borrac": parent_alcohol,
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
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--env", action="store_true")
    parser.add_argument("--model_path", type=str,
                        default="models/consumo_model_latest.pkl")

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

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
