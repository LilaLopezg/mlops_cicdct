#!/usr/bin/env python3
"""
Script para hacer predicciones usando el modelo de consumo de alcohol entrenado.
"""

import pickle
import numpy as np
import os
import argparse


def load_model(model_path="models/consumo_model_smoteenn_latest.pkl"):
    """Carga el modelo entrenado desde el archivo pickle"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def predict_consumo(age, gender, parent_alcohol, academic_semester,
                    model_path="models/consumo_model_smoteenn_latest.pkl"):
    """
    Realiza una predicción del riesgo/nivel de consumo de alcohol.
    
    Args:
        age (float): Edad del estudiante
        gender (int): Género (0=femenino, 1=masculino)
        parent_alcohol (int): Historial de alcoholismo en padres (0/1)
        academic_semester (int): Semestre académico

    Returns:
        dict: Predicción, probabilidades y confianza
    """
    
    # Cargar modelo
    model_data = load_model(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    target_names = model_data.get("target_names", ["No consumo", "Consumo"])
    
    # Preparar datos de entrada
    sample = np.array([[age, gender, parent_alcohol, academic_semester]])
    sample_scaled = scaler.transform(sample)
    
    # Hacer predicción
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]
    
    # Preparar resultado
    return {
        'input': {
            'age': age,
            'gender': gender,
            'parent_alcohol': parent_alcohol,
            'academic_semester': academic_semester
        },
        'prediction': target_names[prediction],
        'prediction_index': int(prediction),
        'probabilities': {
            target_names[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        'confidence': float(max(probabilities))
    }


def main():
    """Función principal para línea de comandos"""
    parser = argparse.ArgumentParser(description='Predicción de consumo de alcohol')

    parser.add_argument('--age', type=float, required=True, help='Edad del estudiante')
    parser.add_argument('--gender', type=int, required=True, help='Género (0=femenino, 1=masculino)')
    parser.add_argument('--parent_alcohol', type=int, required=True, help='Padres con historial de alcoholismo (0/1)')
    parser.add_argument('--academic_semester', type=int, required=True, help='Semestre académico actual (1..10)')
    parser.add_argument('--model_path', type=str, default='models/consumo_model_smoteenn_latest.pkl',
                        help='Ruta al modelo entrenado')

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
    print("Entrada:")
    print(f"  Edad:                {result['input']['age']}")
    print(f"  Género:              {result['input']['gender']}")
    print(f"  Padres con alcohol:  {result['input']['parent_alcohol']}")
    print(f"  Semestre académico:  {result['input']['academic_semester']}")
    print()
    print(f"Predicción: {result['prediction']}")
    print(f"Confianza: {result['confidence']:.4f}")
    print()
    print("Probabilidades:")
    for clase, prob in result['probabilities'].items():
        print(f"  {clase}: {prob:.4f}")
    print("=" * 50)

except Exception as e:
    print(f"❌ Error: {e}")



if __name__ == "__main__":
    main()