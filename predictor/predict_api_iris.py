#!/usr/bin/env python3
"""
API de predicción para el modelo Consumo
Permite hacer predicciones desde variables de entorno o argumentos
"""

import pickle
import numpy as np
import os
import argparse
import json
import sys


def load_model(model_path="models/consumo_model_smoteenn_latest.pkl"):
    """Carga el modelo entrenado desde el archivo pickle"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def predict_iris(sepal_length, sepal_width, petal_length, petal_width, model_path="models/consumo_model_smoteenn_latest.pkl"):
    """
    Hace una predicción para una muestra de Iris
    
    Args:
        sepal_length: Longitud del sépalo
        sepal_width: Ancho del sépalo  
        petal_length: Longitud del pétalo
        petal_width: Ancho del pétalo
        model_path: Ruta al modelo guardado
    
    Returns:
        dict: Predicción y probabilidades
    """
    # Cargar modelo
    model_data = load_model(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    target_names = model_data['target_names']
    
    # Preparar datos de entrada
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    
    # Hacer predicción
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]
    
    # Preparar resultado
    result = {
        'input': {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        },
        'prediction': target_names[prediction],
        'prediction_index': int(prediction),
        'probabilities': {
            target_names[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        'confidence': float(max(probabilities))
    }
    
    return result


def get_prediction_from_env():
    """Obtiene parámetros de predicción desde variables de entorno"""
    try:
        sepal_length = float(os.getenv('SEPAL_LENGTH', '5.1'))
        sepal_width = float(os.getenv('SEPAL_WIDTH', '3.5'))
        petal_length = float(os.getenv('PETAL_LENGTH', '1.4'))
        petal_width = float(os.getenv('PETAL_WIDTH', '0.2'))
        
        return sepal_length, sepal_width, petal_length, petal_width
    except ValueError as e:
        raise ValueError(f"Error al leer variables de entorno: {e}")


def run_prediction_examples():
    """Ejecuta ejemplos de predicción para cada clase"""
    print("=" * 60)
    print("EJEMPLOS DE PREDICCIÓN - DATASET CONSUMO DE ALCOHOL")
    print("=" * 60)
    
    examples = [
        {
            'name': 'Setosa (esperado)',
            'params': (5.1, 3.5, 1.4, 0.2)
        },
        {
            'name': 'Versicolor (esperado)',
            'params': (6.4, 3.2, 4.5, 1.5)
        },
        {
            'name': 'Virginica (esperado)',
            'params': (6.3, 3.3, 6.0, 2.5)
        }
    ]
    
    for example in examples:
        print(f"\n📊 {example['name']}:")
        print("-" * 40)
        
        result = predict_iris(*example['params'])
        
        print(f"Entrada: {example['params']}")
        print(f"Predicción: {result['prediction']}")
        print(f"Confianza: {result['confidence']:.4f}")
        print("Probabilidades:")
        for species, prob in result['probabilities'].items():
            print(f"  {species}: {prob:.4f}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Predictor deL consumo de alcohol')
    parser.add_argument('--sepal_length', type=float, help='Longitud del sépalo')
    parser.add_argument('--sepal_width', type=float, help='Ancho del sépalo')
    parser.add_argument('--petal_length', type=float, help='Longitud del pétalo')
    parser.add_argument('--petal_width', type=float, help='Ancho del pétalo')
    parser.add_argument('--model_path', type=str, default='models/iris_model_latest.pkl', help='Ruta al modelo')
    parser.add_argument('--examples', action='store_true', help='Ejecutar ejemplos de predicción')
    parser.add_argument('--json', action='store_true', help='Salida en formato JSON')
    parser.add_argument('--env', action='store_true', help='Leer parámetros desde variables de entorno')
    
    args = parser.parse_args()
    
    try:
        # Verificar que el modelo existe
        if not os.path.exists(args.model_path):
            print(f"❌ Error: No se encontró el modelo en {args.model_path}")
            sys.exit(1)
        
        # Ejecutar ejemplos si se solicita
        if args.examples:
            run_prediction_examples()
            return
        
        # Obtener parámetros
        if args.env:
            sepal_length, sepal_width, petal_length, petal_width = get_prediction_from_env()
        elif all([args.sepal_length is not None, args.sepal_width is not None, 
                 args.petal_length is not None, args.petal_width is not None]):
            sepal_length, sepal_width, petal_length, petal_width = (
                args.sepal_length, args.sepal_width, args.petal_length, args.petal_width
            )
        else:
            # Usar valores por defecto (ejemplo Setosa)
            sepal_length, sepal_width, petal_length, petal_width = 5.1, 3.5, 1.4, 0.2
            print("⚠️  Usando valores por defecto (ejemplo Setosa)")
        
        # Hacer predicción
        result = predict_iris(sepal_length, sepal_width, petal_length, petal_width, args.model_path)
        
        if args.json:
            # Salida JSON
            print(json.dumps(result, indent=2))
        else:
            # Salida formateada
            print("=" * 50)
            print("🌸 PREDICCIÓN DE ESPECIE DE IRIS")
            print("=" * 50)
            print(f"📥 Entrada:")
            print(f"   Longitud sépalo: {result['input']['sepal_length']} cm")
            print(f"   Ancho sépalo:    {result['input']['sepal_width']} cm")
            print(f"   Longitud pétalo: {result['input']['petal_length']} cm")
            print(f"   Ancho pétalo:    {result['input']['petal_width']} cm")
            print()
            print(f"🎯 Predicción: {result['prediction']}")
            print(f"🎲 Confianza: {result['confidence']:.4f}")
            print()
            print("📊 Probabilidades por clase:")
            for species, prob in result['probabilities'].items():
                bar_length = int(prob * 20)  # Barra de progreso simple
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   {species:12}: {prob:.4f} {bar}")
            print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()