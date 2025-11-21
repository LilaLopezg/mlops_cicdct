#!/bin/bash
# Script para construir la imagen Docker del predictor

set -e

echo "🏗️  Construyendo imagen Docker del predictor Iris..."

# Construir desde la raíz del proyecto para acceder a la carpeta models
cd "$(dirname "$0")/.."

# Verificar que existe el modelo
if [ ! -f "models/iris_model_latest.pkl" ]; then
    echo "❌ Error: No se encontró el modelo iris_model_latest.pkl en la carpeta models/"
    echo "   Ejecuta primero el entrenamiento: cd training && python train.py"
    exit 1
fi

mkdir -p predictor/models && cp models/iris_model_latest.pkl predictor/models/

# Construir imagen
docker build -f predictor/Dockerfile -t iris-predictor:latest .

echo "✅ Imagen construida exitosamente: iris-predictor:latest"
echo ""
echo "Para ejecutar el contenedor:"
echo "  predictor/run_predictor.sh"
echo ""
echo "Para hacer predicciones personalizadas:"
echo "  docker run --rm iris-predictor:latest python predict_api.py --sepal_length 6.0 --sepal_width 3.0 --petal_length 4.5 --petal_width 1.5"