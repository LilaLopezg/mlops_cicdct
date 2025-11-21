#!/bin/bash
# Script para ejecutar el contenedor de predicción

set -e

echo "🚀 Ejecutando predictor Iris en Docker..."

# Verificar que la imagen existe
if ! docker image inspect iris-predictor:latest >/dev/null 2>&1; then
    echo "❌ Error: La imagen iris-predictor:latest no existe"
    echo "   Construye primero la imagen: ./predictor/build_predictor.sh"
    exit 1
fi

# Ejecutar contenedor con ejemplos por defecto
echo "Ejecutando ejemplos de predicción..."
docker run --rm iris-predictor:latest

echo ""
echo "✅ Predicción completada"
echo ""
echo "Otros comandos útiles:"
echo ""
echo "🌟 Predicción personalizada:"
echo "   docker run --rm iris-predictor:latest python predict_api.py --sepal_length 6.0 --sepal_width 3.0 --petal_length 4.5 --petal_width 1.5"
echo ""
echo "🌟 Usando variables de entorno:"
echo "   docker run --rm -e SEPAL_LENGTH=6.0 -e SEPAL_WIDTH=3.0 -e PETAL_LENGTH=4.5 -e PETAL_WIDTH=1.5 iris-predictor:latest python predict_api.py --env"
echo ""
echo "🌟 Salida en JSON:"
echo "   docker run --rm iris-predictor:latest python predict_api.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2 --json"