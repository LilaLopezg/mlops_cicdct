#!/bin/bash
# Script para ejecutar el contenedor de predicción de Consumo de Alcohol

set -e

echo "🚀 Ejecutando predictor de Consumo de Alcohol en Docker..."

# Verificar que la imagen existe
if ! docker image inspect alcohol-predictor:latest >/dev/null 2>&1; then
    echo "❌ Error: La imagen alcohol-predictor:latest no existe"
    echo "   Construye primero la imagen con:"
    echo "   docker build -t alcohol-predictor ."
    exit 1
fi

# Ejecutar contenedor con ejemplos por defecto (--examples en predict_api.py)
echo "Ejecutando ejemplos de predicción..."
docker run --rm alcohol-predictor

echo ""
echo "✅ Predicción completada"
echo ""
echo "Otros comandos útiles:"
echo ""
echo "🌟 Predicción personalizada:"
echo "   docker run --rm alcohol-predictor python predict.py --age 21 --gender 1 --parent_alcohol 0 --academic_semester 4"
echo ""
echo "🌟 Usando variables de entorno dentro del contenedor:"
echo "   docker run --rm -e AGE=20 -e GENDER=1 -e PARENT_ALCOHOL=0 -e ACADEMIC_SEMESTER=3 alcohol-predictor"
echo ""
echo "🌟 Salida en JSON (si tu API lo soporta):"
echo "   docker run --rm alcohol-predictor python predict_api.py --age 21 --gender 1 --parent_alcohol 0 --academic_semester 4 --json"

