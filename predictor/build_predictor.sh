#!/bin/bash
# Script para construir la imagen Docker del predictor de consumo de alcohol

set -e

echo "🏗️  Construyendo imagen Docker del predictor de Consumo de Alcohol..."

# Movernos a la raíz del proyecto
cd "$(dirname "$0")/.."

# Ruta donde se espera que esté el modelo ya entrenado
MODEL_SRC_PATH="training/models/consumo_model_latest.pkl"
MODEL_DEST_PATH="predictor/models/consumo_model_latest.pkl"

# Verificar que existe el modelo entrenado
if [ ! -f "$MODEL_SRC_PATH" ]; then
    echo "❌ Error: No se encontró el modelo en: $MODEL_SRC_PATH"
    echo "   Asegúrate de haber ejecutado el entrenamiento:"
    echo "     python training/train_advanced.py --data ... --out training/models"
    exit 1
fi

# Copiar el modelo al predictor
mkdir -p predictor/models
cp "$MODEL_SRC_PATH" "$MODEL_DEST_PATH"

# Construir imagen Docker
docker build -f predictor/Dockerfile -t consumo-predictor:latest .

echo "✅ Imagen construida exitosamente: consumo-predictor:latest"
echo ""
echo "Para ejecutar el contenedor:"
echo "  predictor/run_predictor.sh"
echo ""
echo "Para hacer predicciones manualmente dentro del contenedor:"
echo "  docker run --rm consumo-predictor:latest python predict.py \\"
echo "       --age 21 --gender 1 --parent_alcohol 0 --academic_semester 4"

