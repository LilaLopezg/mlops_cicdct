#!/bin/bash

# Script para hacer predicciones usando el modelo entrenado en Docker
# Uso: ./predict_docker.sh <sepal_length> <sepal_width> <petal_length> <petal_width>

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar argumentos
if [ $# -ne 4 ]; then
    echo -e "${RED}❌ Error: Se requieren 4 argumentos${NC}"
    echo "Uso: $0 <sepal_length> <sepal_width> <petal_length> <petal_width>"
    echo "Ejemplo: $0 5.1 3.5 1.4 0.2"
    exit 1
fi

SEPAL_LENGTH=$1
SEPAL_WIDTH=$2
PETAL_LENGTH=$3
PETAL_WIDTH=$4

echo -e "${YELLOW}🔮 Haciendo predicción con Docker...${NC}"
echo -e "Parámetros: SL=${SEPAL_LENGTH}, SW=${SEPAL_WIDTH}, PL=${PETAL_LENGTH}, PW=${PETAL_WIDTH}"

# Verificar que el modelo existe
if [ ! -f "./models/consumo_model_smoteenn_latest.pkl" ]; then
    echo -e "${RED}❌ Error: No se encontró el modelo entrenado${NC}"
    echo -e "${YELLOW}💡 Ejecuta primero: ./build_and_runm.sh${NC}"
    exit 1
fi

# Ejecutar predicción en contenedor
docker run --rm \
    -v "$(pwd)/models:/app/models" \
    --name consumo-predict-container \
    consumo-training:latest \
    python predict.py \
    --sepal_length $SEPAL_LENGTH \
    --petal_length $PETAL_LENGTH \
    --petal_width $PETAL_WIDTH

echo -e "${GREEN}✅ Predicción completada${NC}"