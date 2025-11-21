#!/bin/bash

# Script para construir y ejecutar el contenedor de entrenamiento
# Uso: ./build_and_run.sh

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🐳 Construyendo imagen Docker para entrenamiento de Iris...${NC}"

# Construir imagen
docker build -t iris-training:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Imagen construida exitosamente${NC}"
else
    echo -e "${RED}❌ Error construyendo la imagen${NC}"
    exit 1
fi

echo -e "${YELLOW}🚀 Ejecutando contenedor de entrenamiento...${NC}"

# Crear directorio local para modelos si no existe
mkdir -p ./models

# Ejecutar contenedor con volumen montado para persistir modelos
docker run --rm \
    -v "$(pwd)/models:/app/models" \
    --name iris-training-container \
    iris-training:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Entrenamiento completado exitosamente${NC}"
    echo -e "${GREEN}📁 El modelo se ha guardado en ./models/${NC}"
    ls -la ./models/
else
    echo -e "${RED}❌ Error durante el entrenamiento${NC}"
    exit 1
fi