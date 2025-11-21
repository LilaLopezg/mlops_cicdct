#!/bin/bash
set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🐳 Construyendo imagen Docker para entrenamiento de Consumo de Alcohol...${NC}"
docker build -t contenedor_entrenamiento_consumo_alcohol --platform=linux/amd64 python:3.11-slim -f Dockerfile .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Imagen construida exitosamente${NC}"
else
    echo -e "${RED}❌ Error construyendo la imagen${NC}"
    exit 1
fi

echo -e "${YELLOW}🚀 Ejecutando contenedor de entrenamiento...${NC}"
mkdir -p ../models

docker run --rm \
    -v "$(pwd)/../data:/app/data" \
    -v "$(pwd)/../models:/app/models" \
    contenedor_entrenamiento_consumo_alcohol -t 
