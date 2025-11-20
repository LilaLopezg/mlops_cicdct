# MLOps CI/CD - Clasificación de consumo de alcohol en estudiantes universitarios

Este proyecto implementa un pipeline de entrenamiento y predicción para clasificar consumo de alcohol usando Machine Learning.

## Estructura del Proyecto

```
mlops_cicdct/
├── training/
│   ├── train.py       # Script de entrenamiento un modelo RandomForest sobre el consumo de alcohol, hace evaluación, guarda modelo y métricas.
│   ├── predict.py     # Script de predicción que toma argumentos (Ejemplo: longitudes, anchos) y hace predicción con el modelo entrenado.
│   └── requirements.txt
├── predictor/
│   ├── Dockerfile     # Dockerfile para predicción
│   ├── predict.py     # Script de predicción (copia)
│   ├── predict_api.py # API mejorada de predicción
│   ├── requirements.txt
│   ├── build_predictor.sh  # Script para construir imagen Docker del predictor y ejecutarla.
│   └── run_predictor.sh    # Script para ejecutar contenedor
├── models/            # Modelos entrenados (pickle)
├── docker-compose.yml # Configuración Docker Compose (define el servicio para levantar el predictor)
├── requirements.txt   # Dependencias de Python
└── README.md         # Este archivo
```
```bash
#Creo el entorno virtual - mlops
python3 --version
which python3
brew install python
python3.11 -m venv venv
source venv/bin/activate # Activo el aentorno virtual
deactivate #Desactivo el entorno virtual

```

## Instalación

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
## Uso

### 1. Entrenamiento del Modelo

Para entrenar un nuevo modelo:

```bash
cd training
python train_pre.py
```

Este script:
- Carga el dataset del consumo de alcohol
- Preprocesa los datos (escalado y división train/test)
- Aplico técnicas de balanceo
- Entrena un modelo Random Forest
- Evalúa el modelo y muestra métricas
- Guarda el modelo en formato pickle en `models/`

### 2. Entrenamiento con Docker 🐳

**Opción recomendada**: Usar Docker para un entorno aislado y reproducible.

```bash
cd training
#!/bin/bash
docker run -it  --platform linux/amd64 python:3.11-slim /bin/bash #Ejecuto la imagen manualmente del contenedor de python, sobre una arquitectura de donde amd.

echo "🐳 Construyendo imagen Docker para entrenamiento de Consumo de Alcohol..."
docker build -t contenedor_entrenamiento_consumo_alcohol . # Se construye la imagen

echo "🚀 Ejecutando contenedor..."
docker run --rm consumo_alcohol_entrenamiento
```

Este script:
- Construye la imagen Docker con Python 3.13
- Ejecuta el entrenamiento automáticamente
- Guarda el modelo en `./models/` (montado como volumen)

### 3. Hacer Predicciones 🔮

#### Opción A: Python Local
```bash
cd training
python predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2
```

#### Opción B: Docker Predictor (Recomendado) 🐳

**1. Construir imagen del predictor:**
```bash
./predictor/build_predictor.sh
```

**2. Ejecutar ejemplos de predicción:**
```bash
./predictor/run_predictor.sh
```

**3. Predicción personalizada:**
```bash
docker run --rm consumo-predictor:latest python predict_api.py --sepal_length 6.0 --sepal_width 3.0 --petal_length 4.5 --petal_width 1.5
```

**4. Con variables de entorno:**
```bash
docker run --rm -e SEPAL_LENGTH=6.4 -e SEPAL_WIDTH=3.2 -e PETAL_LENGTH=4.5 -e PETAL_WIDTH=1.5 consumo-predictor:latest python predict_api.py --env
```

**5. Con Docker Compose:**
```bash
# Ejecutar ejemplos
docker-compose up iris-predictor

# Predicción personalizada (editar variables en docker-compose.yml)
docker-compose --profile custom up iris-predictor-custom
```

### 4. Ejemplos de Predicción

**Setosa (esperado):**
```bash
python predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2
```

**Versicolor (esperado):**
```bash
python predict.py --sepal_length 6.4 --sepal_width 3.2 --petal_length 4.5 --petal_width 1.5
```

**Virginica (esperado):**
```bash
python predict.py --sepal_length 6.3 --sepal_width 3.3 --petal_length 6.0 --petal_width 2.5
```

## Dataset Iris

El dataset Iris contiene 150 muestras de tres especies de flores:
- **Setosa**: 50 muestras
- **Versicolor**: 50 muestras  
- **Virginica**: 50 muestras

Cada muestra tiene 4 características:
- Longitud del sépalo (sepal length)
- Ancho del sépalo (sepal width)
- Longitud del pétalo (petal length)
- Ancho del pétalo (petal width)

## Modelo

- **Algoritmo**: Random Forest Classifier
- **Características**: Escalado estándar aplicado
- **Evaluación**: División 80/20 train/test con estratificación
- **Formato**: Modelo guardado en pickle con metadatos

## 🐳 Uso con Docker
Para el uso de docker, se realizó con una conexión por SSH
- ssh ubuntu@XXX.XXX.XXX.X
1. Clone el repositorio con SSH
- git clone git@github.com:LilaLopezg/mlops_cicdct.git
2. cd mlops_cicdct.git
Posteriormente, instalo Docker en el servidor
- sudo apt-get update
- sudo apt-get install -y ca-certificates curl gnupg
- sudo install -m 0755 -d /etc/apt/keyrings
- curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null
- sudo chmod a+r /etc/apt/keyrings/docker.asc
- echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
- sudo apt-get update  
- sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
3. Permitir Docker sin sudo
- sudo usermod -aG docker $USERexi
Cerrar sesión SSH y volver a entrar

### Prerrequisitos
- Docker instalado y ejecutándose
- Modelo entrenado (`consumo_model_smoteenn_latest.pkl` en `models/`)

### Predictor Docker

#### Scripts Automatizados
```bash
# 1. Construir imagen del predictor
./predictor/build_predictor.sh

# 2. Ejecutar predictor con ejemplos
./predictor/run_predictor.sh

# 3. Docker Compose
docker-compose up iris-predictor # Cambiar a los datos consumo de alcohol
```

#### Comandos Manuales

**Construir imagen:**
```bash
docker build -f predictor/Dockerfile -t iris-predictor:latest .
```

**Ejecutar ejemplos predefinidos:**
```bash
docker run --rm iris-predictor:latest
```

**Predicción personalizada:**
```bash
docker run --rm iris-predictor:latest python predict_api.py \
    --sepal_length 6.0 --sepal_width 3.0 --petal_length 4.5 --petal_width 1.5
```

**Con variables de entorno:**
```bash
docker run --rm \
    -e SEPAL_LENGTH=6.4 \
    -e SEPAL_WIDTH=3.2 \
    -e PETAL_LENGTH=4.5 \
    -e PETAL_WIDTH=1.5 \
    iris-predictor:latest python predict_api.py --env
```

**Salida en JSON:**
```bash
docker run --rm iris-predictor:latest python predict_api.py \
    --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2 --json
```

### Características del Predictor Docker
- **Base**: Python 3.13-slim
- **Modelo**: Pre-cargado desde `models/iris_model_latest.pkl`
- **API mejorada**: Múltiples opciones de entrada y salida
- **Variables de entorno**: Configuración flexible
- **Ejemplos integrados**: Demonstración automática de las 3 especies
- **Salida JSON**: Para integración con APIs

## Archivos del Modelo

El modelo se guarda en dos archivos:
- `models/iris_model_latest.pkl`: Versión más reciente
- `models/iris_model_YYYYMMDD_HHMMSS.pkl`: Versión con timestamp

Cada archivo pickle contiene:
- Modelo entrenado
- Scaler para normalización
- Nombres de características
- Nombres de clases objetivo
- Métricas de evaluación
- Metadatos del entrenamiento

## Requisitos del Sistema

- Python 3.8+
- scikit-learn >= 1.3.0
- pandas >= 1.5.0
- numpy >= 1.24.0