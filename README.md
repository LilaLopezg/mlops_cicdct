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
#Inicio en un servidor SSH
ssh ubuntu@XXX.XXX.XX.XX
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
python training/train_advanced.py   --data /home/ubuntu/mlops_cicdct/data/train.csv   --out /home/ubuntu/mlops_cicdct/training/models   --n_jobs 4   --optuna_trials 20 #Entrenamiento optimizados de varios modelos.
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
./build_and_runm.sh

echo "🐳 Construyendo imagen Docker para entrenamiento de Consumo de Alcohol..."
docker build -t alcohol_training . # Se construye la imagen
docker build --no-cache -t alcohol_training . #Sin cache

echo "🚀 Ejecutando contenedor..."
docker run --rm \
  -v "/home/ubuntu/mlops_cicdct/training/data:/app/data" \
  -v "/home/ubuntu/mlops_cicdct/training/models:/app/models" \
  alcohol_training
```

Este script:
- Construye la imagen Docker con Python 3.13
- Ejecuta el entrenamiento automáticamente
- Guarda el modelo en `./models/` (montado como volumen)

### 3. Hacer Predicciones 🔮

#### Opción A: Python Local
```bash
cd training
python predict.py \
  --age 21 \
  --gender 1 \
  --parent_alcohol 0 \
  --academic_semester 4 \
  --model_path /home/ubuntu/mlops_cicdct/training/models/consumo_model_latest.pkl

```

#### Opción B: Docker Predictor (Recomendado) 🐳

**1. Construir imagen del predictor:**
```bash
docker build -t alcohol-predictor .
./predictor/build_predictor.sh
docker build -f predictor/Dockerfile -t consumo-predictor:latest predictor/

```

**2. Ejecutar ejemplos de predicción:**
Para ejecutar los ejemplos con ./predictor/run_predictor.sh, hay que agregarle a Dockerfile COPY predictor/
```bash
docker run --rm alcohol-predictor
./predictor/run_predictor.sh
docker run --rm consumo-predictor:latest
```

**3. Predicción personalizada:**
```bash
docker run --rm alcohol-predictor \
    python predict_api.py --age 22 --gender 1 --parent_alcohol 1 --academic_semester 4
```

**4. Con variables de entorno:**
```bash
docker run --rm \
    -e AGE=23 \
    -e GENDER=0 \
    -e PARENT_ALCOHOL=1 \
    -e ACADEMIC_SEMESTER=6 \
    alcohol-predictor \
    python predict_api.py --env
```

**5. Con Docker Compose:**
Instalación docker-compose
- sudo apt update
- sudo apt-get install docker-compose-plugin  ** Se instalo la version 2 de docker compose**
- docker compose version
- docker compose up --build 
```bash
# Ejecutar ejemplos
docker compose up consumo-predictor

# Predicción personalizada (editar variables en docker-compose.yml)
docker compose --profile custom up consumo-predictor-custom
```

### 4. Ejemplos de Predicción

**Consumo de alcohol (edad):**
```bash
python predict.py --age 22 --gender 1 --parent_alcohol 1 --academic_semester 4
```

**Consumo de alcohol (sexo):**
```bash
python predict.py --age 19 --gender 0 --parent_alcohol 0 --academic_semester 6
```
**Consumo de alcohol (familiares borrachos):**
```bash
python predict.py --age 24 --gender 1 --parent_alcohol 1 --academic_semester 8
```

## Dataset Consumo de alcohol de estudiantes de la Universidad de Córdoba - 2022-II

El dataset consumo de alcohol en estudiantes universitarios contiene 2971 muestras con 72 variables:
- **Edad**: 2971 muestras
- **Sexo**: 2971 muestras
- **Familiares_borracho**: 2971 muestras 
- **Semestre_academico**: 2971 muestras

Cada muestra tiene 2971 características:
- Edad estudientes de los programas academicos.
- Sexo biologico de los estudientes
- Familiares que se  borrachos e influyen en los estudientes 
- Semestre academicos de los estudiantes universitarios, entre otras variables.

## Modelo

- **Algoritmo**: Random Forest Classifier, 
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
- Modelo entrenado (`consumo_model_latest.pkl` en `models/`)

### Predictor Docker

#### Scripts Automatizados
```bash
# 1. Construir imagen del predictor
./predictor/build_predictor.sh

# 2. Ejecutar predictor con ejemplos
./predictor/run_predictor.sh

# 3. Docker Compose
docker-compose up consumo-predictor # Cambiar a los datos consumo de alcohol
``

#### Comandos Manuales

**Construir imagen:**
```bash
docker build -f predictor/Dockerfile -t consumo-predictor:latest .
```

**Ejecutar ejemplos predefinidos:**
```bash
docker run --rm consumo-predictor:latest
```

**Predicción personalizada:**
```bash
docker run --rm alcohol-predictor \
    python predict_api.py --age 22 --gender 1 --parent_alcohol 1 --academic_semester 4
```

**Con variables de entorno:**
```bash
docker run --rm \
    -e AGE=23 \
    -e GENDER=0 \
    -e PARENT_ALCOHOL=1 \
    -e ACADEMIC_SEMESTER=6 \
    alcohol-predictor \
    python predict_api.py --env
```

**Salida en JSON:**
```bash
docker run --rm alcohol-predictor:latest python predict_api.py \
    --age 22 --gender 1 --parent_alcohol 1 --academic_semester 4 --json
```

### Características del Predictor Docker
- **Base**: Python 3.13-slim
- **Modelo**: Pre-cargado desde `models/consumo_model_latest.pkl`
- **API mejorada**: Múltiples opciones de entrada y salida
- **Variables de entorno**: Configuración flexible
- **Ejemplos integrados**: Demonstración automática de las muestras del consumo de alcohol
- **Salida JSON**: Para integración con APIs

## Archivos del Modelo

El modelo se guarda en dos archivos:
- `models/consumo_model_latest.pkl`: Versión más reciente
- `models/best_consumo_model_20251121_170743.pkl`: Versión más reciente
- `models/consumo_model_YYYYMMDD_HHMMSS.pkl`: Versión con timestamp

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
