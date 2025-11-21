# Usar Python 3.13 como imagen base
FROM python:3.13-slim

# Establecer información del mantenedor
LABEL maintainer="MLOps Team"
LABEL description="IConsumo de alcohol Prediction API"

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar paquetes Python
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Crear el directorio models
RUN mkdir -p models

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar los scripts de predicción
COPY predict.py .
COPY predict_api.py .

# El contexto del build debe ser desde la raíz del proyecto para acceder a models/
# Por eso se debe construir desde la raíz: docker build -f predictor/Dockerfile .
COPY models/iris_model_latest.pkl models/iris_model_latest.pkl

# Establecer variables de entorno por defecto para un ejemplo
ENV SEPAL_LENGTH=5.1
ENV SEPAL_WIDTH=3.5
ENV PETAL_LENGTH=1.4
ENV PETAL_WIDTH=0.2

# Exponer puerto para futuras implementaciones de API web
EXPOSE 8080

# Establecer el comando por defecto para hacer predicciones usando el API mejorado
CMD ["python", "predict_api.py", "--examples"]