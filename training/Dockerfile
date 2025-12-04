FROM python:3.13-slim

LABEL maintainer="MLOps Team"
LABEL description="Entrenamiento modelo consumo de alcohol"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p models

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir imbalanced-learn matplotlib seaborn
   

# Copiar script de entrenamiento
COPY train_noparem.py .

# No copiar ningún modelo (se genera nuevo)
# COPY models/consumo_model_latest.pkl models/consumo_model_latest.pkl  <-- ELIMINADO

EXPOSE 8080

CMD ["python", "train_noparem.py"]

