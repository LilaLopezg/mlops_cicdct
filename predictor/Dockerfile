FROM python:3.13-slim


WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p models

COPY predictor/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY predictor/predict.py .
COPY predictor/predict_api.py .

COPY predictor/models/ ./models

# Variables correctas para tu predictor
ENV AGE=18
ENV GENDER=0
ENV PARENT_ALCOHOL=0
ENV ACADEMIC_SEMESTER=1

EXPOSE 8080

CMD ["python", "predict_api.py", "--examples"]

