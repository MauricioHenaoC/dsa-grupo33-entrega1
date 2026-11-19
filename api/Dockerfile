FROM python:3.13-slim

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY model/predict.py model/predict.py
COPY model/modelo_final.joblib model/modelo_final.joblib
COPY api/main.py api/main.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]