FROM python:3.13-slim

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt "dash>=2.18,<4"

COPY data/raw/DESERCION_FPI_20261023.csv data/raw/DESERCION_FPI_20261023.csv
COPY model/modelo_final.joblib model/modelo_final.joblib
COPY app/ app/

ENV HOST=0.0.0.0 PORT=8050
EXPOSE 8050
CMD ["python", "app/app.py"]