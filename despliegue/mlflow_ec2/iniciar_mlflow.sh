#!/usr/bin/env bash
# Arranca el servidor de seguimiento de MLflow en la EC2.
# Lo ejecuta el servicio systemd (mlflow.service); no hace falta correrlo a mano.
set -euo pipefail

MLFLOW_HOME="${MLFLOW_HOME:-/home/ubuntu/mlflow}"
PUERTO="${MLFLOW_PORT:-5000}"

# IP pública de la instancia, leída del servicio de metadatos de EC2 (IMDSv2).
# Se lee en cada arranque, así que si la instancia no tiene IP elástica
# y la IP cambia al reiniciar, el servidor se ajusta solo.
TOKEN=$(curl -sf -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300" || true)
IP_PUBLICA=$(curl -sf -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  "http://169.254.169.254/latest/meta-data/public-ipv4" || true)

if [[ -z "${IP_PUBLICA}" ]]; then
  echo "No se pudo leer la IP pública. ¿La instancia tiene IP pública asignada?" >&2
  exit 1
fi

# Desde MLflow 3.5, el servidor rechaza las peticiones cuyo encabezado Host
# no esté en esta lista ("Invalid Host header"). Por defecto solo acepta
# localhost e IPs privadas, así que hay que agregar la IP pública.
# Afecta tanto al navegador como a los scripts que registran corridas.
export MLFLOW_SERVER_ALLOWED_HOSTS="${IP_PUBLICA}:${PUERTO},${IP_PUBLICA},localhost:*,127.0.0.1:*"

echo "MLflow disponible en http://${IP_PUBLICA}:${PUERTO}"

# --artifacts-destination: los artefactos (modelos, figuras) se guardan en el
# disco de la EC2 y el servidor los recibe por HTTP; los clientes no necesitan
# acceso directo a ese disco ni credenciales de AWS.
exec "${MLFLOW_HOME}/venv/bin/mlflow" server \
  --host 0.0.0.0 \
  --port "${PUERTO}" \
  --backend-store-uri "sqlite:///${MLFLOW_HOME}/mlflow.db" \
  --artifacts-destination "${MLFLOW_HOME}/artifacts" \
  --workers 2
