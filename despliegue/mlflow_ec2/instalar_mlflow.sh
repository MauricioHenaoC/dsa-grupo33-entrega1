#!/usr/bin/env bash
# Instala MLflow en una EC2 con Ubuntu 24.04 y lo deja corriendo como servicio.
#
# Uso, en la EC2, dentro de la carpeta mlflow_ec2:
#   bash instalar_mlflow.sh                        # instala la versión más reciente
#   MLFLOW_VERSION=x.y.z bash instalar_mlflow.sh   # instala una versión concreta
#
# El servidor debe tener la misma versión que el .venv del equipo, o una más
# nueva. Un cliente más nuevo que el servidor puede fallar al registrar.
# Para pasar a una versión MÁS VIEJA, borra antes ~/mlflow/mlflow.db (solo si
# aún no tiene corridas): una versión vieja no abre una base creada por otra nueva.
set -euo pipefail

MLFLOW_VERSION="${MLFLOW_VERSION:-}"
MLFLOW_HOME=/home/ubuntu/mlflow
DIR_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 1/4 Paquetes del sistema"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip curl

echo "== 2/4 Swap de 2 GB si la máquina tiene menos de 2 GB de RAM (t2.micro, t3.micro)"
MEM_MB=$(free -m | awk '/^Mem:/ {print $2}')
if [[ "${MEM_MB}" -lt 1900 ]] && ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "== 3/4 Entorno de Python con MLflow ${MLFLOW_VERSION:-(versión más reciente)}"
mkdir -p "${MLFLOW_HOME}/artifacts"
python3 -m venv "${MLFLOW_HOME}/venv"
"${MLFLOW_HOME}/venv/bin/pip" install --upgrade pip
if [[ -n "${MLFLOW_VERSION}" ]]; then
  "${MLFLOW_HOME}/venv/bin/pip" install "mlflow==${MLFLOW_VERSION}"
else
  "${MLFLOW_HOME}/venv/bin/pip" install --upgrade mlflow
fi
VERSION_INSTALADA=$("${MLFLOW_HOME}/venv/bin/python" -c "import mlflow; print(mlflow.__version__)")

echo "== 4/4 Servicio systemd"
# Quita finales de línea de Windows (CRLF) por si los archivos pasaron por Windows.
sed 's/\r$//' "${DIR_SCRIPT}/iniciar_mlflow.sh" > "${MLFLOW_HOME}/iniciar_mlflow.sh"
chmod 755 "${MLFLOW_HOME}/iniciar_mlflow.sh"
sed 's/\r$//' "${DIR_SCRIPT}/mlflow.service" | sudo tee /etc/systemd/system/mlflow.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl restart mlflow

sleep 8
sudo systemctl --no-pager --lines=5 status mlflow || true
echo
echo "MLflow instalado en el servidor: versión ${VERSION_INSTALADA}"
echo "Anota este número: el .venv del equipo debe tener esta versión o una más vieja."
echo
echo "Revisa la línea 'MLflow disponible en http://...' en el registro:"
echo "  sudo journalctl -u mlflow -n 30 --no-pager"
