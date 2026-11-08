# Servidor de MLflow en AWS EC2

Servidor de seguimiento de MLflow del grupo 33. Aquí se registran las corridas del experimento `desercion_sena_riesgo`.

| Archivo | Qué hace |
| --- | --- |
| `instalar_mlflow.sh` | Se corre una vez en la EC2. Instala Python, MLflow y el servicio. |
| `iniciar_mlflow.sh` | Arranca el servidor. Lo ejecuta systemd. |
| `mlflow.service` | Servicio systemd: MLflow sigue corriendo al cerrar SSH y arranca solo al encender la instancia. |
| `.gitattributes` | Evita que Git en Windows convierta los scripts a CRLF. |

Las corridas quedan en el disco de la instancia: la base en `/home/ubuntu/mlflow/mlflow.db` y los artefactos en `/home/ubuntu/mlflow/artifacts/`. Si la instancia se **termina**, se pierden.

## 1. Crear la instancia

En la consola de EC2, *Launch instance*:

- AMI: **Ubuntu Server 24.04 LTS**.
- Tipo: **t3.small** (2 GB de RAM). Con t2.micro o t3.micro (1 GB) también funciona, porque el script agrega 2 GB de swap, pero la instalación es más lenta.
- Par de llaves: crear uno nuevo (`.pem`) o usar el existente.
- Red: dejar activada la asignación automática de IP pública.
- Almacenamiento: 16 GB.
- Grupo de seguridad: ver el paso 3.

## 2. IP elástica (recomendada)

*EC2 → Elastic IPs → Allocate Elastic IP address*, y luego *Actions → Associate* con la instancia. Así la IP no cambia al detener y encender.

La IP elástica tiene costo aunque la instancia esté detenida. Si no se usa, `iniciar_mlflow.sh` toma la IP nueva en cada arranque, pero hay que avisar al equipo y actualizar `MLFLOW_TRACKING_URI`.

Si se asocia la IP elástica con MLflow ya corriendo, hay que reiniciar el servicio para que tome la IP nueva:

```bash
sudo systemctl restart mlflow
```

## 3. Grupo de seguridad

Reglas de entrada, cada IP con `/32`:

| Tipo | Puerto | Origen |
| --- | --- | --- |
| SSH | 22 | IP de quien administra la EC2, IP de Pedro |
| TCP personalizado | 5000 | IP de cada persona que registra corridas o abre la interfaz de MLflow |

Nunca `0.0.0.0/0`. MLflow no tiene usuario ni contraseña, así que el grupo de seguridad es su única protección.

Cada persona obtiene su IP pública en PowerShell con:

```powershell
(Invoke-RestMethod https://checkip.amazonaws.com).Trim()
```

Las IPs de los hogares suelen cambiar. Si alguien deja de conectarse (tiempo de espera agotado), revisa su IP y actualiza la regla.

## 4. Instalar MLflow

Versión: el servidor debe tener la misma versión que el `.venv` del equipo, o una más nueva. Para ver la del `.venv`, en Windows con el `.venv` activo: `pip show mlflow`. Si no se conoce, se instala la más reciente y luego se verifica con una prueba antes de registrar las corridas (paso 5).

Desde Windows, en la raíz del repositorio. La primera vez, deja la llave solo con permiso de lectura para tu usuario (si no, `ssh` la rechaza con *UNPROTECTED PRIVATE KEY FILE*):

```powershell
icacls .\llave.pem /inheritance:r
icacls .\llave.pem /grant:r "$($env:USERNAME):(R)"

scp -i .\llave.pem -r despliegue\mlflow_ec2 ubuntu@IP_DE_LA_EC2:~
ssh -i .\llave.pem ubuntu@IP_DE_LA_EC2
```

Ya en la EC2:

```bash
cd ~/mlflow_ec2
sed -i 's/\r$//' *.sh *.service     # por si los archivos llegaron con CRLF
bash instalar_mlflow.sh                # o: MLFLOW_VERSION=x.y.z bash instalar_mlflow.sh
```

Al final, el script muestra la versión instalada. Para cambiar a una versión más vieja, borra antes `~/mlflow/mlflow.db` (solo si aún no tiene corridas) y vuelve a correr el script con `MLFLOW_VERSION`.

## 5. Verificar

En la EC2:

```bash
sudo systemctl status mlflow
sudo journalctl -u mlflow -n 30 --no-pager    # debe mostrar "MLflow disponible en http://..."
```

Desde Windows, abrir `http://IP_DE_LA_EC2:5000` en el navegador.

Antes de registrar las corridas reales, quien corre `entrenar.py` hace una prueba desde su `.venv`. Registra una corrida con un parámetro, una métrica y un archivo:

```powershell
python -c "import mlflow; mlflow.set_tracking_uri('http://IP_DE_LA_EC2:5000'); mlflow.set_experiment('prueba_conexion'); mlflow.start_run(); mlflow.log_param('p', 1); mlflow.log_metric('m', 1.0); mlflow.log_text('ok', 'prueba.txt'); mlflow.end_run(); print('Conexion OK. Cliente MLflow', mlflow.__version__)"
```

Si imprime `Conexion OK`, cliente y servidor son compatibles. Después se borra el experimento `prueba_conexion` en la interfaz, para que no salga en los pantallazos. Si falla por versión, reinstala el servidor con la versión del cliente (paso 4).

La ruta `/health` no sirve como prueba: responde aunque la IP pública no esté autorizada.

## 6. Registrar corridas

`model/entrenar.py` lee la URI de la variable de entorno (si no está definida, usa `sqlite:///mlflow.db` local). En PowerShell, con el `.venv` activo:

```powershell
$env:MLFLOW_TRACKING_URI = "http://IP_DE_LA_EC2:5000"
python model\entrenar.py
```

La variable solo dura en esa ventana de PowerShell. Si una ejecución falla a mitad de camino, borra en la interfaz las corridas incompletas antes de repetir, para no tener duplicados.

## 7. Pantallazos para la entrega

Con el servidor encendido:

1. `http://IP_DE_LA_EC2:5000` con el experimento `desercion_sena_riesgo` y la tabla de corridas. La IP debe verse en la barra de direcciones.
2. El detalle de una corrida, con parámetros y métricas.
3. La consola de AWS en la página de la instancia: IP pública visible y usuario de la sesión en la esquina superior derecha.

No uses un túnel SSH para los pantallazos: la barra de direcciones mostraría `localhost` en vez de la IP.

## 8. Detener al terminar

```bash
sudo systemctl stop mlflow
```

Con MLflow detenido, conviene sacar una copia de respaldo de las corridas. Desde Windows:

```powershell
mkdir respaldo_mlflow
scp -i .\llave.pem ubuntu@IP_DE_LA_EC2:~/mlflow/mlflow.db .\respaldo_mlflow\
scp -i .\llave.pem -r ubuntu@IP_DE_LA_EC2:~/mlflow/artifacts .\respaldo_mlflow\
```

Luego, en la consola: *Instance state → Stop instance*. **Nunca** *Terminate*, porque borra el disco con las corridas.

El servicio queda habilitado, así que al volver a encender la instancia MLflow arranca solo con las corridas guardadas. Si no se quiere que arranque al encender: `sudo systemctl disable mlflow`.

## Problemas frecuentes

| Síntoma | Causa y solución |
| --- | --- |
| `Invalid Host header - possible DNS rebinding attack detected` | La IP pública cambió o se asoció la IP elástica después de arrancar. `sudo systemctl restart mlflow`. |
| Tiempo de espera agotado en el navegador o en `entrenar.py` | La IP de quien se conecta no está en el puerto 5000 del grupo de seguridad, o cambió. |
| `UNPROTECTED PRIVATE KEY FILE` | Permisos de la llave en Windows. Ver los comandos `icacls` del paso 4. |
| `/usr/bin/env: 'bash\r'` o `$'\r': command not found` | Archivos con CRLF. `sed -i 's/\r$//' *.sh *.service`. |
| El servicio se reinicia en bucle | `sudo journalctl -u mlflow -n 50 --no-pager` para ver el error. |
