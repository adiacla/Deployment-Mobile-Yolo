# Detección y Reconocimiento de Placas Vehiculares con YOLOv8 + FastAPI
## Objetivo

Este proyecto implementa un sistema de detección automática de placas de vehículos y reconocimiento de caracteres (OCR) utilizando un modelo YOLOv8 entrenado mediante transfer learning y un servicio FastAPI para exponer un endpoint de inferencia.

## Tecnologías usadas

**Python 3.13**

**FastAPI** (framework backend)

**Ultralytics** YOLOv8 (detección de objetos)

**EasyOCR** (reconocimiento de texto)

**OpenCV (cv2)** (procesamiento de imágenes)

**Torch** / torchvision

**vUvicorn** (servidor ASGI)

**SQLite** (opcional) para guardar resultados con timestamp

---
## 1. **Backend - FastAPI en AWS EC2**

Si tienes acceso a Learner LAb, incia el Learner Lab
![alt text](https://raw.githubusercontent.com/adiacla/Deployment-Mobile-Yolo/refs/heads/main/imagenes/learnerlab.JPG))

### 1.1 **Configurar la Instancia EC2 en AWS**

1. En la consola de administración de AWS seleccione el servicio de EC2 (servidor virtual) o escriba en buscar.
![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraEC2.JPG?raw=true)

2. Ve a la opción para lanzar la instancia

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irainstancias.JPG?raw=true)

3. Lanza una istancia nueva

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iralanzarinstancia.JPG?raw=true)

4. Inicia una nueva **instancia EC2** en AWS (elige Ubuntu como sistema operativo), puede dejar la imagen por defecto. 

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Instancia%20Ubuntu.PNG?raw=true)

5. Para este proyecto dado que el tamaño del modelo a descargar es grande necesitamos una maquina con más memoria y disco.
   con nuesra licencia tenemos permiso desde un micro lanzar hasta un T2.Large. 


![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iratipodeinstancia.JPG?raw=true)


6. seleccione el par de claves ya creado, o cree uno nuevo (Uno de los dos, pero recuerde guardar esa llave que la puede necesitar, no la pierda)

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraparclaves.JPG?raw=true)

7. Habilite los puertos de shh, web y https, para este proyecto no lo vamos a usar no es necesario, pero si vas a publicar una web es requerido.
   ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irfirewall.JPG?raw=true)

8. Configure el almacenamiento. Este proyecto como se dijo requere capacidad en disco. Aumente el disco minimo a **32** GiB.

   ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraconfiguraralmacenamiento.JPG?raw=true)

9. Finalmente lance la instancia (no debe presentar error, si tiene error debe iniciar de nuevo). Si todo sale bien, por favor haga click en instancias en la parte superior.

   ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/lanzarinstanciafinal.PNG?raw=true)


10. Dado que normalmente en la lista de instancias NO VE la nueva instancia lanzada por favor actualice la pagina Web o en ir a instancias
    
 ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iracutualizarweb.JPG?raw=true)
![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irainstancias.JPG?raw=true)

11. Vamos a seleccionar el servidor ec2 lanzado.
    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irseleccionarinstancia.JPG?raw=true)

12. Verificar la dirección IP pública y el DNS en el resumen de la instancia
    
![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irresumeninstancia.JPG?raw=true)

13. Debido a que vamos a lanzar un API rest debemos habilitar el puerto. Vamos al seguridad

    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraseguirdad.JPG?raw=true)

14. Vamos al grupo de seguridad

   ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iragruposeguridad.JPG?raw=true)

   15. Vamos a ir a Editar la regla de entrada

       ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iraregladeentrada.JPG?raw=true)

16. Ahora vamos a agregar un regla de entrada para habilitar el puerto, recuerden poner IPV 4

    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iragregarregla.JPG?raw=true)

     


17. Abre un puerto en el grupo de seguridad (por ejemplo, puerto **8080** o si requiere el **8720** así está en alguos ejemplos) para permitir acceso a la API.

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Puerto.PNG?raw=true)

18. Guardemos la regla de entrada.
    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irguardarreglas.JPG?raw=true)

19. Ve nuevamente a instancias
    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/iralanzarinstanciaB.JPG?raw=true)

20. Vamos a conectar con la consola del servidor
    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irconectar.JPG?raw=true)

    ![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/irconsola.JPG?raw=true)
    
3. Si no puedes conectarse directamente a la instancia EC2, conectate  con SSH, es decir en la consola de administración de instancia creada hay una opcion de "Conectar", has clic y luego conectar otra vez. Si no puede conectarse puede hacerlo con el SSH:
   

   ```bash
   ssh -i "tu_clave.pem" ubuntu@<tu_ip_ec2>

---

### 1.2 Instalar Dependencias en el Servidor EC2
Una vez dentro de tu instancia EC2, instalar las librerias y complementos como FastAPI y las dependencias necesarias para ello debes crear una carpeta en donde realizaras las instalaciones:

**Ver las carpetas**
 ```bash
ls -la
 ```
**Ver la version de python**
 ```bash
python3 -V
 ```

**Si se requiere, puede actualizar los paquetes**
 ```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0

 ```
![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/aptUpdate.PNG?raw=true)


**Si se requiere: Instalar pip y virtualenv**
 ```bash
sudo apt install python3-pip python3-venv
 ```

**Crear la carpeta del proyecto**
 ```bash
mkdir proyecto
 ```

**Accede a tu carpeta**
 ```bash
cd proyecto
 ```

**Crear y activar un entorno virtual**
 ```bash

python3 -m venv venv
source venv/bin/activate
 ```
Recuerda que en el prompt debe obersar que el env debe quedar activo

**Instalar FastAPI, Uvicorn, Joblib, TensorFlow, Python-Multipart, Pillow**
 ```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install ultralytics fastapi uvicorn easyocr opencv-python-headless pillow numpy python-multipart

 ```
```bash
yolo check
 ```

### Subir el archivo del modelo
**Subir archivos con scp**
El formato general del comando es:
```bash
scp -i "llavewebici.pem" <archivo_local> ubuntu@<DNS_PUBLICO>:/home/ubuntu/<carpeta_destino>
```
Por ejemplo, si quieres subir:

best.pt

app.py

ejecuta en tu sesion cmd de tu pc:
```
scp -i "llavewebici.pem" best.pt app.py ubuntu@ec2-98-81-166-76.compute-1.amazonaws.com:/home/ubuntu/proyecto/
```
Esto copia ambos archivos al directorio /home/ubuntu/proyecto/ dentro de tu instancia EC2.

### 1.3 Crear la API FastAPI

Crea un archivo app.py en tu instancia EC2 para definir la API que servirá las predicciones.

 ```bash
nano app.py
 ```

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/nanoApp.PNG?raw=true)


## Desarrollo del Backend API
Usaremos FastAPI por su rendimiento y facilidad de uso. El backend aceptará una imagen, la procesará con el modelo Yolo8n con el modelo best.pt y devolverá la predicción.
Puede copiar este codigo en tu editor de nano.


## API: Detector de Placas Vehiculares con YOLOv8 y OCR (FastAPI)

Este servicio expone un API REST basado en FastAPI que combina la detección de objetos con 
el reconocimiento óptico de caracteres (OCR) para identificar placas vehiculares en imágenes.

**Flujo general:**
1. El usuario envía una imagen (JPG o PNG) mediante un `POST /predict/`.
2. El modelo YOLOv8 detecta los objetos en la imagen (por ejemplo, vehículos y placas).
3. Si se identifica una placa, se extrae el recorte y se procesa con EasyOCR.
4. El servicio devuelve:
   - El texto leído de la placa (`placa`),
   - Una lista con las detecciones (etiqueta, confianza, coordenadas, texto detectado),
   - La imagen procesada codificada en Base64 (opcional).

**Endpoints principales:**
- `GET /` → Verifica que el servidor esté activo.
- `POST /predict/` → Realiza la detección y OCR sobre una imagen enviada.


```python
#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 + EasyOCR para detección de placas
# Requiere: fastapi uvicorn ultralytics easyocr opencv-python-headless pillow numpy python-multipart

import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import easyocr
import base64

# -------------------------
# Config / Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-plates")

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")  # ruta al best.pt
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")  # ej: "en" o "en,es"
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))  # umbral de confianza para detecciones
RETURN_IMAGE = os.getenv("RETURN_IMAGE", "1") != "0"  # devolver imagen en base64 por defecto

# -------------------------
# App init
# -------------------------
app = FastAPI(title="YOLOv8 - Detector de Placas (OCR)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cambiar en producción por tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Cargar modelo y OCR
# -------------------------
logger.info("Cargando modelo YOLOv8 desde %s ...", MODEL_PATH)
model = YOLO(MODEL_PATH)  # carga pesos
logger.info("Modelo cargado: %s", getattr(model, "model", "YOLO model"))

logger.info("Inicializando EasyOCR con idiomas: %s", OCR_LANGS)
reader = easyocr.Reader(OCR_LANGS, gpu=False)

# -------------------------
# Helpers
# -------------------------
def ocr_read_text_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    """
    Preprocesa el ROI y ejecuta EasyOCR. Devuelve la cadena concatenada (sin espacios)
    o None si no se detecta texto.
    """
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        # Convert BGR -> RGB para EasyOCR
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        # Escala y limpieza básica
        h, w = roi_rgb.shape[:2]
        scale = 1
        if max(h, w) < 200:
            scale = int(200 / max(h, w))
            roi_rgb = cv2.resize(roi_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Convertir a gris + equalize si ayuda
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.equalizeHist(gray)

        # Adaptive threshold (opcional, EasyOCR suele funcionar bien sin binarizar)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY, 11, 2)
        # usar la imagen RGB o gray: EasyOCR acepta numpy array RGB
        # preferimos pasar la imagen RGB para mejores resultados en multi-color
        result = reader.readtext(roi_rgb)
        if not result:
            return None
        # Elegir el texto con mayor confianza
        best = max(result, key=lambda x: x[2])
        text = best[1]
        if not text:
            return None
        # Normalizar: eliminar espacios extra y caracteres no alfanum
        text = "".join(ch for ch in text if ch.isalnum())
        return text.upper() if text else None
    except Exception as e:
        logger.exception("OCR error: %s", e)
        return None

def image_to_base64_jpg(img_bgr: np.ndarray) -> str:
    """Codifica imagen BGR a base64 JPG string"""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode('utf-8')

# -------------------------
# Routes
# -------------------------
@app.get("/")
def home():
    return {"message": "YOLOv8 + OCR server running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen (form-data, campo 'file') y devuelve detecciones y OCR.
    Respuesta JSON:
    {
      "placa": "ABC123" | null,
      "detections": [
         {"label":"car","confidence":0.92,"box":[x1,y1,x2,y2],"text": null}
      ],
      "image": "<base64 jpg>"  # opcional
    }
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}

        # Inference YOLOv8 -> usar predict() para mayor compatibilidad
        # conf=CONF_THRESH controla el umbral
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)

        if not results:
            return {"error": "No result from model"}

        r = results[0]  # primer resultado
        # obtener boxes si existen
        try:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
            boxes_conf = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
            boxes_cls = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        except Exception:
            # fallback si r.boxes no tiene estructura esperada
            boxes_xyxy = np.array([])
            boxes_conf = np.array([])
            boxes_cls = np.array([])

        detections: List[Dict[str, Any]] = []
        placa_detectada: Optional[str] = None

        if boxes_xyxy.size > 0:
            for i, box in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = map(int, box)
                conf = float(boxes_conf[i]) if boxes_conf.size > 0 else None
                cls_id = int(boxes_cls[i]) if boxes_cls.size > 0 else None
                label = model.names[cls_id] if cls_id is not None and cls_id < len(model.names) else str(cls_id)

                # recortar ROI con chequeo de límites
                h, w = frame.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                roi = frame[y1c:y2c, x1c:x2c].copy()

                text_detected = None
                # Ejecutar OCR solo si la etiqueta sugiere placa/license/plate
                if any(k in label.lower() for k in ["placa", "plate", "license"]):
                    text_detected = ocr_read_text_from_roi(roi)
                    if text_detected:
                        placa_detectada = text_detected  # si hay múltiples se sobrescribe; puedes cambiar lógica

                # Dibujar en la imagen final
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)
                label_text = f"{label} {conf:.2f}" if conf is not None else label
                cv2.putText(frame, label_text, (x1c, max(15, y1c-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if text_detected:
                    cv2.putText(frame, text_detected, (x1c, y2c + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                detections.append({
                    "label": label,
                    "confidence": round(conf, 4) if conf is not None else None,
                    "box": [int(x1c), int(y1c), int(x2c), int(y2c)],
                    "text": text_detected
                })

        # convertir imagen a base64 si lo deseamos
        img_b64 = None
        if RETURN_IMAGE:
            img_b64 = image_to_base64_jpg(frame)

        resp = {
            "placa": placa_detectada,
            "detections": detections,
            "image": img_b64
        }
        return resp

    except Exception as e:
        logger.exception("Error en /predict/: %s", e)
        return {"error": str(e)}

# -------------------------
# Iniciar servidor (si se ejecuta app.py directamente)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("Arrancando uvicorn en 0.0.0.0:%s", port)
    # Nota: en producción recomendamos ejecutar: uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

```
---
### docs#
La URL http://3.80.229.31:8080/docs (reemplazando la IP por la tuya) es una interfaz automática de documentación interactiva que FastAPI genera por defecto.

Qué puedes hacer en /docs:

Explorar todos los endpoints disponibles

Por ejemplo:

GET / → Prueba que el servidor está corriendo.

POST /predict/ → Permite subir una imagen y ver la respuesta.

**Ver los parámetros esperados y sus tipos**
FastAPI usa type hints de Python para documentar los parámetros (por ejemplo file: UploadFile = File(...)).

**Subir archivos directamente desde el navegador**
En POST /predict/, verás un campo para seleccionar una imagen y probar el modelo sin usar Postman.

**Observar la respuesta estructurada**
Swagger muestra automáticamente la respuesta JSON del servidor (por ejemplo, la placa detectada y la imagen codificada).

**Generar pruebas rápidas o debugging**
Si algo no funciona, /docs te ayuda a verificar si el backend está recibiendo los archivos correctamente.

### 1.5 Ejecutar el Servidor FastAPI

Para ejecutar el servidor de FastAPI, usa Uvicorn:

 ```bash
source venv/bin/activate
python3 app.py 
 ```

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/ServidorAws.PNG?raw=true)

### 1.6 Error en el Servidor

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Error.PNG?raw=true)

Si al momento de ejecutar el servidor te da un error como en el de la anterior imagen en el cual se excede la memoria del sistema utiliza el siguiente comando y vuelve a intentarlo

```bash
sudo sync; sudo sysctl -w vm.drop_caches=3
 ```

## Pueba del Backend
Puedes usar la prueba manual
Descargue esta imagen de prueba a su pc
![](https://raw.githubusercontent.com/adiacla/Deployment-Mobile-Yolo/refs/heads/main/imagenes/carroprueba.JPG)

**Prueba manual:**

Usa herramientas como Postman o cURL para probar la API antes de integrarla con el frontend. Ejemplo de prueba con cURL:

curl -X POST -F "file=@image.jpg" http://ec2-54-164-41-174.compute-1.amazonaws.com:8080/predict/
Espera un JSON como respuesta con las predicciones.

Si vas a utilizar postman entra en el siguiente enlance https://www.postman.com , crea o ingresa a tu cuenta y sigue los siguientes pasos:
1. Dale click en new request

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/NewRequest.PNG?raw=true)
   
2. Poner las siguientes opciones en la request

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/PostRequest.PNG?raw=true)
   
Recuerda que debes poner la URL de tu EC2 acompañado con el :8080 que es el puerto y con el /predict que es el endpoint que queremos probar.

![alt text](https://raw.githubusercontent.com/adiacla/Deployment-Mobile-Yolo/refs/heads/main/imagenes/postmanprueba.JPG)

La API estará disponible en http://<tu_ip_ec2>:8080.

---

# Guía rápida de instalación — React Native en Windows 11

## Requisitos previos

1. **Node.js y npm**  
   Verifica versiones (deben ser ≥ Node 18, npm ≥ 9):
   ```bash
   node -v
   npm -v
   ```
   > Si no los tienes, instala la versión **LTS** desde [https://nodejs.org](https://nodejs.org)

2. **Java Development Kit (JDK)**  
   Instala **OpenJDK 17** o superior:  
   [https://adoptium.net](https://adoptium.net)

3. **Android Studio**  
   Descarga desde  [https://developer.android.com/studio](https://developer.android.com/studio)

   Luego, abre **SDK Manager** y verifica que estén instaladas:
   - ✅ Android SDK Platform **35**
   - ✅ Android SDK Build-Tools **35.0.0**
   - ✅ Android Emulator
   - ✅ Android SDK Command-line Tools (latest)
   - ✅ NDK (Side by side)
   - ✅ CMake
---

## Configurar variables de entorno

Abre “Editar variables de entorno del sistema” → “Variables de usuario”.

Agrega o verifica las siguientes rutas:

| Variable | Valor sugerido |
|-----------|----------------|
| **ANDROID_HOME** | `%LOCALAPPDATA%\Android\Sdk` |
| **Path** | `%ANDROID_HOME%\platform-tools`|
|**Path** |  `%ANDROID_HOME%\emulator`|
|**Path**  | `%ANDROID_HOME%\cmdline-tools\latest\bin` |

---

## Crear un emulador (AVD)

1. Abre **Android Studio → More Actions → Virtual Device Manager**
2. Crea un dispositivo tipo **Pixel 6a / API 33 o superior**
3. Inicia el emulador **antes** de ejecutar la app.

> También puedes conectar tu teléfono Android con la depuración USB activada.

---

# Activar modo desarrollador a tu telefono
Recomendaciones específicas para tu caso

## Activa el modo desarrollador

Sigue los pasos:

Ajustes → Acerca del teléfono → Información de software → Toca 7 veces Número de compilación.

Verás el mensaje “Ahora eres desarrollador”.

## Activa la depuración USB

Ajustes → Opciones de desarrollador → activa Depuración USB.

Conecta el teléfono por cable USB

Usa un cable de datos (no solo de carga).

Cuando aparezca el mensaje “¿Permitir depuración USB?” → pulsa Permitir siempre y Aceptar.

## Verifica la conexión
En la consola (CMD o terminal):
```bash
adb devices
```

Si ves algo como:
```bash
List of devices attached
R58N123ABC	device
````
Todo está correcto.

Si dice “unauthorized”, toca Permitir depuración USB en tu celular.
---

##  Limpieza de instalaciones previas (solo si tuviste errores antes)

```bash
npm uninstall -g react-native-cli
npm cache clean --force
```

---


# Lector de Placas - Instalación y Ejecución

## Requisitos

No necesitas crear nada manualmente en Android Studio. Solo asegúrate de tener:

- **Android Studio** (para los SDKs y emuladores).  
- **Dispositivo físico** con Depuración USB activada.

---

## Crear el proyecto en React Native / Expo

Si vas a crear una app móvil con React Native / Expo (como el siguiente `App.tsx`), **NO necesitas crear un proyecto nuevo en Android Studio desde cero**.  
React Native se encarga de generar todo lo necesario (Gradle, manifest, APK, etc.).

**Crear proyecto con Expo (recomendado):**

```bash
npx create-expo-app lector-placas --template expo-template-blank-typescript
cd lector-placas
```

Reemplaza App.tsx con el código del lector de placas.

**Estructura del proyecto:**
```lua
lector-placas/
├── App.tsx                👈 Aquí va tu código
├── app.json
├── babel.config.js
├── package.json
├── tsconfig.json
├── node_modules/
└── assets/
    ├── icon.png
    ├── splash.png
    └── ...
```


**1. Crear proyecto**

Cree una carpeta de proyecto
luego
```bash
npx create-expo-app my-camera-app 
cd DetectorPlacas
```

Ajustar package.json para SDK 50:
Abre package.json y modifica las versiones de expo y react-native a las correspondientes al SDK 50 (ej. ~50.0.0 y 0.73.6).

```json
{
  "name": "placas",
  "main": "index.tsx", 
  "version": "1.0.0",
  "scripts": {
    "start": "expo start",
    "reset-project": "node ./scripts/reset-project.js",
    "android": "expo start --android",
    "ios": "expo start --ios",
    "web": "expo start --web",
    "lint": "expo lint"
  },
  "dependencies": {
    "@expo/vector-icons": "^15.0.2",
    "@react-navigation/bottom-tabs": "^7.4.0",
    "@react-navigation/elements": "^2.6.3",
    "@react-navigation/native": "^7.1.8",
    "expo": "~50.0.0",
    "expo-camera": "~14.0.5", 
    "expo-constants": "~18.0.9",
    "expo-font": "~14.0.9",
    "expo-haptics": "~15.0.7",
    "expo-image": "~3.0.9",
    "expo-image-manipulator": "~12.0.5", 
    "expo-linking": "~8.0.8",
    "expo-router": "~6.0.11",
    "expo-splash-screen": "~31.0.10",
    "expo-speech": "~12.0.2", 
    "expo-status-bar": "~3.0.8",
    "expo-symbols": "~1.0.7",
    "expo-system-ui": "~6.0.7",
    "expo-web-browser": "~15.0.8",
    "react": "18.2.0",             
    "react-dom": "18.2.0",         
    "react-native": "0.73.6",
    "react-native-gesture-handler": "~2.28.0",
    "react-native-worklets": "0.5.1",
    "react-native-reanimated": "~4.1.1",
    "react-native-safe-area-context": "~5.6.0",
    "react-native-screens": "~4.16.0",
    "react-native-web": "~0.21.0",
    "axios": "^1.7.2"    
  },          
  "devDependencies": {
    "@babel/core": "^7.20.0",      
    "@types/react": "~18.2.45",    
    "typescript": "~5.1.3",        
    "eslint": "^9.25.0",
    "eslint-config-expo": "~10.0.0"
  },
  "private": true
}

```

**2. Instalar dependencias**
Limpiar e instalar dependencias base (para SDK 50):

``bash
rmdir /s /q node_modules
del package-lock.json
npm install
``
``bash
npx expo install expo-camera expo-image-manipulator expo-speech
npm install axios
```

Copia tu código index.tsx:
Si tu archivo principal actual se llama App.tsx, renómbralo a index.tsx.
Borra el contenido de index.tsx.
Pega el código completo de tu index.tsx que me proporcionaste al inicio de la conversación.
Guarda el archivo.



```tsx
import axios from "axios";
import { Camera, CameraType } from 'expo-camera'; // <-- Agregado CameraType para mayor claridad
import * as ImageManipulator from "expo-image-manipulator";
import * as Speech from "expo-speech";
import React, { useEffect, useRef, useState } from "react";
import { ActivityIndicator, Alert, Button, Image, StyleSheet, Text, TextInput, View, Platform } from "react-native"; // <-- Agregado Platform

export default function App() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [imagen, setImagen] = useState<string | null>(null);
  const [placa, setPlaca] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  // Es buena práctica usar un estado para la IP por defecto, pero permitir que el usuario la cambie.
  // Podrías inicializarla desde una variable de entorno o configuración.
  const [ip, setIp] = useState("18.209.15.47"); 
  const [port, setPort] = useState("8080");
  const cameraRef = useRef<Camera | null>(null);

  useEffect(() => {
    (async () => {
      // Solicitar permiso de cámara
      const { status: cameraStatus } = await Camera.requestCameraPermissionsAsync();
      // Solicitar permiso de micrófono (aunque no lo usas directamente para la foto, Expo lo pide)
      const { status: microphoneStatus } = await Camera.requestMicrophonePermissionsAsync();
      
      // Considerar que ambos deben estar garantizados si ambos son críticos,
      // o manejar individualmente si uno es opcional.
      setHasPermission(cameraStatus === "granted" && microphoneStatus === "granted");
    })();
  }, []);

  // Manejo inicial de permisos antes de renderizar la cámara
  if (hasPermission === null) {
    return (
      <View style={styles.center}>
        <Text style={{ color: "#fff" }}>Cargando permisos de cámara...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.center}>
        <Text style={{ color: "#fff" }}>Necesitas permitir el acceso a la cámara y al micrófono para usar esta aplicación.</Text>
      </View>
    );
  }

  const tomarFoto = async () => {
    if (!cameraReady) { // Asegurarse de que la cámara esté lista antes de intentar tomar la foto
      Alert.alert("Error", "La cámara no está lista aún.");
      return;
    }
    if (cameraRef.current) {
      try {
        setLoading(true); // Mostrar indicador de carga mientras se toma y manipula la foto
        const photo = await cameraRef.current.takePictureAsync({ base64: false });
        // Comprobar si la URI de la foto es válida antes de manipularla
        if (!photo.uri) {
          Alert.alert("Error", "No se pudo obtener la URI de la foto.");
          return;
        }
        
        const manipResult = await ImageManipulator.manipulateAsync(
          photo.uri,
          [{ resize: { width: 800 } }],
          { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG }
        );
        
        if (!manipResult.uri) {
            Alert.alert("Error", "No se pudo manipular la imagen.");
            return;
        }
        setImagen(manipResult.uri);
      } catch (error) {
        console.error("Error al tomar o manipular la foto:", error);
        Alert.alert("Error", "No se pudo tomar o procesar la foto.");
      } finally {
        setLoading(false); // Ocultar indicador de carga
      }
    }
  };

  const enviarImagen = async () => {
    if (!imagen) return Alert.alert("Error", "Toma primero una foto.");
    // Usar la IP por defecto si el usuario no la ha cambiado
    const serverIp = ip || "18.209.15.47";
    const serverPort = port || "8080"; // Usar el puerto por defecto si el usuario no lo ha cambiado

    setLoading(true);
    setPlaca(null); // Limpiar resultados anteriores al enviar una nueva imagen
    setConfidence(null);

    try {
      const formData = new FormData();
      formData.append("file", {
        uri: Platform.OS === 'android' ? imagen : imagen.replace("file://", ""), // Corrección para iOS que a veces necesita remover "file://"
        type: "image/jpeg",
        name: "placa.jpg",
      } as any);
      
      const url = `http://${serverIp}:${serverPort}/predict/`; // Usar serverIp y serverPort

      const response = await axios.post(url, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 10000, // Añadir un timeout para la petición (ej. 10 segundos)
      });

      const data = response.data;
      if (data.placa && data.detections?.length > 0) {
        const detection = data.detections[0];
        setPlaca(data.placa.toUpperCase());
        setConfidence(detection.confidence);
        Speech.speak(`Placa detectada: ${data.placa}`);
      } else {
        Alert.alert("Detección", "No se detectó ninguna placa en la imagen.");
      }
    } catch (error) {
      console.error("Error al enviar imagen:", error);
      // Manejo más específico de errores de red vs. errores del servidor
      if (axios.isAxiosError(error)) {
        if (error.response) {
          // El servidor respondió con un status diferente de 2xx
          Alert.alert("Error del servidor", `Código: ${error.response.status}, Mensaje: ${error.response.data?.message || 'Error desconocido'}`);
        } else if (error.request) {
          // La petición fue hecha pero no se recibió respuesta (ej. red caída, servidor no responde)
          Alert.alert("Error de conexión", `No se pudo conectar con el servidor en http://${serverIp}:${serverPort}. Verifica la IP, puerto y conexión a internet.`);
        } else {
          // Algo más ocurrió al configurar la petición
          Alert.alert("Error", "Algo salió mal al preparar la petición.");
        }
      } else {
        Alert.alert("Error", `Ocurrió un error inesperado: ${error instanceof Error ? error.message : String(error)}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {/* Vista de la cámara */}
      {hasPermission && ( // Solo renderizar la cámara si tenemos permiso
        <Camera
          style={styles.camera}
          type={CameraType.back} // Uso explícito de CameraType
          ref={cameraRef}
          onCameraReady={() => setCameraReady(true)}
          // Puedes añadir un fallback si la cámara no se inicializa
          onMountError={(error) => console.error("Error al montar la cámara:", error)}
        />
      )}

      <View style={styles.overlay}>
        <Text style={styles.label}>IP del servidor:</Text>
        <TextInput
          style={styles.input}
          placeholder="Ej: 18.209.15.47"
          placeholderTextColor="#aaa"
          value={ip}
          onChangeText={setIp}
          autoCapitalize="none" // Para IPs, no capitalizar
          autoCorrect={false}   // Para IPs, no corregir automáticamente
        />
        <Text style={styles.label}>Puerto:</Text>
        <TextInput
          style={styles.input}
          placeholder="8080"
          placeholderTextColor="#aaa"
          keyboardType="numeric"
          value={port}
          onChangeText={setPort}
          maxLength={5} // Un puerto no suele tener más de 5 dígitos
        />

        <View style={styles.buttonContainer}>
          <Button title="Tomar Foto" onPress={tomarFoto} disabled={!cameraReady || loading} />
          {/* Un poco de espacio entre botones */}
          <View style={{ width: 10 }} /> 
          <Button title="Enviar Imagen" onPress={enviarImagen} disabled={!imagen || loading} />
        </View>

        {loading && (
          <>
            <ActivityIndicator size="large" color="#00ffcc" style={{ marginTop: 10 }} />
            <Text style={{ color: "#fff", textAlign: "center", marginTop: 5 }}>Procesando...</Text>
          </>
        )}

        {placa && confidence && (
          <Text style={styles.text}>
            Placa: {placa} {"\n"}
            Confianza: {(confidence * 100).toFixed(2)}%
          </Text>
        )}

        {imagen && (
          <Image 
            source={{ uri: imagen }} 
            style={styles.preview} 
            accessibilityLabel="Imagen tomada por la cámara" // Accesibilidad
          />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  camera: { flex: 1 },
  overlay: {
    position: "absolute",
    bottom: 10,
    width: "90%",
    alignSelf: "center",
    backgroundColor: "rgba(0,0,0,0.8)", // Un poco más opaco para mejor contraste
    padding: 15,
    borderRadius: 10,
  },
  label: { color: "#fff", marginTop: 5, fontSize: 16 },
  input: {
    backgroundColor: "#222",
    color: "#fff",
    paddingHorizontal: 10,
    paddingVertical: 8, // Ligeramente más padding vertical
    borderRadius: 8,
    marginBottom: 10,
    fontSize: 16, // Para mejor legibilidad
  },
  buttonContainer: {
    flexDirection: 'row', // Botones en fila
    justifyContent: 'space-around', // Espacio entre ellos
    marginTop: 10,
    marginBottom: 10, // Espacio antes del indicador de carga
  },
  text: { color: "#fff", fontSize: 20, marginTop: 15, textAlign: "center", fontWeight: "bold" }, // Más grande y en negrita
  preview: { width: "100%", height: 200, marginTop: 10, borderRadius: 8, borderWidth: 1, borderColor: '#00ffcc' }, // Borde para la imagen
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#000" }, // Fondo oscuro también en pantalla de carga
});

```

**ejecutar con npx expo start -c**

Asegúrate de haber guardado todos los cambios en package.json y index.tsx.
Abre tu terminal en el directorio raíz de tu proyecto placas.
Ejecuta el comando:

npx expo start -c

Verás un mensaje que indica que se está limpiando el caché antes de iniciar el packager.
Luego, como de costumbre, aparecerá el código QR.
Escanea el código QR con la aplicación Expo Go en tu teléfono.


# Opción 2 — Usar React Native CLI (nativo puro)

Si quieres generar un APK/IPA nativo y tener acceso completo al código nativo.


**3. Configurar permisos Android**

Editar android/app/src/main/AndroidManifest.xml y agregar:
```
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.INTERNET"/>
```
**4. Código de App.tsx**
```tsx
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  Button,
  StyleSheet,
  Image,
  Alert,
  Platform,
  PermissionsAndroid,
  TextInput,
  ActivityIndicator,
} from 'react-native';
import { launchCamera } from 'react-native-image-picker';
import Tts from 'react-native-tts';
import axios from 'axios';

const App = () => {
  const [imagen, setImagen] = useState<string | null>(null);
  const [placa, setPlaca] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  // Campos para IP y puerto
  const [ip, setIp] = useState('');
  const [port, setPort] = useState('8080');

  // Pedir permisos
  const requestPermissions = async () => {
    if (Platform.OS === 'android') {
      const granted = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.CAMERA,
        PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE,
        PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
      ]);
      const allGranted = Object.values(granted).every(
        status => status === PermissionsAndroid.RESULTS.GRANTED
      );
      if (!allGranted) {
        Alert.alert('Permisos denegados', 'La app necesita permisos para funcionar.');
      }
    }
  };

  useEffect(() => {
    requestPermissions();
    Tts.setDefaultLanguage('es-ES');
    Tts.setDefaultRate(0.5);
  }, []);

  const tomarFoto = () => {
    launchCamera({ mediaType: 'photo', cameraType: 'back', quality: 0.5 }, response => {
      if (response.didCancel) return;
      if (response.errorCode) return console.log(response.errorMessage);
      const uri = response.assets?.[0].uri;
      setImagen(uri || null);
    });
  };

  const enviarImagen = async () => {
    if (!imagen) {
      Alert.alert('Error', 'Toma primero una foto.');
      return;
    }
    if (!ip) {
      Alert.alert('Error', 'Ingresa la dirección IP del servidor.');
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('upload', {
        uri: imagen,
        type: 'image/jpeg',
        name: 'placa.jpg',
      } as any);

      const url = `http://${ip}:${port}/plate-reader/`; // API local

      const response = await axios.post(url, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const result = response.data.results?.[0];
      if (result) {
        setPlaca(result.plate?.toUpperCase() || null);
        setConfidence(result.confidence || null);
        Tts.speak(`Placa detectada: ${result.plate}`);
      } else {
        Alert.alert('No se detectó ninguna placa.');
      }
    } catch (error: any) {
      console.error(error);
      Alert.alert('Error', `No se pudo conectar con ${ip}:${port}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>📷 Lector de Placas</Text>

      <Text style={styles.label}>IP del servidor:</Text>
      <TextInput
        style={styles.input}
        placeholder="Ej: 192.168.1.100"
        placeholderTextColor="#999"
        value={ip}
        onChangeText={setIp}
      />

      <Text style={styles.label}>Puerto:</Text>
      <TextInput
        style={styles.input}
        placeholder="8080"
        placeholderTextColor="#999"
        keyboardType="numeric"
        value={port}
        onChangeText={setPort}
      />

      <Button title="Tomar Foto" onPress={tomarFoto} />
      <View style={{ marginTop: 10 }} />
      <Button title="Enviar Imagen" onPress={enviarImagen} />

      {loading && (
        <ActivityIndicator size="large" color="#007bff" style={{ marginTop: 15 }} />
      )}

      {imagen && <Image source={{ uri: imagen }} style={styles.image} />}

      {placa && confidence && (
        <Text style={styles.text}>
          Placa: {placa} {'\n'}
          Confianza: {(confidence * 100).toFixed(2)}%
        </Text>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f9f9f9',
  },
  title: { fontSize: 24, marginBottom: 20, fontWeight: 'bold', color: '#333' },
  label: { alignSelf: 'flex-start', color: '#333', fontSize: 16, marginTop: 10 },
  input: {
    width: '100%',
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 16,
    backgroundColor: '#fff',
  },
  image: {
    width: 300,
    height: 200,
    marginTop: 20,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ccc',
  },
  text: { marginTop: 10, fontSize: 18, textAlign: 'center', color: '#333' },
});

export default App;

```

5. Ejecutar en Android
# Inicia Metro bundler
npx react-native start

# En otra terminal, ejecuta la app
npx react-native run-android


La app se instalará en tu emulador o dispositivo físico.

Solicitará permisos automáticamente en Android.

6. Notas iOS (opcional)

Agrega en ios/DetectorPlacas/Info.plist:

<key>NSCameraUsageDescription</key>
<string>Necesitamos acceder a la cámara para capturar placas</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Necesitamos acceder a la galería para guardar fotos</string>
<key>NSMicrophoneUsageDescription</key>
<string>Necesitamos usar el micrófono para TTS</string>

✅ Ventajas de cada opción
Opción	Ventajas
Expo	Rápido, sencillo, permisos manejados automáticamente, pruebas inmediatas en celular.
React Native CLI	APK/IPA nativo listo para producción, acceso completo al código nativo, más control sobre permisos y librerías nativa

---

## Solución de errores comunes

| Error | Solución |
|-------|-----------|
| `SDK location not found` | Revisa la variable `ANDROID_HOME`. |
| `JAVA_HOME not set` | Configura la ruta del JDK (`C:\Program Files\Eclipse Adoptium\jdk-17\`). |
| `Emulator not found` | Abre Android Studio y corre el AVD manualmente. |
| `Build failed` | Ejecuta `cd android && gradlew clean` y vuelve a intentar. |

---

## Recomendaciones

- Usa **VS Code** como editor principal.  
- No instales `react-native-cli` globalmente.  
- Usa siempre `npx react-native ...` para evitar conflictos.  
- Mantén Android Studio y las SDK Tools actualizadas.  




