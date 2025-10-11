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

### 2. Instalar dependencias necesarias
```bash
# Expo CLI (si no lo tienes)
npm install -g expo-cli

# Texto a voz
npx expo install expo-speech

# Cámara
npx expo install expo-camera
npm install react-native-image-picker

# Manipulación de imágenes
npx expo install expo-image-manipulator

# Guardar/leer archivos
npx expo install expo-file-system

# Reproducir audio
npx expo install expo-av

# Dibujos y gráficos
npx expo install react-native-svg

# Peticiones HTTP
npm install axios
```
### 3. Ejecutar la app
```bash
npx expo start
```

Prueba en celular con Expo Go o en emulador.

Expo maneja automáticamente permisos de cámara y red.


### Opción 2 — Usar React Native CLI (nativo puro)

Si ya tienes instalado Android Studio y SDKs, y quieres compilar un APK nativo completo (sin Expo), entonces sí usas este flujo:

```bash
npx react-native init DetectorPlacas
cd DetectorPlacas
```

Luego ejecutas:
```bash
npx react-native run-android
```

**Este método:**

Usa Gradle y Android Studio internamente.

Te da acceso al código nativo (Java/Kotlin).

Requiere tener configurado correctamente el SDK de Android, ADB y el emulador o dispositivo físico.


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




