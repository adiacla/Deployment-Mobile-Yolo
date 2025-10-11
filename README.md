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

     


17. Abre un puerto en el grupo de seguridad (por ejemplo, puerto **8080**) para permitir acceso a la API.

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


Desarrollo del Backend API
Usaremos FastAPI por su rendimiento y facilidad de uso. El backend aceptará una imagen, la procesará con el modelo Yolo8n con el modelo best.pt y devolverá la predicción.
Puede copiar este codigo en tu editor de nano.

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import uvicorn
import os

# Inicializar la aplicación FastAPI
app = FastAPI(title="Detección de Placas con YOLOv8 + OCR")

# Cargar modelo YOLO entrenado (usa tu ruta local al best.pt)
MODEL_PATH = "best.pt"  # asegúrate de subirlo al mismo directorio
model = YOLO(MODEL_PATH)

# Inicializar el lector OCR
reader = easyocr.Reader(['en'])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Guardar la imagen temporalmente
        contents = await file.read()
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Leer imagen con OpenCV
        image = cv2.imread(temp_path)

        # Realizar detección con YOLOv8
        results = model(image)[0]

        detections = []
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r.cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Recortar la placa
            placa_img = image[y1:y2, x1:x2]

            # OCR con EasyOCR
            ocr_results = reader.readtext(placa_img)

            if ocr_results:
                texto_detectado = ocr_results[0][1].upper()
            else:
                texto_detectado = ""

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls),
                "text": texto_detectado
            })

        os.remove(temp_path)  # limpiar archivo temporal

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    # Ejecutar el servidor FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8720)

```
### 1.5 Ejecutar el Servidor FastAPI

Para ejecutar el servidor de FastAPI, usa Uvicorn:

 ```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
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

Prueba manual:

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
2. Crea un dispositivo tipo **Pixel 6 / API 33 o superior**
3. Inicia el emulador **antes** de ejecutar la app.

> También puedes conectar tu teléfono Android con la depuración USB activada.

---

##  Limpieza de instalaciones previas (solo si tuviste errores antes)

```bash
npm uninstall -g react-native-cli
npm cache clean --force
```
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

# Ejecutar tu app React Native
Desde la carpeta del proyecto:

npx react-native run-android


React Native detectará tu Samsung automáticamente y compilará la app directamente en él.


---

## Crear y ejecutar tu proyecto

1. Crear un nuevo proyecto:
   ```bash
   npx react-native init MiApp
   cd MiApp
   ```

2. Iniciar la app en Android:
   ```bash
   npx react-native run-android
   ```

3. (Opcional) Ejecutar el servidor Metro:
   ```bash
   npx react-native start
   ```

---

## 🧩 Solución de errores comunes

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




