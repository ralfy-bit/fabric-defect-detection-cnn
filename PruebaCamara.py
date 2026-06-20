import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import PySpin

# ============================================================
# 1. CONFIGURACIÓN Y PARÁMETROS
# ============================================================
MODEL_PATH = r'C:\Users\juane\OneDrive\Escritorio\CNN_ClassWeights\IA\clasificador_telas_IA.keras'
TARGET_SIZE = (224, 224)
CATEGORIAS = ["HUECO (Hole)", "MANCHA (Stain)", "SIN DEFECTO (No Defect)"]
COLORES = [(0, 0, 255), (255, 0, 0), (0, 255, 0)] 

print("Cargando modelo de clasificación...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit()

# ============================================================
# 2. CONFIGURACIÓN DE LA CÁMARA FLIR (PySpin)
# ============================================================
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

if cam_list.GetSize() == 0:
    print("ERROR: No se detectó ninguna cámara FLIR conectada.")
    cam_list.Clear()
    system.ReleaseInstance()
    sys.exit()

cam = cam_list[0]
cam.Init()

nodemap = cam.GetNodeMap()

# CONFIGURACIÓN DEL BUFFER (Anti-Delay)
s_node_map = cam.GetTLStreamNodeMap()
handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferHandlingMode"))
handling_mode_newestonly = handling_mode.GetEntryByName("NewestOnly")
handling_mode.SetIntValue(handling_mode_newestonly.GetValue())

node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())

processor = PySpin.ImageProcessor()

if not cam.IsStreaming():
    cam.BeginAcquisition()
    print("Cámara FL3 inicializada correctamente.")
else:
    print("La cámara ya estaba transmitiendo.")

print("Presiona 'q' en la ventana de OpenCV para salir.")

# ============================================================
# 3. BUCLE PRINCIPAL DE INSPECCIÓN EN TIEMPO REAL
# ============================================================
try:
    while True:
        image_result = cam.GetNextImage(1000) 

        if image_result.IsIncomplete():
            image_result.Release()
            continue

        image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
        frame = image_converted.GetNDArray()
        
        image_result.Release()

        if frame is None:
            continue

        # -----------------------------------------------------------------
        # Aplicamos un desenfoque Gaussiano ultra-leve (kernel 3x3) para eliminar
        # el patrón de Bayer y el Moiré en telas claras antes de procesar o mostrar.
        # -----------------------------------------------------------------
        frame_filtrado = cv2.GaussianBlur(frame, (3, 3), 0)

        # Usamos el frame filtrado para la interfaz visual
        display_frame = frame_filtrado.copy()

        # Preprocesamiento para la Red Neuronal (Usando el frame limpio)
        img_rgb = cv2.cvtColor(frame_filtrado, cv2.COLOR_BGR2RGB)
        
        # Reducción por área (la mejor para mantener homogeneidad en texturas)
        img_resized = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        input_tensor = np.expand_dims(img_resized, axis=0)

        # Inferencia
        predicciones = model.predict(input_tensor, verbose=0)[0]
        clase_id = np.argmax(predicciones)
        confianza = predicciones[clase_id] * 100

        # Interfaz de usuario
        texto_resultado = f"{CATEGORIAS[clase_id]}: {confianza:.2f}%"
        color_alerta = COLORES[clase_id]

        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 50), color_alerta, -1)
        cv2.putText(display_frame, texto_resultado, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar en pantalla
        cv2.namedWindow('Inspección en Tiempo Real - FabricEye', cv2.WINDOW_NORMAL)
        cv2.imshow('Inspección en Tiempo Real - FabricEye', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInspección interrumpida por el usuario.")

# ============================================================
# 4. LIMPIEZA DE HARDWARE Y MEMORIA
# ============================================================
print("\nCerrando sistema de visión y liberando recursos...")
cv2.destroyAllWindows()

try:
    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    print("Cámara desvinculada correctamente.")
except Exception as e:
    print(f"Error al liberar recursos de PySpin: {e}")