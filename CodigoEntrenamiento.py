import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()  

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# 1. CONFIGURACIÓN DE RUTAS
# ============================================================
base_path = r"G:\My Drive\BaseDatosMix"
categorias = ["holes", "stains", "no_defects"]

# ============================================================
# 2. CARGA EFICIENTE DE DATOS (Evita duplicar arreglos en RAM)
# ============================================================
def cargar_dataset(base_path, categorias, target_size=(224, 224)):
    X = []
    y = []
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for idx, cat in enumerate(categorias):
        path = os.path.join(base_path, cat)
        if not os.path.exists(path):
            print(f"ADVERTENCIA: La ruta no existe -> {path}")
            continue
            
        archivos = sorted([f for f in os.listdir(path) if f.lower().endswith(extensiones)])
        print(f"Cargando {len(archivos)} imágenes de la categoría: {cat}")
        
        for archivo in archivos:
            ruta_img = os.path.join(path, archivo)
            img = cv2.imread(ruta_img)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(idx)
                

    return np.array(X, dtype="float32"), np.array(y, dtype="int")

print("============================================================")
print("CARGANDO DATASET TEXTIL...")
print("============================================================")

X, y = cargar_dataset(base_path, categorias)

# División
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

y_train_cat = keras.utils.to_categorical(y_train, num_classes=3)
y_val_cat = keras.utils.to_categorical(y_val, num_classes=3)

# ============================================================
# 3. DATA AUGMENTATION
# ============================================================
datagen = ImageDataGenerator(
    rotation_range=15,       
    zoom_range=0.15,
    brightness_range=[0.8, 1.2], 
    horizontal_flip=True,
    vertical_flip=True       
)

# ============================================================
# 4. ARQUITECTURA (Transfer Learning con MobileNetV2)
# ============================================================
# Cargamos MobileNetV2 sin el clasificador final, usando pesos de ImageNet
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 

modelo = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    base_model,
    
    layers.GlobalAveragePooling2D(),
    
    # Cabeza clasificadora robusta
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(), 
    layers.Dropout(0.4),         
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(3, activation='softmax')
])

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

# ============================================================
# 5. CONFIGURACIÓN DE CALLBACKS Y PESOS
# ============================================================
pesos_clases = {0: 3.0, 1: 1.0, 2: 1.0} 

parada_temprana = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduccion_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# 6. ENTRENAMIENTO
# ============================================================
nepochs = 50 
tam_lote = 32

print("\nIniciando entrenamiento con Transfer Learning...")
historial = modelo.fit(
    datagen.flow(X_train, y_train_cat, batch_size=tam_lote),
    epochs=nepochs,
    validation_data=(X_val, y_val_cat),
    class_weight=pesos_clases,
    callbacks=[parada_temprana, reduccion_lr]
)

# ============================================================
# 7. GUARDAR EL MODELO EXPORTADO CORRECO
# ============================================================
modelo.save('clasificador_telas_IA.keras')
print("\nModelo guardado exitosamente como 'clasificador_telas_v2.keras'")