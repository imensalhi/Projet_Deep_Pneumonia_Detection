# Importations nécessaires
import os  # Manipulation des chemins de fichiers
import numpy as np  # Manipulations mathématiques et tableaux
import matplotlib.pyplot as plt  # Visualisations graphiques
from sklearn.metrics import ConfusionMatrixDisplay  # Matrice de confusion
from tensorflow.keras.models import Sequential  # Construction de modèles
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)  # Couches du modèle
from tensorflow.keras.optimizers import Adam  # Optimiseur
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # Callbacks pour entraînement
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Génération d'images
from tensorflow.keras.applications import DenseNet121  # Modèle pré-entraîné

# Configuration des paramètres
input_shape = (224, 224, 3)  # Ajuster selon vos données
batch_size = 32
num_classes = 2  # Normal/Pneumonia
learning_rate = 1e-4
epochs = 30

# Génération de données avec augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
)

val_data = val_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/val',  # Remplacez par le chemin réel
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
)

# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

# Modèle avec DenseNet121 (pré-entraînement)
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.trainable = False  # Gèle les couches du modèle de base

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.02)),
    Dropout(0.6),
    Dense(num_classes, activation="softmax"),
])

# Compilation du modèle
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Entraînement
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping],
)

# Déverrouiller certaines couches du modèle pré-entraîné pour fine-tuning
base_model.trainable = True
fine_tune_at = len(base_model.layers) // 2

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompiler pour fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Fine-tuning
history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[reduce_lr, early_stopping],
)

# Évaluation des modèles sur les données de test
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/test',  # Remplacez par le chemin réel
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Évaluation du modèle DenseNet121
test_loss_dense, test_accuracy_dense = model.evaluate(test_data)
print(f"DenseNet121 - Test Accuracy: {test_accuracy_dense * 100:.2f}%")
print(f"DenseNet121 - Test Loss: {test_loss_dense:.4f}")

# Sauvegarde du modèle
model.save('pneumonia_classifier_DenseNet.h5')
