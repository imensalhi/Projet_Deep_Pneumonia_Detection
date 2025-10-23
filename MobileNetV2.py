import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

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
    "/Users/mac/Downloads/chest_xray/chest_xray/train",  # Remplacez par le chemin réel
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
)

val_data = val_datagen.flow_from_directory(
    "/Users/mac/Downloads/chest_xray/chest_xray/val",  # Remplacez par le chemin réel
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
)

# Modèle avec MobileNetV2 (pré-entraînement)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.trainable = False  # Gèle les couches du modèle de base

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.02)),
    Dropout(0.6),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
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

# Évaluation finale
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    "/Users/mac/Downloads/chest_xray/chest_xray/test",  # Remplacez par le chemin réel
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Sauvegarder le modèle
model.save('pneumonia_classifier_MobileNetV2.h5')
