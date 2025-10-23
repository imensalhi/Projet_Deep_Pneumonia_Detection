import os  # Pour la manipulation des chemins de fichiers
import matplotlib.pyplot as plt  # Pour les visualisations graphiques
from tensorflow.keras.preprocessing import image  # Pour charger et prétraiter les images
import numpy as np  # Pour les opérations mathématiques et manipulation de tableaux
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Pour les matrices de confusion


# 1) Fonction pour récupérer et afficher les images des classes spécifiques
def display_images_from_class(directory, class_names, num_images=2):
    fig, axes = plt.subplots(1, num_images * len(class_names), figsize=(15, 5))

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(directory, class_name)
        images = os.listdir(class_path)[:num_images]  # Prendre les 2 premières images de chaque classe

        for j, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0  # Normaliser les images

            axes[i * num_images + j].imshow(img_array)
            axes[i * num_images + j].axis('off')
            axes[i * num_images + j].set_title(f"{class_name} ")

    plt.suptitle("Exemples d'images avant l'augmentation")
    plt.show()


# Appeler la fonction pour afficher les images
display_images_from_class(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/train',  # Chemin du dataset
    ['NORMAL', 'PNEUMONIA'],  # Les classes à afficher
    num_images=2  # Nombre d'images par classe
)

# Exemple d'images augmentées
# Supposons que train_data soit un générateur d'images déjà défini
sample_data, _ = next(train_data)  # Extraire un batch d'images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    axes[i].imshow(sample_data[i])
    axes[i].axis('off')
plt.suptitle("Exemples d'images augmentées")
plt.show()


# 2) Visualisation de la répartition des données
def plot_class_distribution(directory, class_names, title):
    counts = [len(os.listdir(os.path.join(directory, class_name))) for class_name in class_names]
    plt.figure(figsize=(6, 4))
    plt.bar(class_names, counts, color=['blue', 'orange'])
    plt.xlabel('Classes')
    plt.ylabel("Nombre d'images")
    plt.title(title)
    plt.show()


# Répartition dans l'ensemble d'entraînement
plot_class_distribution(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/train',
    ['NORMAL', 'PNEUMONIA'],
    "Distribution des images dans l'ensemble d'entraînement"
)

# Répartition dans l'ensemble de validation
plot_class_distribution(
    '/kaggle/input/chest-xray-pneumonia/chest_xray/val',
    ['NORMAL', 'PNEUMONIA'],
    "Distribution des images dans l'ensemble de validation"
)

# 3) Visualisation des courbes (exemple pour MobileNetV2)
plt.figure(figsize=(14, 10))

# Courbe de précision
plt.subplot(3, 2, 3)
plt.plot(history_mobilenet_fine.history['accuracy'], label='MobileNetV2 - Train')
plt.plot(history_mobilenet_fine.history['val_accuracy'], label='MobileNetV2 - Validation')
plt.title('MobileNetV2 - Courbes de Précision')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()

# Courbe de perte
plt.subplot(3, 2, 4)
plt.plot(history_mobilenet_fine.history['loss'], label='MobileNetV2 - Train')
plt.plot(history_mobilenet_fine.history['val_loss'], label='MobileNetV2 - Validation')
plt.title('MobileNetV2 - Courbes de Perte')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()

plt.show()

# 4) Matrices de confusion (exemple pour MobileNetV2)
y_pred_mobilenet = np.argmax(model_mobilenet.predict(test_data), axis=1)
y_true_mobilenet = test_data.classes

cm_mobilenet = confusion_matrix(y_true_mobilenet, y_pred_mobilenet)
ConfusionMatrixDisplay(cm_mobilenet, display_labels=test_data.class_indices.keys()).plot()
plt.title('MobileNetV2 - Matrice de Confusion')
plt.show()
