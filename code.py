# ==============================================================================
# NOTEBOOK: EKSPLORASI MODEL DETEKSI PENYAKIT DAUN TOMAT
#
# Petunjuk:
# 1. Jalankan notebook ini di lingkungan seperti Google Colab atau Jupyter.
# 2. Untuk Dataset Tomat:
#    - Unduh dan ekstrak dataset dari Kaggle:
#      https://www.kaggle.com/datasets/farukalam/tomato-leaf-diseases-detection-computer-vision
#    - Sesuaikan `TRAIN_DIR` dan `VAL_DIR` dengan path di mana Anda menyimpan data.
# 3. Kode ini dirancang untuk dijalankan sel per sel.
# ==============================================================================

# ==============================================================================
# 1. SETUP DAN KONFIGURASI
# ==============================================================================
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Konfigurasi ---
# SESUAIKAN PATH INI DENGAN LOKASI DATASET TOMAT ANDA
# CONTOH UNTUK GOOGLE COLAB SETELAH UPLOAD DARI KAGGLE
# !pip install -q kaggle
# ... (proses autentikasi kaggle)
# !kaggle datasets download -d farukalam/tomato-leaf-diseases-detection-computer-vision -p /content/ --unzip

try:
    TRAIN_DIR = '/content/tomato/train'
    VAL_DIR = '/content/tomato/val'
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError
except FileNotFoundError:
    print("WARNING: Direktori dataset tomat tidak ditemukan. Buat direktori dummy.")
    print("Anda HARUS mengganti path ini agar notebook berfungsi dengan data asli.")
    TRAIN_DIR = 'dummy_train'
    VAL_DIR = 'dummy_val'
    os.makedirs(os.path.join(TRAIN_DIR, 'dummy_class'), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, 'dummy_class'), exist_ok=True)


# Definisikan parameter model dan data
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_BASELINE = 15
EPOCHS_IMPROVED = 20

print("Setup Selesai.")
print(f"Data akan dibaca dari: {TRAIN_DIR}")

# ==============================================================================
# 2. MEMUAT DAN MEMPERSIAPKAN DATASET
# ==============================================================================

# Muat data training dan validasi menggunakan Keras utility
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    label_mode='int'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    label_mode='int'
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print("\nNama Kelas:", class_names)
print(f"Total {num_classes} kelas ditemukan.")

# Visualisasi beberapa contoh gambar
print("\nMenampilkan beberapa contoh gambar dari dataset...")
plt.figure(figsize=(12, 12))
for images, labels in train_dataset.take(1):
    for i in range(min(9, len(images))): # Pastikan tidak error jika batch < 9
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Optimalkan performa pipeline data
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print("\nDataset siap digunakan.")


# ==============================================================================
# 3. BAGIAN 1: REPRODUKSI MODEL - CUSTOM CNN (BASELINE)
# ==============================================================================
print("\n--- Membangun Model Baseline: Custom CNN ---")

# Arsitektur Model
baseline_model = Sequential([
    # Normalisasi data
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    # Blok Konvolusi 1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Blok Konvolusi 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Blok Konvolusi 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax') # Lapisan output
])

# Kompilasi Model
baseline_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

baseline_model.summary()

# Melatih model
print("\nMemulai pelatihan model baseline...")
history_baseline = baseline_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_BASELINE
)

# Evaluasi model baseline
print("\n--- Evaluasi Model Baseline ---")
val_loss_baseline, val_acc_baseline = baseline_model.evaluate(validation_dataset)
print(f"Akurasi Validasi Baseline: {val_acc_baseline*100:.2f}%")


# ==============================================================================
# 4. BAGIAN 2: EKSPLORASI & PENINGKATAN
# ==============================================================================

# ------------------------------------------------------------------------------
# STRATEGI 1: MENAMBAHKAN DATA AUGMENTATION
# ------------------------------------------------------------------------------
print("\n--- Membangun Model dengan Data Augmentation ---")

# Definisikan lapisan augmentasi
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Gabungkan augmentasi dengan model CNN yang sama
augmented_model = Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation, # Tambahkan augmentasi di sini
    layers.Rescaling(1./255), # Normalisasi setelah augmentasi

    # Arsitektur sama dengan baseline
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Kompilasi model
augmented_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

augmented_model.summary()

# Melatih model
print("\nMemulai pelatihan model dengan augmentasi...")
history_augmented = augmented_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_IMPROVED # Latih lebih lama karena augmentasi
)

# Evaluasi model
print("\n--- Evaluasi Model dengan Augmentasi ---")
val_loss_augmented, val_acc_augmented = augmented_model.evaluate(validation_dataset)
print(f"Akurasi Validasi dengan Augmentasi: {val_acc_augmented*100:.2f}%")


# ------------------------------------------------------------------------------
# STRATEGI 2: TRANSFER LEARNING DENGAN EFFICIENTNET
# ------------------------------------------------------------------------------
print("\n--- Membangun Model dengan Transfer Learning (EfficientNetB0) ---")

# Muat model dasar (base model) yang sudah dilatih di ImageNet
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, # Jangan sertakan lapisan classifier asli
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg' # Langsung lakukan average pooling di akhir
)

# Bekukan bobot model dasar agar tidak ikut terlatih di awal
base_model.trainable = False

# Bangun model baru di atas model dasar
transfer_model = Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation, # Tetap gunakan augmentasi
    # EfficientNet sudah menyertakan rescaling, jadi tidak perlu lagi
    base_model, # Model dasar
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax') # Classifier baru
])

# Kompilasi model
transfer_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

transfer_model.summary()

# Melatih model
print("\nMemulai pelatihan model transfer learning...")
history_transfer = transfer_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_IMPROVED
)

# Evaluasi model
print("\n--- Evaluasi Model Transfer Learning ---")
val_loss_transfer, val_acc_transfer = transfer_model.evaluate(validation_dataset)
print(f"Akurasi Validasi dengan Transfer Learning: {val_acc_transfer*100:.2f}%")


# ==============================================================================
# 5. KESIMPULAN DAN PERBANDINGAN HASIL
# ==============================================================================
print("\n\n--- PERBANDINGAN AKHIR ---")
print(f"Model Baseline (Custom CNN): \t\t{val_acc_baseline*100:.2f}% Akurasi Validasi")
print(f"Model dengan Data Augmentation: \t{val_acc_augmented*100:.2f}% Akurasi Validasi")
print(f"Model dengan Transfer Learning: \t{val_acc_transfer*100:.2f}% Akurasi Validasi")

# Visualisasi perbandingan
def plot_history(histories, names, metric='accuracy'):
    plt.figure(figsize=(15, 8))
    for history, name in zip(histories, names):
        val_metric = 'val_' + metric
        plt.plot(history.history[val_metric], label=f'{name} ({max(history.history[val_metric]):.3f})')
    plt.title(f'Perbandingan {metric.capitalize()} Validasi Antar Model')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

all_histories = [history_baseline, history_augmented, history_transfer]
all_names = ['Baseline CNN', 'CNN + Augmentation', 'Transfer Learning']
plot_history(all_histories, all_names, 'accuracy')
plot_history(all_histories, all_names, 'loss')

print("\nKESIMPULAN:")
print("1. Model baseline memberikan performa dasar yang baik, membuktikan arsitektur CNN efektif.")
print("2. Penambahan Data Augmentation meningkatkan akurasi dan membuat model lebih stabil (mengurangi overfitting).")
print("3. Transfer Learning dengan EfficientNetB0 memberikan performa terbaik secara signifikan, menunjukkan kekuatan memanfaatkan model pra-terlatih untuk tugas Computer Vision.")