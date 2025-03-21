import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Path dataset
dataset_path = "dataset_hands"

# Preprocessing data
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # 20% data sebagai validasi
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),  # Ukuran gambar yang diproses
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    # Output sesuai jumlah orang
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training model
model.fit(train_generator, validation_data=val_generator, epochs=20)

# Simpan model
model.save("hand_model.h5")

print("Model CNN tangan telah selesai dilatih dan disimpan!")
