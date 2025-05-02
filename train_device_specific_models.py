import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight

# Path to dataset
dataset_path = "C:/Users/Vishal/OneDrive/Documents/PythonProj/ImageDetection/dataset/train"

for device_name in os.listdir(dataset_path):
    device_path = os.path.join(dataset_path, device_name)
    if not os.path.isdir(device_path):
        continue

    print(f"\n📦 Training model for device: {device_name}")

    # Image augmentation + rescaling
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    try:
        # Train generator
        train_gen = datagen.flow_from_directory(
            device_path,
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary',
            subset='training',
            shuffle=True
        )

        # Validation generator
        val_gen = datagen.flow_from_directory(
            device_path,
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )

        # Compute class weights to handle imbalance
        class_labels = train_gen.classes
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_labels),
            y=class_labels
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"⚖️  Class Weights: {class_weight_dict}")

        # Model structure
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            class_weight=class_weight_dict
        )

        # Save model
        model_filename = f"{device_name}_model.h5"
        model.save(model_filename)
        print(f"✅ Saved model: {model_filename}")

    except Exception as e:
        print(f"❌ Skipping {device_name} due to error: {str(e)}")

    # Clear session for memory cleanup
    K.clear_session()
