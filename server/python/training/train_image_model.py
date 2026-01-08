"""
Image Deepfake Detection Model Training Script
Uses ResNet50 architecture with transfer learning
"""

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "../models/image_detector.h5"

# Data paths (update these to your dataset locations)
TRAIN_DATA_PATH = "../data/faceforensics/train"
VAL_DATA_PATH = "../data/faceforensics/val"
TEST_DATA_PATH = "../data/faceforensics/test"


def create_model():
    """
    Create ResNet50-based deepfake detection model
    """
    # Load pre-trained ResNet50 (without top layers)
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model


def create_data_generators():
    """
    Create data generators with augmentation
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    # Validation/Test data (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DATA_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_generator = val_datagen.flow_from_directory(
        TEST_DATA_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def train_model():
    """
    Main training function
    """
    print("=" * 50)
    print("Image Deepfake Detection Model Training")
    print("=" * 50)

    # Create model
    print("\n[1/5] Creating model...")
    model, base_model = create_model()

    # Compile model
    print("[2/5] Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    print(f"\nModel Summary:")
    model.summary()

    # Create data generators
    print("\n[3/5] Loading data...")
    train_gen, val_gen, test_gen = create_data_generators()

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")

    # Callbacks
    print("\n[4/5] Setting up callbacks...")
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
    ]

    # Train model (Phase 1: Frozen base)
    print("\n[5/5] Training model (Phase 1: Frozen base)...")
    history_phase1 = model.fit(
        train_gen, epochs=20, validation_data=val_gen, callbacks=callbacks, verbose=1
    )

    # Fine-tuning (Phase 2: Unfreeze base)
    print("\n[5/5] Training model (Phase 2: Fine-tuning)...")
    base_model.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    history_phase2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=20,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    print("=" * 50)
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(
        f"  F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}"
    )

    # Save training history
    history_path = (
        f'../models/training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy'
    )
    np.save(
        history_path,
        {"phase1": history_phase1.history, "phase2": history_phase2.history},
    )

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Training history saved to: {history_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU")

    # Train model
    train_model()
