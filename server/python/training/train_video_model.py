"""
Video Deepfake Detection Model Training Script
Uses 3D CNN + LSTM architecture for temporal analysis
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv3D, MaxPooling3D, LSTM, Dense, Dropout, 
    TimeDistributed, Flatten, BatchNormalization, Attention
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import os
from datetime import datetime

# Configuration
FRAME_COUNT = 16  # Number of frames per video
IMG_SIZE = 224
BATCH_SIZE = 8  # Smaller batch size for video
EPOCHS = 40
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = '../models/video_detector.h5'

# Data paths
TRAIN_DATA_PATH = '../data/dfdc/train'
VAL_DATA_PATH = '../data/dfdc/val'
TEST_DATA_PATH = '../data/dfdc/test'


def create_3d_cnn_lstm_model():
    """
    Create 3D CNN + LSTM model for video deepfake detection
    """
    model = Sequential([
        # 3D Convolutional layers
        Conv3D(32, (3, 3, 3), activation='relu', 
               input_shape=(FRAME_COUNT, IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        
        Conv3D(64, (3, 3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        
        Conv3D(128, (3, 3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        
        Conv3D(256, (3, 3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling3D((2, 2, 2)),
        
        # Reshape for LSTM
        TimeDistributed(Flatten()),
        
        # LSTM layers for temporal modeling
        LSTM(256, return_sequences=True),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        
        # Classification head
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    return model


def load_video_frames(video_path, num_frames=FRAME_COUNT):
    """
    Load and preprocess video frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize and normalize
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)
    
    cap.release()
    
    # Pad if necessary
    while len(frames) < num_frames:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    
    return np.array(frames)


class VideoDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for video data
    """
    def __init__(self, data_path, batch_size=BATCH_SIZE, shuffle=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load video paths and labels
        self.video_paths = []
        self.labels = []
        
        for class_name in ['real', 'fake']:
            class_path = os.path.join(data_path, class_name)
            label = 0 if class_name == 'real' else 1
            
            if os.path.exists(class_path):
                for video_file in os.listdir(class_path):
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        self.video_paths.append(os.path.join(class_path, video_file))
                        self.labels.append(label)
        
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.video_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_videos = []
        batch_labels = []
        
        for i in batch_indices:
            video_frames = load_video_frames(self.video_paths[i])
            batch_videos.append(video_frames)
            batch_labels.append(self.labels[i])
        
        return np.array(batch_videos), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def train_model():
    """
    Main training function
    """
    print("=" * 50)
    print("Video Deepfake Detection Model Training")
    print("=" * 50)
    
    # Create model
    print("\n[1/4] Creating model...")
    model = create_3d_cnn_lstm_model()
    
    # Compile model
    print("[2/4] Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print(f"\nModel Summary:")
    model.summary()
    
    # Create data generators
    print("\n[3/4] Loading data...")
    train_gen = VideoDataGenerator(TRAIN_DATA_PATH, shuffle=True)
    val_gen = VideoDataGenerator(VAL_DATA_PATH, shuffle=False)
    test_gen = VideoDataGenerator(TEST_DATA_PATH, shuffle=False)
    
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    print(f"Test batches: {len(test_gen)}")
    
    # Callbacks
    print("\n[4/4] Setting up callbacks...")
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[4/4] Training model...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
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
    print(f"  F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
    
    # Save training history
    history_path = f'../models/video_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npy'
    np.save(history_path, history.history)
    
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    print(f"Training history saved to: {history_path}")
    print("\nTraining complete!")


if __name__ == '__main__':
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Enable GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(e)
    
    # Train model
    train_model()
