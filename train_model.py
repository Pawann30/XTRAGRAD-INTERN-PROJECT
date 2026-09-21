
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# 1. SETTINGS
# ==========================================

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 2

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

MODEL_PATH = "models/image_detector.keras"


# ==========================================
# 2. CREATE MODELS FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

# Training data with automatic validation split
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)


# Test data preprocessing
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# ==========================================
# 4. LOAD TRAINING DATA (80%)
# ==========================================

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=42
)


# ==========================================
# 5. LOAD VALIDATION DATA (20%)
# ==========================================

validation_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=42
)


# ==========================================
# 6. LOAD TEST DATA
# ==========================================

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


# ==========================================
# 7. CHECK CLASS LABELS
# ==========================================

print("\nClass Labels:")
print(train_data.class_indices)

# Expected:
# {'FAKE': 0, 'REAL': 1}


# ==========================================
# 8. CREATE CNN MODEL
# ==========================================

model = Sequential([

    # Input layer
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # First convolution layer
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Second convolution layer
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Third convolution layer
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Fourth convolution layer
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten the features
    Flatten(),

    # Dense layer
    Dense(128, activation="relu"),

    # Prevent overfitting
    Dropout(0.5),

    # Output layer
    Dense(1, activation="sigmoid")

])


# ==========================================
# 9. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 10. DISPLAY MODEL SUMMARY
# ==========================================

print("\nModel Summary:\n")
model.summary()


# ==========================================
# 11. EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ==========================================
# 12. TRAIN MODEL
# ==========================================

print("\nTraining Started...\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# ==========================================
# 13. TEST MODEL
# ==========================================

print("\nTesting Model...\n")

test_loss, test_accuracy = model.evaluate(
    test_data
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ==========================================
# 14. SAVE MODEL
# ==========================================

model.save(MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)

