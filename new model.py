import numpy as np
import pandas as pd
import cv2 as cv
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load and preprocess data
data = pd.read_csv(r'C:\Users\Uday Hiremath\Downloads\CNN_DATASET\Chest_XRay_Dataset\Exist_data_set.csv')

print("Class distribution:")
print(data['Finding Labels'].value_counts()[:20])

# Process labels
data["Finding Labels"] = data["Finding Labels"].str.split('|')
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(data["Finding Labels"])

print(f"Number of classes: {len(mlb.classes_)}")
print(f"Classes: {mlb.classes_}")

# Calculate class weights for imbalanced data
class_weights = []
for i in range(len(mlb.classes_)):
    pos_samples = np.sum(labels[:, i])
    neg_samples = len(labels) - pos_samples
    if pos_samples > 0:
        weight = neg_samples / pos_samples
    else:
        weight = 1.0
    class_weights.append(weight)

class_weights = np.array(class_weights)
print(f"Class weights: {class_weights}")

filepaths = data['filepath'].values


# Enhanced preprocessing function
def preprocess_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])

    # Convert to grayscale and back to RGB for X-ray images
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)

    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # Apply histogram equalization-like enhancement
    image = tf.image.adjust_contrast(image, contrast_factor=1.2)
    image = tf.image.adjust_brightness(image, delta=0.1)

    # Clip values to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


# Data augmentation function for training
def augment_image(image, label):
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random rotation (small angle for medical images)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    # Random zoom
    image = tf.image.random_crop(image, size=[int(224 * 0.9), int(224 * 0.9), 3])
    image = tf.image.resize(image, [224, 224])

    # Random brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Add slight noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, label


# Create datasets
dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Split data
total_size = len(dataset)
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
test_size = total_size - train_size - val_size

# Create train, validation, and test sets
train_ds = dataset.take(train_size)
remaining_ds = dataset.skip(train_size)
val_ds = remaining_ds.take(val_size)
test_ds = remaining_ds.skip(val_size)

# Apply augmentation only to training data
train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch
BATCH_SIZE = 100  # Reduced batch size for better gradient updates
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Enhanced model architecture
def create_enhanced_model(num_classes, model_type='efficientnet'):
    if model_type == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_type == 'densenet':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze base model initially
    base_model.trainable = False

    # Enhanced head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)

    # Global pooling with both average and max
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])

    # Dense layers with proper activations
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


# Custom metrics
def f1_score(y_true, y_pred):
    # Cast both to float32 to ensure compatibility
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)


def precision_score(y_true, y_pred):
    # Cast both to float32 to ensure compatibility
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    return tf.reduce_mean(precision)


def recall_score(y_true, y_pred):
    # Cast both to float32 to ensure compatibility
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return tf.reduce_mean(recall)


# Focal loss for imbalanced data
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)

        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = weight * cross_entropy

        return tf.reduce_mean(focal_loss)

    return focal_loss_fixed


# Create and compile model
print("Creating enhanced model...")
model, base_model = create_enhanced_model(len(mlb.classes_), 'efficientnet')

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        'best_chest_xray_model.h5',
        monitor='val_f1_score',
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# Phase 1: Train with frozen base model
print("Phase 1: Training with frozen base model...")
model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss='binary_crossentropy',  # You can also try focal_loss()
    metrics=['binary_accuracy', f1_score, precision_score, recall_score]
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks_list,
    verbose=1
)

# Phase 2: Fine-tune with unfrozen model
print("Phase 2: Fine-tuning with unfrozen model...")
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) // 2

# Freeze earlier layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy', f1_score, precision_score, recall_score]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks_list,
    verbose=1
)

# Final evaluation
print("Final evaluation on test set:")
test_loss, test_acc, test_f1, test_precision, test_recall = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Save the trained model and label binarizer
model.save('final_chest_xray_model.h5')
joblib.dump(mlb, 'mlb_chest_xray.pkl')

print("Model training completed and saved!")

# Optional: Plot training history
import matplotlib.pyplot as plt


def plot_training_history(history1, history2):
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Uncomment to plot training history
# plot_training_history(history1, history2)