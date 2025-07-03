import numpy as np
import pandas as pd
import cv2 as cv
import warnings
import os
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50, DenseNet121

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
data = data[~data["Finding Labels"].apply(lambda labels: 'No Finding' in labels)]
data = data.reset_index(drop=True)
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

# Print class distribution
print("\nClass distribution:")
for i, class_name in enumerate(mlb.classes_):
    pos_count = np.sum(labels[:, i])
    total_count = len(labels)
    print(f"{class_name}: {pos_count}/{total_count} ({pos_count / total_count * 100:.2f}%)")

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

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02)
    image = image + noise

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


# Stratified split for multi-label data
def stratified_split_multilabel(filepaths, labels, test_size=0.2, val_size=0.1, random_state=42):
    """Simple stratified split for multi-label data"""

    # Create a single label for stratification (most frequent positive label)
    single_labels = []
    for label_array in labels:
        if np.sum(label_array) == 0:  # No positive labels
            single_labels.append(-1)  # Special case
        else:
            # Use the first positive label
            single_labels.append(np.argmax(label_array))

    # Handle case where we have -1 labels (no positive labels)
    single_labels = np.array(single_labels)

    # First split: train+val vs test
    indices = np.arange(len(filepaths))

    # Stratified split
    try:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=single_labels, random_state=random_state
        )
    except ValueError:
        # If stratification fails, use random split
        print("Stratification failed, using random split")
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

    # Second split: train vs val
    train_val_labels = single_labels[train_val_idx]
    val_ratio = val_size / (1 - test_size)

    try:
        train_idx_local, val_idx_local = train_test_split(
            np.arange(len(train_val_idx)), test_size=val_ratio,
            stratify=train_val_labels, random_state=random_state
        )
    except ValueError:
        print("Validation stratification failed, using random split")
        train_idx_local, val_idx_local = train_test_split(
            np.arange(len(train_val_idx)), test_size=val_ratio, random_state=random_state
        )

    # Map back to original indices
    train_idx = train_val_idx[train_idx_local]
    val_idx = train_val_idx[val_idx_local]

    return train_idx, val_idx, test_idx


# Create stratified splits
train_idx, val_idx, test_idx = stratified_split_multilabel(filepaths, labels)

print(f"Train samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")
print(f"Test samples: {len(test_idx)}")

# Create file paths and labels for each split
train_files = filepaths[train_idx]
val_files = filepaths[val_idx]
test_files = filepaths[test_idx]

train_labels = labels[train_idx]
val_labels = labels[val_idx]
test_labels = labels[test_idx]

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))

# Apply preprocessing
train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Apply augmentation only to training data
train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch
BATCH_SIZE = 32  # Reduced batch size for better gradient updates
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Improved custom metrics with configurable threshold
def create_metrics_with_threshold(threshold=0.5):
    def f1_score_metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_mean(f1)

    def precision_score_metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        return tf.reduce_mean(precision)

    def recall_score_metric(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        return tf.reduce_mean(recall)

    return f1_score_metric, precision_score_metric, recall_score_metric


# Weighted focal loss implementation
def weighted_focal_loss(class_weights, gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate focal loss components
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)

        cross_entropy = -tf.math.log(p_t)
        weight = alpha_t * tf.pow((1 - p_t), gamma)

        # Apply class weights
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        # Broadcast class weights to match the shape
        class_weights_broadcast = tf.tile(tf.expand_dims(class_weights_tensor, 0), [tf.shape(y_true)[0], 1])

        focal_loss = weight * cross_entropy * class_weights_broadcast

        return tf.reduce_mean(focal_loss)

    return focal_loss_fixed


# Alternative: Weighted binary crossentropy
def weighted_binary_crossentropy(class_weights):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate weighted binary crossentropy
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        class_weights_broadcast = tf.tile(tf.expand_dims(class_weights_tensor, 0), [tf.shape(y_true)[0], 1])

        # Weighted loss
        loss_pos = y_true * tf.math.log(y_pred) * class_weights_broadcast
        loss_neg = (1 - y_true) * tf.math.log(1 - y_pred)

        return -tf.reduce_mean(loss_pos + loss_neg)

    return loss


# Enhanced model architecture
def create_enhanced_model(num_classes):
    # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze base model initially
    base_model.trainable = False

    # Enhanced head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)

    # Global pooling with both average and max
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])

    # Dense layers with proper regularization
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


# Find optimal threshold after training
def find_optimal_threshold(model, val_ds, mlb_classes):
    """Find optimal threshold for each class"""
    print("Finding optimal thresholds...")

    # Get predictions
    y_pred_proba = model.predict(val_ds, verbose=0)

    # Get true labels
    y_true = []
    for _, labels in val_ds:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)

    optimal_thresholds = []

    for class_idx in range(len(mlb_classes)):
        try:
            precision, recall, thresholds = precision_recall_curve(
                y_true[:, class_idx], y_pred_proba[:, class_idx]
            )

            # Calculate F1 scores for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

            # Find threshold with maximum F1 score
            if len(f1_scores) > 0:
                best_threshold_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
                max_f1 = f1_scores[best_threshold_idx]
            else:
                optimal_threshold = 0.5
                max_f1 = 0.0

            optimal_thresholds.append(optimal_threshold)

            print(f"Class {mlb_classes[class_idx]}: Threshold = {optimal_threshold:.3f}, F1 = {max_f1:.3f}")

        except Exception as e:
            print(f"Error finding threshold for class {mlb_classes[class_idx]}: {e}")
            optimal_thresholds.append(0.5)

    return optimal_thresholds


# Create and compile model
print("Creating enhanced model...")
model, base_model = create_enhanced_model(len(mlb.classes_))

# Start with default threshold metrics
f1_metric, precision_metric, recall_metric = create_metrics_with_threshold(threshold=0.5)

# Enhanced callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor loss initially
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        mode='min',
        min_lr=1e-7
    ),
    callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
]

# Phase 1: Train with frozen base model
print("Phase 1: Training with frozen base model...")
base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=weighted_focal_loss(class_weights, gamma=2.0, alpha=0.25),
    metrics=['binary_accuracy', f1_metric, precision_metric, recall_metric]
)

print("Model summary:")
model.summary()

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculate class weights
y_train_1d = labels.flatten()  # Make sure it's 1D
classes = np.unique(y_train_1d)
class_weights = compute_class_weight(
    'balanced',
    classes=classes,
    y=y_train_1d
)

# Convert to dictionary format that Keras expects
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}
print(f"Class weights: {class_weight_dict}")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_list,
    class_weight=class_weight_dict,
    verbose=1
)

# Find optimal threshold after Phase 1
optimal_thresholds = find_optimal_threshold(model, val_ds, mlb.classes_)
avg_threshold = np.mean(optimal_thresholds)
print(f"\nAverage optimal threshold: {avg_threshold:.3f}")

# Update metrics with optimal threshold
f1_metric, precision_metric, recall_metric = create_metrics_with_threshold(threshold=avg_threshold)

# Phase 2: Fine-tune with unfrozen model
print("\nPhase 2: Fine-tuning with unfrozen model...")
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) // 2

# Freeze earlier layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Fine-tuning from layer {fine_tune_at} onwards")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=weighted_focal_loss(class_weights, gamma=2.0, alpha=0.25),
    metrics=['binary_accuracy', f1_metric, precision_metric, recall_metric]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks_list,
    verbose=1
)

# Final threshold optimization
print("\nFinal threshold optimization...")
optimal_thresholds = find_optimal_threshold(model, val_ds, mlb.classes_)
avg_threshold = np.mean(optimal_thresholds)
print(f"Final average optimal threshold: {avg_threshold:.3f}")

# Final evaluation with optimal threshold
f1_metric, precision_metric, recall_metric = create_metrics_with_threshold(threshold=avg_threshold)

print("\nFinal evaluation on test set:")
test_loss, test_acc, test_f1, test_precision, test_recall = model.evaluate(test_ds, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Detailed per-class evaluation
print("\nDetailed per-class evaluation:")
y_pred_proba = model.predict(test_ds, verbose=0)
y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

print(f"{'Class Name':<20} {'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-" * 80)

for i, class_name in enumerate(mlb.classes_):
    threshold = optimal_thresholds[i]
    y_pred_class = (y_pred_proba[:, i] > threshold).astype(int)

    # Calculate metrics using sklearn for verification
    precision = precision_score(y_true[:, i], y_pred_class, zero_division=0)
    recall = recall_score(y_true[:, i], y_pred_class, zero_division=0)
    f1 = f1_score(y_true[:, i], y_pred_class, zero_division=0)
    support = np.sum(y_true[:, i])

    print(f"{class_name:<20} {threshold:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10}")

# Save model and related files
print("\nSaving model and related files...")
model.save('final_chest_xray_model.h5')
joblib.dump(mlb, 'mlb_chest_xray.pkl')
joblib.dump(optimal_thresholds, 'optimal_thresholds.pkl')
joblib.dump(class_weights, 'class_weights.pkl')

# Save training history
history_combined = {
    'phase1': history1.history,
    'phase2': history2.history
}
joblib.dump(history_combined, 'training_history.pkl')

print("Model training completed and saved!")
print(f"Files saved:")
print(f"  - final_chest_xray_model.h5")
print(f"  - mlb_chest_xray.pkl")
print(f"  - optimal_thresholds.pkl")
print(f"  - class_weights.pkl")
print(f"  - training_history.pkl")

# Optional: Plot training history
try:
    import matplotlib.pyplot as plt


    def plot_training_history(history1, history2):
        # Combine histories
        metrics = ['loss', 'binary_accuracy', 'f1_score_metric', 'precision_score_metric', 'recall_score_metric']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break

            # Get training and validation data
            train_data = history1.history.get(metric, []) + history2.history.get(metric, [])
            val_data = history1.history.get(f'val_{metric}', []) + history2.history.get(f'val_{metric}', [])

            if train_data and val_data:
                epochs = range(1, len(train_data) + 1)

                axes[i].plot(epochs, train_data, 'b-', label=f'Training {metric}')
                axes[i].plot(epochs, val_data, 'r-', label=f'Validation {metric}')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epochs')
                axes[i].set_ylabel(metric.replace("_", " ").title())
                axes[i].legend()
                axes[i].grid(True)

        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training history plot saved as 'training_history.png'")


    plot_training_history(history1, history2)

except ImportError:
    print("Matplotlib not available. Skipping plot generation.")
except Exception as e:
    print(f"Error plotting training history: {e}")

print("\nTraining completed successfully!")