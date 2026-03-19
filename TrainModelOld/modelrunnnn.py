"""

dataset: GTE9 (205 classes, ~10 videos/class)
shape: (70, 63, 4) - 70 frames × 63 landmarks × 4 features

venv: source /home/aryan/opensource_lab_proj/venv/bin/activate
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, multilabel_confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import json
import csv


# ==================== CONFIGURATION ====================
LANDMARK_PATH = "../gte9_landmarks"
MODEL_PATH = "../models"
RESULTS_PATH = "../testing/model_comparison_results"

SEQUENCE_LENGTH = 70
NUM_LANDMARKS = 63
NUM_FEATURES = 4
FEATURE_DIM = NUM_LANDMARKS * NUM_FEATURES  # 252

EPOCHS = 140
BATCH_SIZE = 32  
TEST_SIZE = 0.2
VAL_SIZE = 0.15

# Augmentation settings
AUGMENTATION_FACTOR = 8  # 8x data augmentation (increased to combat overfitting)
INCREMENTAL_JSON = os.path.join(RESULTS_PATH, "incremental_results.json")

# Create results directory
os.makedirs(RESULTS_PATH, exist_ok=True)

# ==================== GPU CONFIGURATION ====================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" GPU detected: {gpus[0].name}")
        
        # Enable mixed precision for faster training
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f" Mixed precision enabled: {policy.name}")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("⚠️ No GPU detected, running on CPU.")



def augment_landmarks(X, augmentation_factor=5, preserve_normalization=True):

    print(f"\nApplying {augmentation_factor}x data augmentation...")
    print(f"Original shape: {X.shape}")
    
    augmented_data = []
    
    for video in X:
        # Always include original
        augmented_data.append(video)
        
        for _ in range(augmentation_factor - 1):
            aug_video = video.copy()
            
            # 1. ROTATION (around z-axis, ±15 degrees)
            # Rotates in x-y plane while preserving normalization
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            x_coords = aug_video[:, :, 0].copy()
            y_coords = aug_video[:, :, 1].copy()
            
            aug_video[:, :, 0] = cos_a * x_coords - sin_a * y_coords
            aug_video[:, :, 1] = sin_a * x_coords + cos_a * y_coords
            
            # 2. SCALING (±10%)
            # Simulates different body sizes while keeping proportions
            scale = np.random.uniform(0.90, 1.10)
            aug_video[:, :, :3] *= scale
            
            # 3. TRANSLATION (small shifts)
            # Simulates different camera positions
            shift_x = np.random.uniform(-0.05, 0.05)
            shift_y = np.random.uniform(-0.05, 0.05)
            aug_video[:, :, 0] += shift_x
            aug_video[:, :, 1] += shift_y
            
            # 4. TEMPORAL JITTERING (time shifts)
            # Simulates different signing speeds
            if np.random.random() > 0.5:
                shift = np.random.randint(-3, 4)
                aug_video = np.roll(aug_video, shift, axis=0)
            
            # 5. GAUSSIAN NOISE (small, realistic)
            # Simulates detection uncertainty
            noise_std = 0.01
            noise = np.random.normal(0, noise_std, aug_video[:, :, :3].shape)
            aug_video[:, :, :3] += noise
            
            # 6. RANDOM FRAME DROPOUT (simulate missing detections)
            # Mirrors real-world detection failures
            if np.random.random() > 0.7:
                num_frames_to_drop = np.random.randint(1, 5)
                frames_to_drop = np.random.choice(70, num_frames_to_drop, replace=False)
                aug_video[frames_to_drop] = 0.0
            

            # 8. VISIBILITY PERTURBATION (for pose landmarks)
            # Randomly adjust visibility scores slightly
            if np.random.random() > 0.7:
                vis_noise = np.random.uniform(-0.1, 0.1, aug_video[:, :21, 3].shape)
                aug_video[:, :21, 3] = np.clip(aug_video[:, :21, 3] + vis_noise, 0, 1)
            
            augmented_data.append(aug_video)
    
    result = np.array(augmented_data, dtype=np.float32)
    print(f"ugmented shape: {result.shape}")
    print(f" Augmentation ratio: {augmentation_factor}x")
    
    return result


# ==================== DATA LOADING ====================

def turn_data_into_readable_for_ltsm(
    dataset_landmark_path=LANDMARK_PATH,
):
    X, y = [], []

    for word in os.listdir(dataset_landmark_path):

        word_folder = os.path.join(dataset_landmark_path, word)

        if os.path.isdir(word_folder):
            for npfile in os.listdir(word_folder):
                if npfile.endswith(".npy"):
                    # shape  =  70, 57, 4
                    arr = np.load(os.path.join(word_folder, npfile))

                    # new shape = 70,57*4  = 70,228
                    arr = arr.reshape(arr.shape[0], -1)
                    print(arr.shape)
                    X.append(arr)
                    y.append(word)
    return np.array(X, dtype=np.float32), np.array(y)


def load_data():
    
    
    
    if os.path.exists(f"{LANDMARK_PATH}/x.npy"):
        X = np.load(f"{LANDMARK_PATH}/x.npy")
        y = np.load(f"{LANDMARK_PATH}/y.npy")
        y_encoded = np.load(f"{LANDMARK_PATH}/y_encoded.npy")
        y_onehot = np.load(f"{LANDMARK_PATH}/y_onehot.npy")
        
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        
        
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Classes: {len(label_encoder.classes_)}")
        print(f"   Samples per class: ~{len(X) / len(label_encoder.classes_):.1f}")
        
        return X, y, y_encoded, y_onehot, label_encoder
    else:
        x, y = turn_data_into_readable_for_ltsm()
        print(y)
        print(x)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_onehot = to_categorical(y_encoded)

        print(y.shape)
        print(y[0].shape)
        np.save(LANDMARK_PATH+"/x.npy", x)
        np.save(LANDMARK_PATH+"/y.npy", y)
        np.save(LANDMARK_PATH+"/y_encoded.npy", y_encoded)
        np.save(LANDMARK_PATH+"/y_onehot.npy", y_onehot)
        
        return x, y, y_encoded, y_onehot, label_encoder




def build_lightweight_bilstm_balanced_regularization(num_classes, name="Lightweight_BiLSTM_Balanced_Regularization"):

    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        
        
        layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                         kernel_regularizer=tf.keras.regularizers.l2(0.008))),
        layers.BatchNormalization(),
        layers.Dropout(0.55),
        
        layers.Bidirectional(layers.LSTM(32,
                                         kernel_regularizer=tf.keras.regularizers.l2(0.008))),
        layers.BatchNormalization(),
        layers.Dropout(0.55),
        
        
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.010)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ], name=name)
    
    return model



def build_bigru_balanced_regularization(num_classes, name="BiGRU_Balanced_Regularization"):

    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
        
        
        layers.Bidirectional(layers.GRU(96, return_sequences=True,
                        kernel_regularizer=tf.keras.regularizers.l2(0.005))),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        
        layers.Bidirectional(layers.GRU(48,
                        kernel_regularizer=tf.keras.regularizers.l2(0.005))),
        layers.BatchNormalization(),
        layers.Dropout(0.45),
        
        
        layers.Dense(64, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.006)),
        layers.BatchNormalization(),
        layers.Dropout(0.45),
        
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ], name=name)
    
    return model



def build_3dcnn_model(num_classes, name="3D_CNN"):

    inputs = layers.Input(shape=(SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES, 1))
    
    # First 3D Conv block
    x = layers.Conv3D(32, kernel_size=(3, 3, 2), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = layers.Dropout(0.4)(x)
    
    # Second 3D Conv block
    x = layers.Conv3D(64, kernel_size=(3, 3, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = layers.Dropout(0.4)(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


# ==================== TRAINING UTILITIES ====================

def get_callbacks(model_name):
    
    early_stop = EarlyStopping(
        monitor='val_loss',  
        patience=18,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        mode='min',
        min_lr=1e-6,
        verbose=1
    )
    
    # Save with timestamp to prevent overwriting good models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = ModelCheckpoint(
        f"{RESULTS_PATH}/{model_name}_{timestamp}_best.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Also keep a "latest" version for convenience
    checkpoint_latest = ModelCheckpoint(
        f"{RESULTS_PATH}/{model_name}_latest.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )
    
    return [early_stop, reduce_lr, checkpoint, checkpoint_latest]


def train_model(model, X_train, y_train, X_val, y_val, model_name, class_weight_dict):
    """Train a model and return history"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.10)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.006),
        loss=loss_fn,
        metrics=['accuracy', top5]
    )
    
    model.summary()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=get_callbacks(model_name),
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    """Evaluate model and generate reports"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Predictions with simple test-time augmentation (noise-averaged)
    y_pred = model.predict(X_test, verbose=0)
    tta_runs = 4
    for _ in range(tta_runs):
        noise = np.random.normal(0, 0.002, X_test.shape).astype(np.float32)
        y_pred += model.predict(X_test + noise, verbose=0)
    y_pred /= (tta_runs + 1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    eval_vals = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(eval_vals, list):
        loss, accuracy, top5_acc = eval_vals
    else:
        loss, accuracy = eval_vals, None
        top5_acc = None
    print(f" Test Accuracy: {accuracy*100:.2f}%")
    print(f" Top-5 Accuracy: {top5_acc*100:.2f}%" if top5_acc is not None else "")
    print(f" Test Loss: {loss:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true_labels, y_pred_labels,
        target_names=label_encoder.classes_,
        zero_division=0,
        digits=3
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Aggregate TP/TN/FP/FN across one-vs-rest per class
    mcm = multilabel_confusion_matrix(y_true_labels, y_pred_labels, labels=range(len(label_encoder.classes_)))
    tn = int(mcm[:, 0, 0].sum())
    fp = int(mcm[:, 0, 1].sum())
    fn = int(mcm[:, 1, 0].sum())
    tp = int(mcm[:, 1, 1].sum())
    binary_confusion = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    
    return accuracy, loss, cm, y_pred_labels, y_true_labels, top5_acc, binary_confusion, y_pred


def save_top5_predictions(y_pred, y_true_labels, label_encoder, model_name):
    """Save detailed Top-5 predictions per sample to CSV and JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get top-5 class indices for each sample
    top5_indices = np.argsort(y_pred, axis=1)[:, -5:][:, ::-1]  # Shape: (num_samples, 5)
    
    # Get top-5 probabilities
    top5_probs = np.take_along_axis(y_pred, top5_indices, axis=1)
    
    # Convert indices to class names
    top5_class_names = label_encoder.inverse_transform(top5_indices.flatten()).reshape(-1, 5)
    
    # True class names
    true_class_names = label_encoder.inverse_transform(y_true_labels)
    
    # Prepare data for export
    results_list = []
    for i in range(len(y_true_labels)):
        result = {
            'sample_id': i,
            'true_class': true_class_names[i],
            'correct': true_class_names[i] in top5_class_names[i],
            'top1_class': top5_class_names[i][0],
            'top1_prob': float(top5_probs[i][0]),
            'top2_class': top5_class_names[i][1],
            'top2_prob': float(top5_probs[i][1]),
            'top3_class': top5_class_names[i][2],
            'top3_prob': float(top5_probs[i][2]),
            'top4_class': top5_class_names[i][3],
            'top4_prob': float(top5_probs[i][3]),
            'top5_class': top5_class_names[i][4],
            'top5_prob': float(top5_probs[i][4]),
        }
        results_list.append(result)
    
    # Save as CSV
    csv_file = f"{RESULTS_PATH}/{model_name}_{timestamp}_top5_predictions.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)
    print(f" Top-5 predictions saved to CSV: {csv_file}")
    
    # Save as JSON (more detailed format)
    json_file = f"{RESULTS_PATH}/{model_name}_{timestamp}_top5_predictions.json"
    with open(json_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f" Top-5 predictions saved to JSON: {json_file}")
    
    # Print summary statistics
    correct_top1 = sum(1 for r in results_list if r['top1_class'] == r['true_class'])
    correct_top5 = sum(1 for r in results_list if r['correct'])
    print(f"\n Top-5 Analysis:")
    print(f"   Top-1 Correct: {correct_top1}/{len(results_list)} ({correct_top1/len(results_list)*100:.2f}%)")
    print(f"   Top-5 Correct: {correct_top5}/{len(results_list)} ({correct_top5/len(results_list)*100:.2f}%)")
    print(f"   Improvement: +{(correct_top5-correct_top1)/len(results_list)*100:.2f}% from Top-1 to Top-5")


def plot_results(history, cm, label_encoder, model_name, binary_confusion):
    """Generate visualization plots (training curves + compact 2x2 confusion)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/{model_name}_{timestamp}_training_curves.png", dpi=150)
    plt.close()
    
    # Compact 2x2 confusion matrix (TN, FP; FN, TP)
    plt.figure(figsize=(6, 5))
    mat = np.array([[binary_confusion['tn'], binary_confusion['fp']],
                    [binary_confusion['fn'], binary_confusion['tp']]])
    im = plt.imshow(mat, cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ['Pred: Neg', 'Pred: Pos'])
    plt.yticks([0, 1], ['True: Neg', 'True: Pos'])
    for (i, j), val in np.ndenumerate(mat):
        plt.text(j, i, f"{val}", ha='center', va='center', fontsize=12, color='black')
    plt.title(f'{model_name} - Binary Confusion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/{model_name}_{timestamp}_binary_confusion.png", dpi=150)
    plt.close()
    
    print(f" Plots saved to {RESULTS_PATH}/")


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("ASL SIGN LANGUAGE RECOGNITION - COMPREHENSIVE MODEL TESTING")
    
    
    # Load data
    X, y, y_encoded, y_onehot, label_encoder = load_data()
    num_classes = len(label_encoder.classes_)
    
    print(f"\nDataset Statistics:")
    print(f"   Total samples: {len(X)}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Samples per class: {len(X) / num_classes:.1f}")
    print(f"   Data shape (loaded): {X.shape}")
    
    # Optional class pruning by count to improve generalization
    MIN_SAMPLES_PER_CLASS = 8
    print(f"\n🔧 Pruning classes with < {MIN_SAMPLES_PER_CLASS} samples (if any)...")
    # Build counts
    unique, counts = np.unique(y, return_counts=True)
    keep_classes = set(unique[counts >= MIN_SAMPLES_PER_CLASS])
    if len(keep_classes) < len(unique):
        keep_mask = np.array([label in keep_classes for label in y])
        X = X[keep_mask]
        y = y[keep_mask]
        # Rebuild encoder and one-hot
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)
        print(f"Pruned to {num_classes} classes; samples: {len(X)}")
    else:
        print("No pruning applied; all classes meet minimum samples.")

    # Reshape from (N, 70, 252) to (N, 70, 63, 4) for augmentation
    X = X.reshape(-1, SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES)
    print(f"   Data shape (reshaped): {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=TEST_SIZE, random_state=123, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=np.argmax(y_train, axis=1), random_state=123
    )
    
    print(f"\nData Split:")
    print(f"   Training: {X_train.shape[0]} samples, shape: {X_train.shape}")
    print(f"   Validation: {X_val.shape[0]} samples, shape: {X_val.shape}")
    print(f"   Test: {X_test.shape[0]} samples, shape: {X_test.shape}")
    
    # Apply augmentation ONLY to training data
    # X_train/val/test are already in shape (N, 70, 63, 4)
    X_train_aug = augment_landmarks(X_train, augmentation_factor=AUGMENTATION_FACTOR)
    y_train_aug = np.repeat(y_train, AUGMENTATION_FACTOR, axis=0)
    
    print(f"\nTraining data after augmentation: {X_train_aug.shape[0]} samples")
    print(f"   Shape: {X_train_aug.shape}")
    
    # Reshape for LSTM/GRU models (flatten landmarks)
    X_train_flat = X_train_aug.reshape(-1, SEQUENCE_LENGTH, FEATURE_DIM)
    X_val_flat = X_val.reshape(-1, SEQUENCE_LENGTH, FEATURE_DIM)
    X_test_flat = X_test.reshape(-1, SEQUENCE_LENGTH, FEATURE_DIM)
    
    
    print(f"   Training: {X_train_flat.shape}")
    print(f"   Validation: {X_val_flat.shape}")
    print(f"   Test: {X_test_flat.shape}")
    
    # Keep 4D shape for 3D CNN and hybrid models
    X_train_3d = X_train_aug  # (N, 70, 63, 4)
    X_val_3d = X_val  # (N, 70, 63, 4)
    X_test_3d = X_test  # (N, 70, 63, 4)
    
    
    print(f"   Training: {X_train_3d.shape}")
    print(f"   Validation: {X_val_3d.shape}")
    print(f"   Test: {X_test_3d.shape}")
    
    # Compute class weights
    y_train_labels = np.argmax(y_train_aug, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    
    # Store results
    results = {}
    

    
    print("2️⃣  Testing: Lightweight Bi-LSTM - Balanced Regularization ⭐ RECOMMENDED")
    
    model = build_lightweight_bilstm_balanced_regularization(num_classes)
    history = train_model(model, X_train_flat, y_train_aug, X_val_flat, y_val, 
                         "Lightweight_BiLSTM_Balanced_Regularization", class_weight_dict)
    acc, loss, cm, y_pred_labels, y_true, top5, bin_conf, y_pred_probs = evaluate_model(model, X_test_flat, y_test, 
                                                    label_encoder, "Lightweight_BiLSTM_Balanced_Regularization")
    plot_results(history, cm, label_encoder, "Lightweight_BiLSTM_Balanced_Regularization", bin_conf)
    save_top5_predictions(y_pred_probs, y_true, label_encoder, "Lightweight_BiLSTM_Balanced_Regularization")
    # Capture final train accuracy
    train_acc = float(history.history.get('accuracy', [None])[-1]) if 'accuracy' in history.history else None
    results['Lightweight_BiLSTM_Balanced_Regularization'] = {
        'accuracy': acc,
        'loss': loss,
        'top5': float(top5) if top5 is not None else None,
        'train_accuracy': train_acc
    }
    # Incremental JSON save
    try:
        existing = {}
        if os.path.exists(INCREMENTAL_JSON):
            with open(INCREMENTAL_JSON, 'r') as jf:
                try:
                    existing = json.load(jf)
                except Exception:
                    existing = {}
        existing["Lightweight_BiLSTM_Balanced_Regularization"] = results['Lightweight_BiLSTM_Balanced_Regularization']
        with open(INCREMENTAL_JSON, 'w') as jf:
            json.dump(existing, jf, indent=2)
        print(f" Incremental results saved to {INCREMENTAL_JSON}")
    except Exception as e:
        print(f"⚠️ Could not save incremental JSON: {e}")
    del model
    
    
    
    print(" FINAL RESULTS SUMMARY")
    
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<25} {'Accuracy':<12} {'Top-5':<10} {'Train Acc':<12} {'Loss':<10}")
    print("-" * 90)
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        top5_str = f"{metrics['top5']*100:>6.2f}%" if metrics.get('top5') is not None else "   n/a  "
        train_acc_str = f"{metrics.get('train_accuracy', 0)*100:>6.2f}%" if metrics.get('train_accuracy') is not None else "   n/a  "
        print(f"{rank:<6} {model_name:<25} {metrics['accuracy']*100:>6.2f}%     {top5_str:<10} {train_acc_str:<12} {metrics['loss']:>6.4f}")
    
    # Save summary
    summary_file = f"{RESULTS_PATH}/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write("ASL SIGN LANGUAGE RECOGNITION - MODEL COMPARISON RESULTS\n")
        
        f.write(f"Dataset: GTE9 ({num_classes} classes)\n")
        f.write(f"Augmentation: {AUGMENTATION_FACTOR}x\n")
        f.write(f"Training samples: {X_train_aug.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n\n")
        f.write(f"{'Rank':<6} {'Model':<25} {'Accuracy':<12} {'Top-5':<10} {'Train Acc':<12} {'Loss':<10}\n")
        f.write("-" * 90 + "\n")
        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            top5_str = f"{metrics['top5']*100:>6.2f}%" if metrics.get('top5') is not None else "   n/a  "
            train_acc_str = f"{metrics.get('train_accuracy', 0)*100:>6.2f}%" if metrics.get('train_accuracy') is not None else "   n/a  "
            f.write(f"{rank:<6} {model_name:<25} {metrics['accuracy']*100:>6.2f}%     {top5_str:<10} {train_acc_str:<12} {metrics['loss']:>6.4f}\n")
    
    print(f"\n Summary saved to: {summary_file}")
    print(f" All results saved to: {RESULTS_PATH}/")
    
    print("🎉 Testing Complete!")
    


if __name__ == "__main__":
    main()
