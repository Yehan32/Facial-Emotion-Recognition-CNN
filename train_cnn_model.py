import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE      = (128, 128)
BATCH_SIZE    = 16
EPOCHS        = 100
LEARNING_RATE = 0.001
NUM_CLASSES   = 6

DATASETS = {
    'CK':    'processed_CK_dataset',
    'JAFFE': 'processed_JAFFE_dataset',
}

# ============================================================
# DATA LOADING
# ============================================================

def build_data_generators(dataset_path):
    """
    Training generator applies mild augmentation.
    Test generator only rescales - no augmentation to avoid bias.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=IMG_SIZE, color_mode='grayscale',
        batch_size=BATCH_SIZE, class_mode='categorical',
        shuffle=True, seed=42,
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=IMG_SIZE, color_mode='grayscale',
        batch_size=BATCH_SIZE, class_mode='categorical',
        shuffle=False,
    )
    return train_gen, test_gen, list(train_gen.class_indices.keys())


# ============================================================
# CNN ARCHITECTURE
# ============================================================

def build_custom_cnn(num_classes=NUM_CLASSES):
    """
    Three-block CNN with GlobalAveragePooling2D head.

    Design rationale:
    - Two Conv layers per block increase representational capacity
      without adding dense-layer parameters.
    - GlobalAveragePooling2D keeps the head at ~17K params (vs 8.3M
      with Flatten+Dense(256)), preventing the overfitting seen in
      the first attempt.
    - Dropout set to 0.25 (spatial) and 0.4 (dense) — lower than
      the second attempt (0.3/0.5) which caused under-fitting, higher
      than zero to regularise.
    - No L2 regularisation: unnecessary once Dropout and GAP are used.
    - Learning rate 0.001 with ReduceLROnPlateau provides adaptive
      scheduling without manual tuning.

    Total parameters: ~204K — appropriate for ~400–800 training samples.
    """
    model = models.Sequential([
        # ---- Block 1: 32 filters ----
        layers.Input(shape=(*IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ---- Block 2: 64 filters ----
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ---- Block 3: 128 filters ----
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ---- Lightweight head (GlobalAveragePooling avoids 8M-param Dense) ----
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(train_gen, test_gen, dataset_name):
    model = build_custom_cnn(num_classes=NUM_CLASSES)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    print(f"\nModel Summary ({dataset_name}):")
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=8, min_lr=1e-6, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=f'best_cnn_{dataset_name}.keras',
            monitor='val_accuracy', save_best_only=True, verbose=0,
        ),
    ]

    # Inverse-frequency class weights
    class_counts = np.array([
        len(os.listdir(os.path.join(train_gen.directory, c)))
        for c in sorted(train_gen.class_indices, key=train_gen.class_indices.get)
    ])
    total = class_counts.sum()
    n_classes = len(class_counts)
    class_weight = {i: total / (n_classes * count) for i, count in enumerate(class_counts)}
    print(f"\nClass weights: { {k: round(v, 3) for k, v in class_weight.items()} }")

    history = model.fit(
        train_gen, validation_data=test_gen,
        epochs=EPOCHS, callbacks=cb_list,
        class_weight=class_weight, verbose=1,
    )
    return model, history


# ============================================================
# EVALUATION & VISUALISATION
# ============================================================

def plot_training_history(history, dataset_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History - {dataset_name}', fontsize=14, fontweight='bold')

    epochs = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs, history.history['accuracy'],     label='Train',      color='steelblue')
    ax1.plot(epochs, history.history['val_accuracy'], label='Validation', color='darkorange')
    ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.history['loss'],     label='Train',      color='steelblue')
    ax2.plot(epochs, history.history['val_loss'], label='Validation', color='darkorange')
    ax2.set_title('Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'CNN_training_history_{dataset_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_confusion_matrix(y_true, y_pred, class_names, dataset_name, accuracy):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_title(f'Confusion Matrix - {dataset_name}  (Accuracy: {accuracy:.2%})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Emotion')
    ax.set_ylabel('True Emotion')
    plt.tight_layout()
    out = f'CNN_confusion_matrix_{dataset_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_sample_predictions(model, test_gen, class_names, dataset_name, n=9):
    test_gen.reset()
    images, labels_onehot = next(test_gen)
    preds = model.predict(images, verbose=0)
    y_true = np.argmax(labels_onehot, axis=1)
    y_pred = np.argmax(preds, axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.suptitle(f'Sample Predictions - {dataset_name}', fontsize=13, fontweight='bold')
    for idx, ax in enumerate(axes.flat):
        if idx >= n or idx >= len(images):
            ax.axis('off'); continue
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.axis('off')
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        ax.set_title(
            f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
            fontsize=9, color=color,
        )
    plt.tight_layout()
    out = f'CNN_sample_predictions_{dataset_name}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def evaluate_model(model, test_gen, class_names, dataset_name):
    print(f"\n{'='*60}\nEVALUATING ON: {dataset_name}\n{'='*60}")
    test_gen.reset()
    loss, accuracy = model.evaluate(test_gen, verbose=0)
    print(f"  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}  ({accuracy:.2%})")

    test_gen.reset()
    y_true, y_pred = [], []
    for _ in range(len(test_gen)):
        imgs, lbls = next(test_gen)
        preds = model.predict(imgs, verbose=0)
        y_true.extend(np.argmax(lbls, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print("\nPer-class Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names,
                                digits=3, zero_division=0))

    plot_confusion_matrix(y_true, y_pred, class_names, dataset_name, accuracy)
    plot_sample_predictions(model, test_gen, class_names, dataset_name)
    return accuracy, y_true, y_pred


# ============================================================
# MAIN
# ============================================================

def run_pipeline(dataset_key, dataset_path):
    print(f"\n{'='*60}\nCUSTOM CNN PIPELINE: {dataset_key}\n{'='*60}")
    train_gen, test_gen, class_names = build_data_generators(dataset_path)
    print(f"  Classes:       {class_names}")
    print(f"  Train samples: {train_gen.samples}")
    print(f"  Test  samples: {test_gen.samples}")

    model, history = train_model(train_gen, test_gen, dataset_key)
    plot_training_history(history, dataset_key)
    accuracy, y_true, y_pred = evaluate_model(model, test_gen, class_names, dataset_key)

    results = {
        'dataset': dataset_key, 'class_names': class_names,
        'test_accuracy': accuracy, 'history': history.history,
        'y_true': y_true, 'y_pred': y_pred,
    }
    with open(f'CNN_results_{dataset_key}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Results saved to: CNN_results_{dataset_key}.pkl")
    return results


def compare_results(ck_results, jaffe_results):
    print("\n" + "="*60)
    print("RESULTS COMPARISON: Custom CNN vs HOG+SVM Baseline")
    print("="*60)
    print(f"{'Dataset':<10} {'HOG+SVM':>12} {'Custom CNN':>12} {'Change':>12}")
    print("-"*50)
    baselines = {'CK': 0.6084, 'JAFFE': 0.6182}
    for key, res in [('CK', ck_results), ('JAFFE', jaffe_results)]:
        b = baselines[key]
        c = res['test_accuracy']
        sign = '+' if c - b >= 0 else ''
        print(f"{key:<10} {b:>11.2%} {c:>11.2%} {sign}{c-b:>11.2%}")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FACIAL EMOTION RECOGNITION - CUSTOM CNN (Assignment 2)")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    results = {}
    for key, path in DATASETS.items():
        if os.path.isdir(path):
            results[key] = run_pipeline(key, path)
        else:
            print(f"\n  WARNING: Dataset not found: {path}")

    if 'CK' in results and 'JAFFE' in results:
        compare_results(results['CK'], results['JAFFE'])

    print("\n" + "="*60)
    print("CNN TRAINING COMPLETE")
    print("="*60)