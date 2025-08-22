import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, argparse, json, ast, random
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report


os.environ["PYTHONHASHSEED"] = "123"
os.environ["TF_DETERMINISTIC_OPS"] = "1"     # make GPU ops more deterministic
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"   # cuDNN convs deterministic

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# turn images into tf datasets
def build_ds(image_paths, img_size=(256, 256), batch_size=32, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        image_paths,
        labels="inferred",  # copy labels from directory structure
        label_mode="int",   # the labels are integers 0,1,2,3
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=123,
    )

# CNN model
def cnn_model(img_size=(256,256), drop=0.4, num_classes=4):
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(img_size[0], img_size[1]),
        layers.Rescaling(1.0/255)
    ])
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=123),   
        layers.RandomRotation(0.05, seed=123),       
        layers.RandomZoom(0.10, seed=123),           
    
    ])

    model = models.Sequential([
        resize_and_rescale,
        data_augmentation,

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),


        layers.Dropout(drop),
        layers.Dense(4, activation='softmax'),

    ])



    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # accept both spellings but we'll override to the local SageMaker dir after parsing
    parser.add_argument("--model-dir", "--model_dir", dest="model_dir",
                        default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train",      default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--test",       default=os.environ.get("SM_CHANNEL_TEST"))

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=str, default="[256, 256]")

    args, _ = parser.parse_known_args()

    # Force model_dir to the local container path SageMaker expects.
    # This avoids TF trying to write directly to s3:// (which it can't).
    args.model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    try:
        img_size = tuple(ast.literal_eval(args.img_size))
    except Exception:
        img_size = (256, 256)

    print("Tensorflow Version:", tf.__version__)
    print(f"[INFO] Channels -> train: {args.train} | val: {args.validation} | test: {args.test}")
    print(f"[INFO] Model dir: {args.model_dir}")

    # 1) Build datasets directly (no unzip needed)
    print("[INFO] Building datasets...")
    train_ds = build_ds(args.train, img_size, args.batch_size, shuffle=True)
    val_ds   = build_ds(args.validation, img_size, args.batch_size, shuffle=False)
    test_ds  = build_ds(args.test, img_size, args.batch_size, shuffle=False)

    # Capture class names BEFORE any mapping/prefetch to avoid attribute loss
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Class names:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    # 2) Build & train your CNN
    print("[INFO] Building CNN model...")
    model = cnn_model(img_size=img_size, num_classes=num_classes)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, verbose=1)
    print()

    # Plot learning curves
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    os.makedirs(args.model_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, "learning_curves.png"))
    plt.close()

    # 3) Evaluate on test data
    metrics = model.evaluate(test_ds, verbose=2)
    print("Test metrics:", dict(zip(model.metrics_names, metrics)))

    # Get predictions for test set
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 4) Save model + class names (to local /opt/ml/model — SageMaker will upload to S3 for you)
    export_dir = os.path.join(args.model_dir, "1")   # TF Serving expects versioned folder
    tf.saved_model.save(model, export_dir)
    with open(os.path.join(args.model_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)
