import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
import mlflow
import mlflow.tensorflow
import os

# Set MLflow tracking URI to your MLflow server
mlflow.set_tracking_uri("http://mlflow:5500")  # Change if your mlflow server is at different address

# Start MLflow run
with mlflow.start_run(run_name="CNN_Binary_Classifier"):

    df_combined = pd.read_csv('/opt/airflow/dags/combined_images.csv')

    train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=(224, 224),
        batch_size=64,
        class_mode='binary'
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=(224, 224),
        batch_size=64,
        class_mode='binary'
    )

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    initial_learning_rate = 1e-4
    batch_size = 64
    epochs = 100

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-6)

    # Log parameters to MLflow
    mlflow.log_param("initial_learning_rate", initial_learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("model_type", "Simple CNN")
    mlflow.tensorflow.autolog()  # Enable automatic logging of metrics and model

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stop, lr_schedule]
    )

    # After training, manually log additional metrics (best val accuracy and val loss)
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    mlflow.log_metric("best_val_accuracy", best_val_acc)
    mlflow.log_metric("best_val_loss", best_val_loss)

    # Save model manually (optional since autolog saves the model)
    model_path = "/tmp/model"
    model.save(model_path)
    mlflow.tensorflow.log_model(tf_saved_model_dir=model_path, artifact_path="model")

    # Plot and save accuracy/loss figure, then log as artifact
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    plot_path = "/tmp/training_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
