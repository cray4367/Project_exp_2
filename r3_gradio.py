import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import datetime
import os
from tensorflow.keras.layers import LeakyReLU
import gradio as gr

# logging
logging.basicConfig(filename='model_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_dataset(file_path):
    if not os.path.exists(file_path):
        return None, None

    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    logging.info(f"Loaded dataset: {file_path}")
    return X, y

def load_test_dataset(file_path, scaler):
    if not os.path.exists(file_path):
        return None, None

    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_scaled = scaler.transform(X)
    return X_scaled, y


def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(128, kernel_initializer='he_normal', activation='swish'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(0.2),

        layers.Dense(64, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(0.2),

        layers.Dense(32, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.01),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name="auc")]
    )
    return model


def conformity_score(predictions, true_labels):
    return 1 - np.abs(predictions.flatten() - true_labels)


def save_report(accuracy, precision, recall, f1, auc, sensitivity, conformity_scores):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "Sensitivity"],
        "Value": [accuracy, precision, recall, f1, auc, sensitivity]
    }).to_csv(f"model_report_{timestamp}.csv", index=False)

    pd.DataFrame({"Conformity Score": conformity_scores}).to_csv(f"conformity_scores_{timestamp}.csv", index=False)


def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Model Loss")
    plt.savefig("plots/training_history.png")
    plt.close()


def plot_conformity_distribution(conformity_scores):
    plt.figure(figsize=(8, 6))
    sns.histplot(conformity_scores, bins=50, kde=True, color='blue')
    plt.title("Conformity Score Distribution")
    plt.savefig("plots/conformity_distribution.png")
    plt.close()


def plot_roc_curve(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("plots/roc_curve.png")
    plt.close()


def train_and_evaluate(train_path, test_path):
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    X_train, y_train = load_dataset(train_path)
    if X_train is None:
        print(f"Error: Training file not found at {train_path}")
        return None, None

    X_test, y_test = load_dataset(test_path)
    if X_test is None:
        print(f"Error: Test file not found at {test_path}")
        return None, None

    X_train_scaled, scaler = preprocess_data(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=50,
        validation_split=0.2,
        verbose=1
    )

    y_pred = model.predict(X_test_scaled).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    conformity_scores = conformity_score(y_pred, y_test)
    save_report(
        accuracy_score(y_test, y_pred_binary),
        precision_score(y_test, y_pred_binary),
        recall_score(y_test, y_pred_binary),
        f1_score(y_test, y_pred_binary),
        roc_auc_score(y_test, y_pred),
        recall_score(y_test, y_pred_binary),
        conformity_scores
    )

    plot_training_history(history)
    plot_conformity_distribution(conformity_scores)
    plot_roc_curve(y_test, y_pred)

    return model, scaler


def gradio_interface(model, scaler):
    def predict(input_str):
        try:
            features = np.array([list(map(float, input_str.split(',')))])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled).flatten()[0]
            conformity = max(prediction, 1 - prediction)
            result = "False Data" if prediction > 0.5 else "Legit Data"
            return f"Result: {result} | Confidence: {prediction:.4f} | Conformity: {conformity:.4f}"
        except Exception as e:
            return f"Error: {str(e)}"

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=1, placeholder="Comma-separated feature values"),
        outputs="text",
        title="Live Data Detection",
        description="Enter comma-separated feature values to classify as Legit or False data."
    )
    demo.launch()


if __name__ == "__main__":
    train_path = 'train.csv'
    test_path = 'test.csv'

    model, scaler = train_and_evaluate(train_path, test_path)

    if model and scaler:
        gradio_interface(model, scaler)
