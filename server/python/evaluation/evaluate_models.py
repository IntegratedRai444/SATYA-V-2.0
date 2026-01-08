"""
Model Evaluation Script
Evaluates trained models and generates performance metrics
"""

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

# Model paths
IMAGE_MODEL_PATH = "../models/image_detector.h5"
VIDEO_MODEL_PATH = "../models/video_detector.h5"
AUDIO_MODEL_PATH = "../models/audio_detector.h5"

# Test data paths
IMAGE_TEST_PATH = "../data/faceforensics/test"
VIDEO_TEST_PATH = "../data/dfdc/test"
AUDIO_TEST_PATH = "../data/asvspoof/test"


def evaluate_model(model_path, test_generator, model_name):
    """
    Evaluate a single model and return metrics
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name} Model")
    print(f"{'='*50}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Get predictions
    print("Generating predictions...")
    y_true = []
    y_pred_proba = []

    for i in range(len(test_generator)):
        X_batch, y_batch = test_generator[i]
        predictions = model.predict(X_batch, verbose=0)

        y_true.extend(y_batch)
        y_pred_proba.extend(predictions.flatten())

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print results
    print(f"\n{model_name} Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC-ROC:   {auc_roc:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    # Return metrics
    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc_roc),
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred_proba": y_pred_proba.tolist(),
    }


def plot_confusion_matrix(cm, model_name, save_path):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_pred_proba, model_name, save_path):
    """
    Plot and save ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_comparison(all_metrics, save_path):
    """
    Plot comparison of all models
    """
    models = [m["model_name"] for m in all_metrics]
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.15

    for i, metric in enumerate(metrics):
        values = [m[metric] for m in all_metrics]
        ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())

    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0.8, 1.0])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def main():
    """
    Main evaluation function
    """
    print("=" * 50)
    print("Model Evaluation Suite")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../evaluation_results/{timestamp}"

    # Create results directory
    import os

    os.makedirs(results_dir, exist_ok=True)

    all_metrics = []

    # Evaluate Image Model
    try:
        from train_image_model import \
            create_data_generators as create_image_generators

        _, _, image_test_gen = create_image_generators()

        image_metrics = evaluate_model(IMAGE_MODEL_PATH, image_test_gen, "Image")
        all_metrics.append(image_metrics)

        # Plot confusion matrix
        plot_confusion_matrix(
            np.array(image_metrics["confusion_matrix"]),
            "Image Model",
            f"{results_dir}/image_confusion_matrix.png",
        )

        # Plot ROC curve
        plot_roc_curve(
            image_metrics["y_true"],
            image_metrics["y_pred_proba"],
            "Image Model",
            f"{results_dir}/image_roc_curve.png",
        )
    except Exception as e:
        print(f"Error evaluating image model: {e}")

    # Evaluate Video Model
    try:
        from train_video_model import VideoDataGenerator

        video_test_gen = VideoDataGenerator(VIDEO_TEST_PATH, shuffle=False)

        video_metrics = evaluate_model(VIDEO_MODEL_PATH, video_test_gen, "Video")
        all_metrics.append(video_metrics)

        plot_confusion_matrix(
            np.array(video_metrics["confusion_matrix"]),
            "Video Model",
            f"{results_dir}/video_confusion_matrix.png",
        )

        plot_roc_curve(
            video_metrics["y_true"],
            video_metrics["y_pred_proba"],
            "Video Model",
            f"{results_dir}/video_roc_curve.png",
        )
    except Exception as e:
        print(f"Error evaluating video model: {e}")

    # Evaluate Audio Model
    try:
        from train_audio_model import AudioDataGenerator

        audio_test_gen = AudioDataGenerator(AUDIO_TEST_PATH, shuffle=False)

        audio_metrics = evaluate_model(AUDIO_MODEL_PATH, audio_test_gen, "Audio")
        all_metrics.append(audio_metrics)

        plot_confusion_matrix(
            np.array(audio_metrics["confusion_matrix"]),
            "Audio Model",
            f"{results_dir}/audio_confusion_matrix.png",
        )

        plot_roc_curve(
            audio_metrics["y_true"],
            audio_metrics["y_pred_proba"],
            "Audio Model",
            f"{results_dir}/audio_roc_curve.png",
        )
    except Exception as e:
        print(f"Error evaluating audio model: {e}")

    # Plot comparison
    if len(all_metrics) > 1:
        plot_comparison(all_metrics, f"{results_dir}/model_comparison.png")

    # Save metrics to JSON
    metrics_file = f"{results_dir}/evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    for metrics in all_metrics:
        print(f"\n{metrics['model_name']} Model:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    print(f"\nAll results saved to: {results_dir}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
