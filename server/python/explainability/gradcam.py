#!/usr/bin/env python3
"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Provides explainability for deepfake detection predictions
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for CNN model explainability
    """

    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM

        Args:
            model: PyTorch model
            target_layer: Target layer for activation maps
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self, input_image: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map

        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class index (None for predicted class)

        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize(
        self, original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay CAM on original image

        Args:
            original_image: Original image (H, W, 3)
            cam: CAM heatmap (h, w)
            alpha: Overlay transparency

        Returns:
            Visualization image
        """
        # Resize CAM to match image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        visualization = heatmap * alpha + original_image * (1 - alpha)
        visualization = np.uint8(visualization)

        return visualization


class DeepfakeExplainer:
    """
    High-level explainability interface for deepfake detection
    """

    def __init__(self, model, device="cpu"):
        """
        Initialize explainer

        Args:
            model: Trained deepfake detection model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device

        # Get target layer (last conv layer)
        self.target_layer = self._get_target_layer()

        # Initialize Grad-CAM
        self.gradcam = GradCAM(self.model, self.target_layer)

    def _get_target_layer(self):
        """Get the last convolutional layer"""
        # For ResNet50
        if hasattr(self.model, "backbone"):
            if hasattr(self.model.backbone, "layer4"):
                return self.model.backbone.layer4[-1]

        # Fallback: find last Conv2d layer
        for module in reversed(list(self.model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module

        raise ValueError("Could not find target layer")

    def explain_prediction(
        self, image_path: str, save_path: Optional[str] = None
    ) -> dict:
        """
        Generate explanation for a prediction

        Args:
            image_path: Path to input image
            save_path: Path to save visualization

        Returns:
            Dictionary with prediction and explanation
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Preprocess for model
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.item()
            is_fake = prediction > 0.5
            confidence = prediction if is_fake else 1 - prediction

        # Generate CAM
        cam = self.gradcam.generate_cam(input_tensor)

        # Create visualization
        image_resized = cv2.resize(image_np, (224, 224))
        visualization = self.gradcam.visualize(image_resized, cam)

        # Create explanation plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image_resized)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Attention Heatmap", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(visualization)
        label = "FAKE" if is_fake else "REAL"
        color = "red" if is_fake else "green"
        axes[2].set_title(
            f"Prediction: {label} ({confidence*100:.1f}%)",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Explanation saved to: {save_path}")

        plt.close()

        return {
            "prediction": "fake" if is_fake else "real",
            "confidence": float(confidence),
            "raw_score": float(prediction),
            "attention_map": cam,
            "visualization": visualization,
        }

    def batch_explain(self, image_paths: list, output_dir: str):
        """
        Generate explanations for multiple images

        Args:
            image_paths: List of image paths
            output_dir: Directory to save visualizations
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path}")

            save_path = os.path.join(output_dir, f"explanation_{i+1}.png")
            result = self.explain_prediction(image_path, save_path)
            results.append(result)

        print(f"\n✅ Generated {len(results)} explanations in {output_dir}")
        return results


def demo_gradcam():
    """Demo Grad-CAM functionality"""
    print("\n" + "=" * 60)
    print("GRAD-CAM EXPLAINABILITY DEMO")
    print("=" * 60 + "\n")

    print("Grad-CAM provides visual explanations for model predictions")
    print("by highlighting the regions that influenced the decision.\n")

    print("Features:")
    print("  ✅ Visual attention maps")
    print("  ✅ Heatmap overlays")
    print("  ✅ Confidence scores")
    print("  ✅ Batch processing")
    print("  ✅ High-resolution outputs\n")

    print("Usage:")
    print("  from explainability.gradcam import DeepfakeExplainer")
    print("  explainer = DeepfakeExplainer(model)")
    print("  result = explainer.explain_prediction('image.jpg', 'output.png')")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    demo_gradcam()
