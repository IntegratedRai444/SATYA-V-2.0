import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import os

def download_model():
    # Create models directory if it doesn't exist
    os.makedirs("models/dfdc_efficientnet_b7", exist_ok=True)
    
    print("Downloading a lightweight image classification model...")
    
    try:
        # Download a lightweight model
        model_name = "microsoft/resnet-50"
        model = AutoModelForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Save the model
        model_path = "models/dfdc_efficientnet_b7/model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_extractor': feature_extractor
        }, model_path)
        
        print(f"Model saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    download_model()
