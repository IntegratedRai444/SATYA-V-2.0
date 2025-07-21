import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# XceptionNet implementation (public, simplified for deepfake detection)
# Source: https://github.com/ondyari/FaceForensics/blob/master/classification/network/models.py
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=stride, bias=False) if (out_filters != in_filters or stride != 1) else None
        self.skipbn = nn.BatchNorm2d(out_filters) if self.skip is not None else None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, 1))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, 1, 1))
            rep.append(nn.BatchNorm2d(out_filters))
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)
    def forward(self, inp):
        x = self.rep(inp)
        skip = self.skip(inp) if self.skip is not None else inp
        if self.skipbn is not None:
            skip = self.skipbn(skip)
        x += skip
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)
    def features(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x
    def forward(self, input):
        x = self.features(input)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Lazy model loading for faster startup
_model = None
_preprocess = None

def get_model():
    """Lazy load model only when needed"""
    global _model, _preprocess
    
    if _model is None:
        print("🤖 Loading XceptionNet model...")
        MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xception_deepfake.pth')
        _model = Xception(num_classes=1)
        if os.path.exists(MODEL_PATH):
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            print("✅ Model weights loaded")
        else:
            print('⚠️ Warning: XceptionNet weights not found. Using random weights!')
        _model.eval()
        
        # Preprocessing pipeline for XceptionNet
        _preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        print("✅ Model ready for inference")
    
    return _model, _preprocess

def predict_deepfake(arr: np.ndarray, faces=None, gan_artifacts=None):
    """
    Takes a preprocessed image (numpy array), optional face crops and artifact info,
    and returns (label, confidence, explanation).
    """
    # Lazy load model
    model, preprocess = get_model()
    
    # Convert arr to torch tensor
    if arr.max() <= 1.0:
        arr = (arr * 255).astype('uint8')
    input_tensor = preprocess(arr)
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        label = 'FAKE' if prob > 0.5 else 'REAL'
        confidence = prob * 100
        explanation = [f'XceptionNet output: {prob:.2f}']
    return label, confidence, explanation 