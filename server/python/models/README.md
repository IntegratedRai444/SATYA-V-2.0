# SatyaAI ML Models Directory

## Overview

This directory contains the machine learning models and utilities for the SatyaAI deepfake detection platform.

## Model Architecture

### Core Models
- **Image Models**: EfficientNet-B7, Xception, ResNet50
- **Video Models**: 3D CNN + LSTM, Temporal ConvNets
- **Audio Models**: Wav2Vec2, ECAPA-TDNN
- **Text Models**: RoBERTa, BERT
- **Multimodal**: Fusion models for cross-modal analysis

### Model Files

#### EfficientNet-B7
- **Location**: `../models/dfdc_efficientnet_b7/model.pth`
- **Size**: 91.7 MB
- **Status**: ✅ FIXED - Real PyTorch model
- **Purpose**: Image deepfake detection
- **Input**: 224x224 RGB images
- **Output**: Binary classification (real/fake)

#### Xception
- **Location**: `../models/xception/model.pth`
- **Size**: 91.7 MB
- **Status**: ✅ WORKING
- **Purpose**: Image deepfake detection
- **Architecture**: FaceForensics++ compatible

#### Video Models
- **Location**: `../models/video_deepfake_detector.pth`
- **Size**: 33.6 MB
- **Status**: ✅ WORKING
- **Purpose**: Video deepfake detection

## Model Management

### Loading Models
Models are loaded through the `DeepfakeClassifier` class in `deepfake_classifier.py`:

```python
from models.deepfake_classifier import DeepfakeClassifier

# Initialize classifier
classifier = DeepfakeClassifier(model_type='efficientnet', device='cpu')

# Make prediction
result = classifier.predict_image(image)
```

### Available Model Types
- `efficientnet`: EfficientNet-B7 with HuggingFace fallbacks
- `xception`: Xception with FaceForensics++ compatibility
- `vit`: Vision Transformer
- `resnet50`: ResNet-50
- `swin`: Swin Transformer

### Fallback Mechanism
The system has multiple fallback layers:
1. Local model files
2. HuggingFace models
3. torchvision pretrained models

## Model Generation

### Creating New Models
Use the `EfficientNetModelGenerator` class:

```python
from models.efficientnet_model_generator import EfficientNetModelGenerator

generator = EfficientNetModelGenerator()
model_path = generator.generate_and_save('imagenet_pretrained')
```

### Available Configurations
- `imagenet_pretrained`: ImageNet weights with custom classifier
- `deepfake_optimized`: Optimized for deepfake detection
- `lightweight`: Faster inference with smaller classifier

## Model Verification

### Testing Models
Run the verification script:

```bash
cd server/python
python models/efficientnet_fix_verification.py
```

This tests:
- File integrity
- Model loading
- Performance with multiple inputs
- Robustness with edge cases
- Model metadata

### Model Reports
Generate comprehensive model reports:

```python
from models.efficientnet_fix_verification import generate_model_report

report = generate_model_report()
print(report)
```

## Security & Integrity

### Model Validation
- All models are validated before loading
- File integrity checks prevent corrupted models
- Model metadata is verified for compatibility

### Cryptographic Proof
Models generate cryptographic proofs for analysis results:
- SHA-256 hashes of model weights
- Timestamp verification
- Result authenticity verification

## Performance Optimization

### Model Optimization
- Dynamic quantization support
- Mixed precision training (FP16)
- Model pruning capabilities
- Mobile optimization

### Caching
- Model weights caching
- Inference result caching
- Feature extraction caching

## Troubleshooting

### Common Issues

#### Model Loading Failures
1. Check file integrity
2. Verify PyTorch version compatibility
3. Ensure sufficient memory
4. Check CUDA availability

#### Performance Issues
1. Enable GPU acceleration
2. Use quantized models
3. Optimize batch sizes
4. Check memory usage

#### Accuracy Issues
1. Verify model training data
2. Check preprocessing pipeline
3. Validate input format
4. Review model configuration

### Debug Mode
Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development

### Adding New Models
1. Create model class in appropriate module
2. Add to `deepfake_classifier.py`
3. Update model configurations
4. Add tests
5. Update documentation

### Model Training
See training scripts:
- `training/train_image_model.py`
- `training/train_video_model.py`
- `training/train_audio_model.py`

## Production Deployment

### Model Loading Strategy
1. Load models at startup
2. Use singleton pattern
3. Implement health checks
4. Monitor memory usage

### Monitoring
- Model performance metrics
- Inference latency tracking
- Error rate monitoring
- Resource usage tracking

## API Integration

### Model Endpoints
Models are exposed through FastAPI endpoints:
- `/api/v2/analysis/image` - Image analysis
- `/api/v2/analysis/video` - Video analysis
- `/api/v2/analysis/audio` - Audio analysis
- `/api/v2/analysis/text` - Text analysis

### Request Format
```json
{
  "file": "base64_encoded_data",
  "model_type": "efficientnet",
  "options": {
    "confidence_threshold": 0.5,
    "return_features": true
  }
}
```

### Response Format
```json
{
  "prediction": "fake",
  "confidence": 0.85,
  "features": [...],
  "proof": {
    "model_hash": "sha256_hash",
    "timestamp": "2026-01-01T00:00:00Z"
  }
}
```

## Contributing

### Model Updates
1. Test new models thoroughly
2. Update documentation
3. Add regression tests
4. Update version numbers

### Code Standards
- Follow PEP 8
- Add type hints
- Include docstrings
- Add unit tests

## License

All models are licensed under the SatyaAI license. See LICENSE file for details.

## Support

For model-related issues:
1. Check troubleshooting guide
2. Review model logs
3. Run verification tests
4. Contact development team

---

**Last Updated**: January 2026
**Version**: 1.0.0
**Status**: Production Ready
