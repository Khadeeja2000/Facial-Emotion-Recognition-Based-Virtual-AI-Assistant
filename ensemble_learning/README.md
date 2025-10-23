# Ensemble Learning for Emotion Recognition

## Overview

This ensemble learning system combines three state-of-the-art CNN architectures for robust facial emotion recognition:

- **Mini-XCEPTION**: Custom CNN architecture with 68.6% accuracy
- **MobileNetV2**: Transfer learning approach with 60.7% accuracy  
- **EfficientNetB0**: State-of-the-art architecture with 61.7% accuracy

The ensemble uses weighted voting to combine the strengths of each model, targeting **72-75% accuracy** in real-world scenarios.

## Architecture

### Ensemble Strategy
- **Weighted Soft Voting**: Combines probability outputs from all models
- **Dynamic Weight Assignment**: 
  - Mini-XCEPTION: 50% (highest individual accuracy)
  - MobileNetV2: 25% (transfer learning benefits)
  - EfficientNetB0: 25% (state-of-the-art architecture)

### Model Integration
```
Input Image (48x48x1) → Face Detection → Preprocessing
    ↓
┌─────────────────────────────────────────────────────┐
│  Mini-XCEPTION    MobileNetV2    EfficientNetB0    │
│  (Grayscale)      (RGB)          (RGB)             │
│  Weight: 0.5      Weight: 0.25   Weight: 0.25     │
└─────────────────────────────────────────────────────┘
    ↓
Ensemble Prediction (Weighted Average)
    ↓
Final Emotion Classification (7 classes)
```

## Features

### Core Capabilities
- **Multi-Model Ensemble**: Combines 3 different CNN architectures
- **Real-Time Inference**: Optimized for live camera processing
- **Comprehensive Evaluation**: Detailed performance metrics and analysis
- **Real-World Testing**: Camera-based validation with performance tracking

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro, Micro, and Weighted F1 scores
- **Precision & Recall**: Per-class and macro averages
- **ROC-AUC**: One-vs-rest multiclass ROC analysis
- **Confusion Matrix**: Detailed classification breakdown

### Visualization & Analysis
- **Performance Comparison**: Side-by-side model comparison
- **Per-Class Analysis**: Emotion-specific performance metrics
- **Improvement Analysis**: Ensemble vs individual model gains
- **Real-World Testing**: Live camera performance tracking
- **Radar Charts**: Multi-dimensional performance visualization

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.6+
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Clone or navigate to the ensemble_learning directory
cd ensemble_learning

# Install requirements
pip install -r requirements.txt

# Verify FER2013 dataset is available
ls ../fer2013_project/data/fer2013.csv
```

## Usage

### Quick Start - Complete Pipeline
```bash
# Run the complete ensemble learning pipeline
python ensemble_pipeline.py
```

This will execute all steps:
1. Train individual models
2. Comprehensive evaluation
3. Model comparison analysis
4. Real-world testing

### Individual Components

#### 1. Train Ensemble Models
```bash
python ensemble_trainer.py
```
- Trains Mini-XCEPTION, MobileNetV2, and EfficientNetB0
- Creates ensemble with weighted voting
- Saves trained models and weights

#### 2. Comprehensive Evaluation
```bash
python ensemble_evaluator.py
```
- Evaluates all models on FER2013 test set
- Calculates detailed performance metrics
- Creates comparison visualizations
- Generates comprehensive reports

#### 3. Model Comparison
```bash
python ensemble_comparison.py
```
- Compares ensemble vs individual models
- Analyzes improvement contributions
- Creates detailed comparison visualizations
- Generates comparison reports

#### 4. Real-World Testing
```bash
python real_world_testing.py
```
- Tests ensemble with live camera feed
- Tracks performance metrics in real-time
- Analyzes multiple test sessions
- Generates real-world performance reports

## Expected Results

### Performance Targets
- **Ensemble Accuracy**: 72-75% (vs 68.6% best individual)
- **Real-Time Inference**: ≥25 FPS
- **Robustness**: Consistent performance across lighting conditions
- **Generalization**: Better handling of edge cases

### Generated Outputs

#### Models & Weights
- `mini_xception_ensemble.h5`
- `mobilenetv2_ensemble.h5`
- `efficientnetb0_ensemble.h5`
- `ensemble_weights.json`

#### Evaluation Results
- `evaluation_results.json`
- `comprehensive_evaluation_report.txt`

#### Visualizations
- `comprehensive_metrics_comparison.png`
- `per_class_performance_comparison.png`
- `improvement_analysis.png`
- `confusion_matrix_comparison.png`
- `performance_radar_chart.png`
- `ensemble_contribution_analysis.png`

#### Real-World Testing
- `real_world_testing_report.txt`
- `live_sessions/` (session data and analysis)

## File Structure

```
ensemble_learning/
├── ensemble_trainer.py          # Train ensemble models
├── ensemble_evaluator.py        # Comprehensive evaluation
├── ensemble_comparison.py       # Model comparison analysis
├── real_world_testing.py        # Live camera testing
├── ensemble_pipeline.py         # Complete pipeline orchestration
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── results/                     # Generated outputs
    ├── *.h5                     # Trained models
    ├── *.json                   # Results and weights
    ├── *.txt                    # Reports
    ├── *.png                    # Visualizations
    └── live_sessions/           # Real-world test data
```

## Technical Details

### Model Architectures

#### Mini-XCEPTION
- **Input**: 48x48x1 grayscale images
- **Architecture**: Custom CNN with separable convolutions
- **Training**: 30 epochs with data augmentation
- **Performance**: 68.6% accuracy, 30 FPS

#### MobileNetV2 (Transfer Learning)
- **Input**: 48x48x3 RGB images (converted from grayscale)
- **Base**: Pre-trained ImageNet weights
- **Training**: Frozen base + custom head (10 epochs)
- **Performance**: 60.7% accuracy, 25 FPS

#### EfficientNetB0 (Transfer Learning)
- **Input**: 48x48x3 RGB images (converted from grayscale)
- **Base**: Pre-trained ImageNet weights
- **Training**: Frozen base + custom head (10 epochs)
- **Performance**: 61.7% accuracy, 20 FPS

### Ensemble Methodology
- **Voting Strategy**: Weighted soft voting
- **Weight Optimization**: Based on individual model performance
- **Prediction Fusion**: Probability-weighted averaging
- **Output**: 7-class emotion probabilities

### Data Processing
- **Face Detection**: Haar Cascade (OpenCV)
- **Preprocessing**: Normalization, resizing, augmentation
- **Input Preparation**: Grayscale (Mini-XCEPTION) vs RGB (Transfer Learning)
- **Batch Processing**: Optimized for real-time inference

## Performance Analysis

### Expected Improvements
- **Accuracy Gain**: 3-6% over best individual model
- **Robustness**: Better handling of difficult cases
- **Consistency**: More stable predictions across scenarios
- **Generalization**: Improved performance on unseen data

### Benchmarking
- **Dataset**: FER2013 PrivateTest set (3,589 samples)
- **Metrics**: Accuracy, F1-Score, Precision, Recall, ROC-AUC
- **Comparison**: Ensemble vs individual models
- **Validation**: Cross-validation and real-world testing

## Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU, models will run on CPU (slower)
```

#### 2. Memory Issues
```bash
# Reduce batch size in training scripts
# Or use gradient accumulation
```

#### 3. Dataset Issues
```bash
# Verify FER2013 dataset path
ls ../fer2013_project/data/fer2013.csv

# Check data format and size
```

#### 4. Model Loading Issues
```bash
# Ensure models are trained first
python ensemble_trainer.py

# Check model files exist
ls results/*.h5
```

### Performance Optimization
- **GPU Acceleration**: Use CUDA-compatible GPU
- **Batch Processing**: Optimize batch sizes for your hardware
- **Model Quantization**: Consider INT8 quantization for deployment
- **TensorRT**: Use TensorRT for production deployment

## Contributing

### Adding New Models
1. Create new model architecture in `ensemble_trainer.py`
2. Add model to ensemble weights dictionary
3. Update evaluation and comparison scripts
4. Test with real-world scenarios

### Extending Evaluation
1. Add new metrics in `ensemble_evaluator.py`
2. Create corresponding visualizations
3. Update comparison reports
4. Validate with additional datasets

## License

This ensemble learning system is part of the Emotion Recognition project. See the main project LICENSE for details.

## Citation

If you use this ensemble learning system in your research, please cite:

```bibtex
@software{ensemble_emotion_recognition,
  title={Ensemble Learning for Facial Emotion Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/emotion-recognition}
}
```

## Contact

For questions, issues, or contributions:
- Create an issue in the main project repository
- Contact: [your-email@domain.com]

---

**Note**: This ensemble learning system is designed for academic and research purposes. For production deployment, consider additional optimization, validation, and security measures.
