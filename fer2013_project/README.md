# FER2013 Facial Emotion Recognition Project

A comprehensive facial emotion recognition system using three different deep learning models on the FER2013 dataset.

## Project Structure

```
fer2013_project/
├── data/
│   └── fer2013.csv                    # FER2013 dataset
├── mini_xception/                      # Mini-XCEPTION model (59.9% accuracy)
│   ├── train.py                       # Training script
│   ├── live.py                        # Live detection script
│   ├── best.h5                        # Best model checkpoint
│   ├── last.h5                        # Last model checkpoint
│   └── results/                       # Evaluation outputs
├── mobilenetv2/                       # MobileNetV2 model (60.5% accuracy)
│   ├── train.py                       # Training script
│   ├── live.py                        # Live detection script
│   ├── best.h5                        # Best model checkpoint
│   ├── last.h5                        # Last model checkpoint
│   └── results/                       # Evaluation outputs
├── efficientnetb0/                     # EfficientNetB0 model (to be trained)
│   ├── train.py                       # Training script
│   ├── live.py                        # Live detection script
│   ├── best.h5                        # Best model checkpoint
│   ├── last.h5                        # Last model checkpoint
│   └── results/                       # Evaluation outputs
├── comparisons/                        # Cross-model analysis
│   ├── aggregate_metrics.csv          # Model comparison metrics
│   ├── violin_plots.png              # Performance comparison plots
│   └── notes.md                       # Benchmark notes
├── utils.py                           # Shared utilities
├── requirements.txt                   # Dependencies
└── README.md                         # This file
```

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to explain model predictions by highlighting the most important regions in the input image. This helps understand what facial features (eyes, eyebrows, mouth) the models focus on for emotion recognition.

### Features:
- **Live Grad-CAM**: Press 'g' during live detection to generate Grad-CAM visualizations
- **Batch Processing**: Use `gradcam_make.py` to generate Grad-CAM for saved images
- **Model Support**: Works with all three models (Mini-XCEPTION, MobileNetV2, EfficientNetB0)
- **Output**: Saves heatmap overlays to `results/gradcam_live/` directory

### Usage:
```bash
# Generate Grad-CAM for a specific model and image
python3 gradcam_make.py --model mini_xception --image path/to/image.jpg
python3 gradcam_make.py --model mobilenetv2 --image path/to/image.jpg
python3 gradcam_make.py --model efficientnetb0 --image path/to/image.jpg

# Override default last conv layer
python3 gradcam_make.py --model efficientnetb0 --image test.jpg --last_conv block7a_expand_conv
```

### Technical Details:
- **Mini-XCEPTION**: Uses `conv2d_2` layer (64x64 input)
- **MobileNetV2**: Uses `block_16_project_BN` layer (160x160 input)  
- **EfficientNetB0**: Uses `block6a_expand_conv` layer (224x224 input)
- **Reference**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2016)

## Models

### 1. Mini-XCEPTION
- **Architecture**: Custom CNN with XCEPTION-inspired blocks
- **Accuracy**: 59.9%
- **Training**: 102 epochs
- **Input**: 64x64x1 grayscale images
- **Status**: Complete

### 2. MobileNetV2
- **Architecture**: Transfer learning with MobileNetV2 base
- **Accuracy**: 60.5%
- **Training**: 15 epochs
- **Input**: 224x224x3 RGB images
- **Status**: Complete

### 3. EfficientNetB0
- **Architecture**: Transfer learning with EfficientNetB0 base
- **Accuracy**: TBD
- **Training**: 20 epochs (planned)
- **Input**: 224x224x3 RGB images
- **Status**: Ready for training

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fer2013_project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download FER2013 dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
   - Place `fer2013.csv` in the `data/` folder

## Usage

### Training Models

#### Mini-XCEPTION
```bash
cd mini_xception
python train.py --epochs 102 --batch_size 32
```

#### MobileNetV2
```bash
cd mobilenetv2
python train.py --epochs 15 --batch_size 32
```

#### EfficientNetB0
```bash
cd efficientnetb0
python train.py --epochs 20 --batch_size 32
```

### Live Emotion Detection

#### Mini-XCEPTION
```bash
cd mini_xception
python live.py
```

#### MobileNetV2
```bash
cd mobilenetv2
python live.py
```

#### EfficientNetB0
```bash
cd efficientnetb0
python live.py
```

### Model Comparison

```bash
python utils.py
```

This will generate comparison metrics and plots in the `comparisons/` folder.

## Features

### Training Features
- **Data Augmentation**: Rotation, shift, shear, zoom, horizontal flip
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Model Checkpointing**: Saves best and last models
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrices

### Live Detection Features
- **Real-time Processing**: 30 FPS emotion detection
- **Face Detection**: Haar cascade integration
- **1-minute Monitoring**: Continuous emotion tracking
- **Personalized Recommendations**: Content suggestions based on emotions
- **Session Logging**: Detailed analytics and session data
- **Beautiful GUI**: Pygame interface with live camera feed

### Personalized Content
- **Videos**: YouTube links for different emotions
- **Music**: Curated playlists for emotional states
- **Activities**: Suggested actions based on detected emotions
- **Real-time Analysis**: 1-minute emotion monitoring with recommendations

## Model Performance

| Model | Accuracy | Macro F1 | Training Time | Inference Speed |
|-------|----------|----------|---------------|-----------------|
| **MobileNetV2** | **60.5%** | **TBD** | **Fast** | **Fast** |
| **Mini-XCEPTION** | **59.9%** | **TBD** | **Medium** | **Medium** |
| **EfficientNetB0** | **TBD** | **TBD** | **TBD** | **TBD** |

## File Descriptions

### Training Scripts (`train.py`)
- Load and preprocess FER2013 data
- Create and train model architecture
- Implement data augmentation
- Save model checkpoints
- Generate evaluation metrics and plots

### Live Detection Scripts (`live.py`)
- Real-time camera processing
- Face detection and emotion recognition
- 1-minute emotion monitoring
- Personalized content recommendations
- Session data logging
- Beautiful GUI interface

### Utilities (`utils.py`)
- Data loading and preprocessing
- Metrics saving and plotting
- Personalized content database
- Model comparison functions

## Dependencies

- **TensorFlow**: Deep learning framework
- **Keras**: High-level neural network API
- **OpenCV**: Computer vision and camera processing
- **Pygame**: GUI and live interface
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Plotting and visualization
- **Scikit-learn**: Machine learning utilities

## Notes

- All models use the same FER2013 dataset
- Models are trained with different architectures and strategies
- Live detection includes personalized recommendations
- Session data is automatically saved for analysis
- GUI provides real-time feedback and interaction

## Future Improvements

- Add more model architectures
- Implement ensemble methods
- Improve data augmentation strategies
- Add more personalized content
- Enhance GUI features
- Add model interpretability tools

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Contact

For questions or support, please open an issue in the repository.
