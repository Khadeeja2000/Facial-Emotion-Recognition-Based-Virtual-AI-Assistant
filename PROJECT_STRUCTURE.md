# Project Structure Documentation

## Overview
This document describes the complete structure of the Emotion Recognition System project, including all files, directories, and their purposes.

## Directory Structure

```
emotion-recognition/
├── .github/                          # GitHub-specific files
│   └── workflows/                    # GitHub Actions workflows
│       └── python-app.yml           # CI/CD pipeline configuration
├── models/                           # Neural network models and architectures
│   ├── _mini_XCEPTION.102-0.66.hdf5 # Pre-trained emotion recognition model
│   └── cnn.py                       # CNN architecture definitions
├── haarcascade_files/               # OpenCV face detection models
│   ├── haarcascade_frontalface_default.xml  # Front face detection
│   └── haarcascade_eye.xml         # Eye detection (for future use)
├── fer2013/                         # Dataset directory
│   └── fer2013/                     # FER2013 emotion dataset
│       └── fer2013.csv             # Main dataset file (not in repo)
├── emotions/                        # Sample emotion images (for documentation)
│   ├── angry.PNG
│   ├── disgust.PNG
│   ├── Happy.PNG
│   └── [other emotion images]
├── docs/                            # Documentation (auto-generated)
│   └── site/                        # Built documentation site
├── tests/                           # Test files
│   └── test_emotion_recognition.py  # Unit tests
├── .gitignore                       # Git ignore patterns
├── LICENSE                          # MIT License
├── PROJECT_STRUCTURE.md             # This file
├── README.md                        # Main project documentation
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation script
├── real_time_video.py               # Main application
├── train_emotion_classifier.py      # Model training script
└── load_and_process.py              # Data processing utilities
```

## File Descriptions

### Core Application Files

#### `real_time_video.py`
- **Purpose**: Main application for real-time emotion recognition
- **Functionality**: 
  - Webcam input processing
  - Face detection using Haar Cascade
  - Emotion classification using pre-trained CNN
  - Real-time visualization of results
- **Key Features**:
  - Multi-face detection support
  - Emotion probability visualization
  - Clean, modular code structure
  - Error handling and graceful shutdown

#### `train_emotion_classifier.py`
- **Purpose**: Training pipeline for custom emotion recognition models
- **Functionality**:
  - Data loading and preprocessing
  - Model creation and compilation
  - Training with callbacks and monitoring
  - Model checkpointing and saving
- **Key Features**:
  - Data augmentation
  - Early stopping and learning rate reduction
  - Comprehensive logging
  - Validation split handling

#### `load_and_process.py`
- **Purpose**: Data loading and preprocessing utilities
- **Functionality**:
  - FER2013 dataset loading
  - Image preprocessing and normalization
  - Data validation and error handling
- **Key Features**:
  - Robust error handling
  - Multiple preprocessing options
  - Dataset information utilities
  - Memory-efficient processing

### Model Files

#### `models/cnn.py`
- **Purpose**: Neural network architecture definitions
- **Architectures**:
  - `simple_CNN`: Basic convolutional network
  - `simpler_CNN`: Optimized version
  - `tiny_XCEPTION`: Lightweight XCEPTION variant
  - `mini_XCEPTION`: Balanced XCEPTION variant
  - `big_XCEPTION`: Full XCEPTION architecture
- **Key Features**:
  - Modular design
  - Configurable parameters
  - Batch normalization
  - Dropout for regularization

#### `models/_mini_XCEPTION.102-0.66.hdf5`
- **Purpose**: Pre-trained emotion recognition model
- **Specifications**:
  - Architecture: mini_XCEPTION
  - Accuracy: 66% on FER2013 test set
  - File size: ~102MB
  - Input shape: (48, 48, 1) grayscale images
  - Output: 7 emotion classes

### Configuration Files

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Categories**:
  - Core dependencies (OpenCV, NumPy, Pandas)
  - Deep learning frameworks (TensorFlow, Keras)
  - Development tools (pytest, flake8, black)
  - Optional GPU support

#### `setup.py`
- **Purpose**: Package installation and distribution
- **Features**:
  - Package metadata
  - Dependency management
  - Console script entry points
  - Development dependencies
  - License and author information

#### `.gitignore`
- **Purpose**: Git ignore patterns
- **Categories**:
  - Python-specific files
  - Model files (large binary files)
  - Dataset files
  - IDE and OS files
  - Temporary and cache files

### Documentation Files

#### `README.md`
- **Purpose**: Main project documentation
- **Sections**:
  - Project overview and features
  - Installation instructions
  - Usage examples
  - Project structure
  - Troubleshooting guide
  - Contributing guidelines

#### `PROJECT_STRUCTURE.md`
- **Purpose**: Detailed project structure documentation
- **Content**: This file, explaining all components

### GitHub Configuration

#### `.github/workflows/python-app.yml`
- **Purpose**: Automated CI/CD pipeline
- **Features**:
  - Multi-Python version testing
  - Code quality checks (flake8)
  - Automated testing (pytest)
  - Package building
  - Documentation deployment

### Test Files

#### `test_emotion_recognition.py`
- **Purpose**: Unit tests for core functionality
- **Test Categories**:
  - Data preprocessing validation
  - Emotion label verification
  - Input validation
  - Error handling

## Data Flow

```
Webcam Input → Face Detection → Emotion Classification → Visualization
     ↓              ↓                    ↓                ↓
Frame Capture → Haar Cascade → CNN Model → Probability Bars
     ↓              ↓                    ↓                ↓
Video Stream → Face ROI → Preprocessing → Real-time Display
```

## Development Workflow

1. **Code Development**: Write/modify Python files
2. **Testing**: Run unit tests with `python -m pytest`
3. **Code Quality**: Check with `flake8` and `black`
4. **Documentation**: Update README and docstrings
5. **Commit**: Git commit with descriptive messages
6. **Push**: Push to GitHub triggers automated testing
7. **Review**: Code review and merge to main branch

## Deployment

### Local Development
```bash
# Clone repository
git clone <repository-url>
cd emotion-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python real_time_video.py
```

### Production Deployment
```bash
# Install package
pip install -e .

# Run as command-line tool
emotion-recognition
```

