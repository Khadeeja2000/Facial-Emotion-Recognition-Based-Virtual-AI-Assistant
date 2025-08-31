# Emotion Recognition System

A comprehensive real-time emotion recognition platform that detects and classifies human emotions using computer vision and deep learning techniques, providing personalized content recommendations based on detected emotional states.

## Overview

This system provides real-time emotion detection through webcam input, capable of recognizing seven distinct emotions: angry, disgust, scared, happy, sad, surprised, and neutral. It features a modular architecture with pre-trained models, real-time processing, and comprehensive training capabilities. The system goes beyond simple emotion detection by offering personalized content recommendations tailored to the user's current emotional state.

## Features

- **Real-time Emotion Detection**: Live webcam-based emotion recognition
- **Multi-Face Support**: Detect and analyze emotions for multiple faces simultaneously
- **High Accuracy**: Pre-trained CNN model achieving 66% accuracy on FER2013 dataset
- **Modular Architecture**: Clean, maintainable code structure
- **Training Pipeline**: Complete training infrastructure for custom models
- **Data Augmentation**: Built-in data augmentation for improved model performance
- **Personalized Content Recommendations**: Intelligent content suggestions based on detected emotions
- **Emotion-Based User Experience**: Adaptive interface that responds to user emotional states

## System Requirements

- Python 3.8 or higher
- Webcam or camera device
- GPU recommended for training (optional for inference)
- 4GB RAM minimum, 8GB recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Khadeeja2000/Facial-Emotion-Recognition-Based-Virtual-AI-Assistant.git
cd Facial-Emotion-Recognition-Based-Virtual-AI-Assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models and datasets:
   - Place the pre-trained model in `models/` directory
   - Download FER2013 dataset and place in `fer2013/fer2013/` directory

## Usage

### Real-time Emotion Recognition

Run the main application:
```bash
python real_time_video.py
```

Controls:
- Press 'q' to quit the application
- The system will automatically detect faces and display emotions

### Training Custom Models

To train a new emotion classification model:
```bash
python train_emotion_classifier.py
```

Training parameters can be modified in the script:
- Batch size: 32
- Epochs: 10000
- Input shape: (48, 48, 1)
- Validation split: 20%

## Personalized Content Features

The system provides intelligent content recommendations based on detected emotions:

### **Emotion-Based Content Suggestions:**
- **Happy**: Uplifting content, motivational videos, social activities
- **Sad**: Comforting content, stress-relief exercises, positive affirmations
- **Angry**: Calming content, breathing exercises, conflict resolution tips
- **Stressed**: Relaxation techniques, meditation guides, stress management
- **Neutral**: Balanced content, learning opportunities, productivity tips
- **Surprised**: Educational content, new experiences, discovery-based activities
- **Disgust**: Health and wellness content, positive environment suggestions

### **Adaptive User Experience:**
- **Dynamic Interface**: UI elements change based on emotional state
- **Content Filtering**: Automatically filters content to match emotional needs
- **Recommendation Engine**: Learns user preferences over time
- **Emotional Well-being Tracking**: Monitors emotional patterns and trends

## Project Structure

```
emotion-recognition/
├── models/                          # Pre-trained models and CNN architectures
│   ├── _mini_XCEPTION.102-0.66.hdf5
│   └── cnn.py
├── haarcascade_files/              # Face detection models
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
├── fer2013/                        # Dataset directory
│   └── fer2013/
│       └── fer2013.csv
├── real_time_video.py              # Main application
├── train_emotion_classifier.py     # Training script
├── load_and_process.py             # Data processing utilities
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Model Architecture

The system uses a modified XCEPTION architecture optimized for emotion recognition:
- Input: 48x48 grayscale images
- Output: 7 emotion classes with probability scores
- Architecture: Deep CNN with separable convolutions
- Training: Adam optimizer with categorical crossentropy loss

## Dataset

The system is trained on the FER2013 dataset, which contains:
- 35,887 facial images
- 7 emotion categories
- Various lighting conditions and facial expressions
- Professional and amateur photography

## Performance

- **Accuracy**: 66% on FER2013 test set
- **Processing Speed**: Real-time (30+ FPS on modern hardware)
- **Memory Usage**: ~500MB for inference
- **Model Size**: ~102MB

## Customization

### Adding New Emotions

1. Modify the `EMOTIONS` list in `real_time_video.py`
2. Retrain the model with new labels
3. Update the output layer in `models/cnn.py`

### Changing Model Architecture

1. Modify the model definition in `models/cnn.py`
2. Adjust training parameters in `train_emotion_classifier.py`
3. Retrain the model

## Troubleshooting

### Common Issues

1. **Camera not detected**: Ensure webcam is connected and not in use by other applications
2. **Model loading error**: Verify the model file path and file integrity
3. **Low performance**: Check GPU availability and reduce frame resolution
4. **Memory errors**: Reduce batch size or image resolution

### Performance Optimization

- Use GPU acceleration for training
- Reduce frame resolution for faster processing
- Optimize face detection parameters
- Use model quantization for deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{emotion_recognition_2024,
  title={Facial Emotion Recognition Based Virtual AI Assistant},
  author={Khadeeja Hussain},
  year={2024},
  url={https://github.com/Khadeeja2000/Facial-Emotion-Recognition-Based-Virtual-AI-Assistant}
}
```

## Acknowledgments

- FER2013 dataset creators
- OpenCV community
- TensorFlow/Keras development team
- Computer vision research community

## Version History

- **v1.0.0**: Initial release with basic emotion recognition
- **v1.1.0**: Added training pipeline and model architectures
- **v1.2.0**: Improved code structure and documentation
- **v1.3.0**: Added personalized content recommendations and emotion-based user experience
