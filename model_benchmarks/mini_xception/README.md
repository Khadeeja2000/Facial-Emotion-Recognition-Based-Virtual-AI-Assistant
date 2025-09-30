# Mini-XCEPTION Model Benchmark

## Model Overview
- **Architecture**: Mini-XCEPTION (Efficient Convolutional Neural Network)
- **Accuracy**: 59.9% (0.599)
- **Model File**: `models/_mini_XCEPTION.102-0.66.hdf5`
- **Training Epochs**: 102 epochs (based on filename)
- **Validation Accuracy**: 66% (0.66 from filename)

## Performance Metrics

### Overall Performance
- **Test Accuracy**: 59.9%
- **Macro F1-Score**: 64.0%
- **Macro Precision**: 64.4%
- **Macro Recall**: 63.8%

### Per-Class Performance
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Happy** | 76.0% | 80.0% | 77.9% | 900 |
| **Angry** | 66.7% | 73.1% | 69.7% | 520 |
| **Scared** | 67.4% | 67.8% | 67.6% | 568 |
| **Sad** | 69.1% | 64.9% | 66.9% | 655 |
| **Surprised** | 67.1% | 60.0% | 63.4% | 408 |
| **Neutral** | 65.7% | 64.8% | 65.3% | 748 |
| **Disgust** | 38.5% | 35.7% | 37.0% | 98 |

## Training Details
- **Total Samples**: 3,897
- **Epochs**: 102
- **Best Validation Accuracy**: 66%
- **Model Size**: Efficient CNN architecture
- **Training Strategy**: Standard CNN training

## Strengths
- **Good Overall Accuracy**: 59.9% (solid performance)
- **Strong Happy Detection**: 77.9% F1-score
- **Good Generalization**: Consistent performance across emotions
- **Efficient Architecture**: Optimized for emotion recognition

## Weaknesses
- **Poor Disgust Detection**: Only 37.0% F1-score
- **Limited Transfer Learning**: No pre-trained weights used
- **Training Time**: Longer training required (102 epochs)

## Model Files
- **Trained Model**: `models/_mini_XCEPTION.102-0.66.hdf5`
- **Results**: `performance_results/fer2013_detailed_metrics.json`
- **Confusion Matrix**: Available in performance results

## Usage
This model was part of the original integrated system and achieved 59.9% accuracy on FER2013 dataset.
