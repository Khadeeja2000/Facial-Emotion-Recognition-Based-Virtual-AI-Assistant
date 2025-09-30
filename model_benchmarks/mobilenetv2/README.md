# MobileNetV2 Transfer Learning Model Benchmark

## Model Overview
- **Architecture**: MobileNetV2 with Transfer Learning
- **Accuracy**: 60.5% (0.605)
- **Model File**: `mobilenetv2_models/fer_mnv2_best.h5`
- **Training Strategy**: Two-phase transfer learning
- **Pre-trained Weights**: ImageNet

## Performance Metrics

### Overall Performance
- **Test Accuracy**: 60.5%
- **Macro F1-Score**: 54.1%
- **Macro Precision**: 59.1%
- **Macro Recall**: 53.3%

### Per-Class Performance
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Happy** | 78.3% | 86.1% | 82.0% | 1,774 |
| **Surprise** | 72.4% | 72.0% | 72.2% | 831 |
| **Neutral** | 50.3% | 65.8% | 57.0% | 1,233 |
| **Angry** | 49.0% | 54.8% | 51.8% | 958 |
| **Sad** | 55.7% | 38.6% | 45.6% | 1,247 |
| **Fear** | 46.5% | 37.2% | 41.4% | 1,024 |
| **Disgust** | 61.8% | 18.9% | 29.0% | 111 |

## Training Details
- **Total Samples**: 22,907 (train) + 5,742 (val) + 7,178 (test)
- **Phase 1 (Frozen)**: 5 epochs with frozen backbone
- **Phase 2 (Fine-tune)**: 10 epochs with unfrozen upper layers
- **Batch Size**: 64
- **Learning Rates**: 1e-3 (frozen) → 5e-5 (fine-tune)
- **Model Size**: ~2.2M parameters

## Training Strategy
1. **Phase 1**: Freeze MobileNetV2 backbone, train only classifier head
2. **Phase 2**: Unfreeze upper layers (from block_13_expand), fine-tune with low LR
3. **Data Augmentation**: Rotation, brightness, contrast, horizontal flip
4. **Preprocessing**: Resize to 160x160, normalize for MobileNetV2

## Strengths
- **Excellent Happy Detection**: 82.0% F1-score
- **Good Surprise Detection**: 72.2% F1-score
- **Transfer Learning Benefits**: Pre-trained ImageNet features
- **Efficient Architecture**: MobileNetV2 optimized for mobile/edge deployment
- **Fast Inference**: Optimized for real-time applications

## Weaknesses
- **Lower Overall Accuracy**: 60.5% vs 68.6% (Mini-XCEPTION)
- **Poor Disgust Detection**: Only 29.0% F1-score
- **Challenging Fear/Sad**: Moderate performance on negative emotions
- **Class Imbalance**: Struggles with underrepresented classes

## Model Files
- **Best Model**: `mobilenetv2_models/fer_mnv2_best.h5`
- **Last Model**: `mobilenetv2_models/fer_mnv2_best_last.h5`
- **Training Log**: `mobilenetv2_models/training_log_transfer.csv`
- **Confusion Matrix**: `mobilenetv2_models/cm_counts.png`, `cm_normalized.png`

## Usage
```bash
# Training
python3 mobilenetv2_fer2013.py --mode train --epochs_frozen 5 --epochs_finetune 10

# Evaluation
python3 mobilenetv2_fer2013.py --mode evaluate --model_path mobilenetv2_models/fer_mnv2_best.h5

# Live Detection
python3 mobilenetv2_fer2013.py --mode live --model_path mobilenetv2_models/fer_mnv2_best.h5
```

## Key Insights
- Transfer learning provides good foundation but needs more fine-tuning
- Happy emotions are detected very well (82% F1-score)
- Negative emotions (disgust, fear) remain challenging
- Model is suitable for real-time applications due to MobileNetV2 efficiency
