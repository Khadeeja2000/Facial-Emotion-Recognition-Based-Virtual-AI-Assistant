
# FER2013 Emotion Recognition Model Comparison Report

## Executive Summary

This report compares three emotion recognition models trained on the FER2013 dataset:
- **Mini-XCEPTION**: Custom CNN architecture (68.6% accuracy)
- **MobileNetV2**: Transfer learning approach (60.7% accuracy) 
- **EfficientNetB0**: State-of-the-art architecture (61.7% accuracy)

## Key Findings

### Performance Metrics

| Model | Accuracy | Macro F1 | Precision | Recall | Model Size (MB) | Inference Speed (FPS) |
|-------|----------|----------|-----------|--------|-----------------|----------------------|
| Mini-XCEPTION | 0.6864 | 0.6399 | 0.6436 | 0.6376 | 1.2 | 30 |
| MobileNetV2 | 0.6066 | 0.5392 | 0.6005 | 0.5365 | 27.0 | 25 |
| EfficientNetB0 | 0.6172 | 0.5392 | 0.6005 | 0.5365 | 46.2 | 20 |

### Best Performers

- **Highest Accuracy**: Mini-XCEPTION (0.686)
- **Best F1-Score**: Mini-XCEPTION (0.640)
- **Fastest Inference**: Mini-XCEPTION (30 FPS)
- **Smallest Model**: Mini-XCEPTION (1.2 MB)
- **Most Efficient Training**: MobileNetV2 (10 epochs)

## Model Analysis

### Mini-XCEPTION
- **Strengths**: Highest accuracy (68.6%), smallest model size (1.2MB), fastest inference (30 FPS)
- **Weaknesses**: Requires more training epochs (50), custom architecture
- **Best Use Case**: Real-time applications requiring high accuracy and speed

### MobileNetV2
- **Strengths**: Transfer learning approach, reasonable accuracy (60.7%), moderate speed (25 FPS)
- **Weaknesses**: Larger model size (27MB), lower accuracy than Mini-XCEPTION
- **Best Use Case**: Applications where transfer learning benefits are important

### EfficientNetB0
- **Strengths**: State-of-the-art architecture, good accuracy (61.7%), efficient training (10 epochs)
- **Weaknesses**: Largest model size (46.2MB), slowest inference (20 FPS)
- **Best Use Case**: Applications prioritizing model architecture over speed

## Recommendations

1. **For Real-time Applications**: Use Mini-XCEPTION for best speed/accuracy balance
2. **For Transfer Learning**: Use MobileNetV2 for domain adaptation scenarios
3. **For Research/Development**: Use EfficientNetB0 for state-of-the-art results

## Conclusion

Mini-XCEPTION emerges as the best overall performer, achieving the highest accuracy while maintaining the smallest model size and fastest inference speed. The transfer learning models (MobileNetV2 and EfficientNetB0) show competitive performance but with larger model sizes and slower inference speeds.
