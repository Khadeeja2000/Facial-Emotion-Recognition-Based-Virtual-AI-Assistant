# Facial Emotion Recognition Mental Health Assessment System

A comprehensive research project implementing real-time facial emotion recognition integrated with mental health assessment and therapeutic intervention capabilities.

## Repository Overview

This repository contains multiple implementations of facial emotion recognition systems, ranging from basic emotion detection to complete mental health assessment with virtual AI assistant integration.

## Repository Structure

### 🌟 Main Branches

#### 1. **Main Branch** (`main`)
**Clean repository overview and navigation**

- **Purpose**: Repository overview and branch navigation
- **Content**: Documentation for all branches and components
- **Files**: Core documentation and navigation files only

#### 2. **FER2013 Emotion Detection** (`fer2013-emotion-detection`)
**Pure emotion recognition system with 77% accuracy**

- **Features**: Real-time 7-emotion detection (angry, disgust, scared, happy, sad, surprised, neutral)
- **Model**: Mini-XCEPTION CNN trained on FER2013 dataset
- **Performance**: 77% accuracy, 30+ FPS real-time processing
- **Use Case**: Basic emotion recognition for research and development

#### 3. **Complete Integrated System** (`complete-integrated-system`)
**Full mental health assessment with virtual AI assistant**

- **Features**: FER2013 emotions + RAVDESS mental health models + therapeutic interventions
- **Models**: Emotion detection + Depression/Anxiety/Stress assessment
- **Interface**: Professional popup with clickable therapeutic content
- **Use Case**: Complete mental health screening and intervention system

#### 4. **Legacy Branches**
- `fer-77-live` - Original working FER2013 implementation
- `Addition-of-more-features` - Development branch with experimental features

## Quick Start

### For Basic Emotion Recognition
```bash
git checkout fer2013-emotion-detection
python real_time_video.py
```

### For Complete Mental Health System
```bash
git checkout complete-integrated-system
python ravdess_mental_health_trainer.py  # Train models first
python fer_ravdess_integrated_system.py  # Run complete system
```

## Technical Specifications

### FER2013 Emotion Detection
- **Dataset**: 35,887 facial images
- **Model**: Mini-XCEPTION CNN
- **Accuracy**: 77% on test set
- **Processing**: 30+ FPS real-time
- **Input**: 64x64x1 grayscale images
- **Output**: 7 emotion probabilities

### RAVDESS Mental Health Models
- **Dataset**: 2,452 professional actor samples
- **Algorithm**: Random Forest (selected via cross-validation)
- **Conditions**: Depression, Anxiety, Stress detection
- **Features**: 22 engineered features per sample
- **Validation**: 70/30 split with 5-fold cross-validation

### Complete Integrated System
- **Real-time Processing**: <50ms per frame
- **Memory Usage**: ~500MB
- **Monitoring Cycle**: Exactly 60 seconds
- **Interface**: Pygame-based interactive popup
- **Content**: Therapeutic videos, music, podcasts

## Research Contributions

### Technical Achievements
1. **Complete ML Pipeline**: Proper validation methodology with cross-validation
2. **Real-time Integration**: Emotion recognition + mental health assessment
3. **Evidence-based Analysis**: Transparent explanations for all predictions
4. **Professional Interface**: Therapeutic intervention system

### Academic Value
- Demonstrates systematic approach to emotion-mental health mapping
- Implements research-grade training pipeline
- Documents limitations and methodological considerations
- Provides foundation for clinical validation studies

## System Architecture

```
Camera Input → FER2013 Emotion Detection → RAVDESS Mental Health Models → 
Evidence-based Analysis → Virtual AI Assistant → Therapeutic Interventions
```

## Performance Metrics

| Component | Accuracy | Processing Speed | Memory Usage |
|-----------|----------|------------------|--------------|
| FER2013 Emotion Detection | 77% | 30+ FPS | ~200MB |
| RAVDESS Mental Health Models | 100%* | <50ms | ~300MB |
| Complete Integrated System | Combined | <50ms/frame | ~500MB |

*Note: 100% accuracy may indicate overfitting - see documentation for details

## Installation Requirements

### Hardware
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5+)
- **RAM**: 8GB minimum
- **Camera**: 720p webcam with 30fps
- **Storage**: 200MB for models and code

### Software Dependencies
```bash
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install opencv-python==4.8.0
pip install scikit-learn==1.3.0
pip install pygame==2.5.0
pip install pandas==2.0.0
pip install numpy==1.24.0
```

## Ethical Considerations

- **Not for Clinical Diagnosis**: Screening and wellness tool only
- **Privacy**: Facial data processed locally, not stored
- **Disclaimers**: Clear limitations documentation
- **Professional Help**: Encourages seeking professional mental health support

## Limitations

### Methodological
- Emotion-mental health correlations based on research approximations
- Single-modal input (facial expressions only)
- No temporal persistence analysis
- Not validated on clinical populations

### Technical
- Requires good lighting conditions
- Processes one face at a time
- Static correlations (no personalization)
- Internet required for therapeutic content

## Future Work

- Clinical validation with patient populations
- Multi-modal fusion (facial + voice + physiological)
- Temporal pattern analysis for emotion persistence
- Personalization algorithms for individual differences
- Integration with validated clinical assessment scales

## Documentation

Each branch contains detailed documentation:
- **FER2013 Branch**: Emotion detection technical specifications
- **Complete System Branch**: Full pipeline documentation with performance metrics
- **Individual Files**: Inline documentation and comments

## Citation

If you use this system in your research, please cite:

```
FER2013 Dataset:
@article{fer2013,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and others},
  journal={Neural Networks},
  volume={64},
  pages={59--63},
  year={2015}
}

RAVDESS Dataset:
@article{ravdess2018,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R and Russo, Frank A},
  journal={PLoS one},
  volume={13},
  number={5},
  pages={e0196391},
  year={2018}
}
```

## Contact

For technical questions or research collaboration, please refer to the individual branch documentation and issue tracking system.

---

**Repository Statistics**:
- **Branches**: 5 (main + 4 feature branches)
- **Total Code**: 2,000+ lines
- **Models**: 4 (FER2013 + 3 RAVDESS mental health models)
- **Datasets**: 2 (FER2013 + RAVDESS)
- **Performance Metrics**: Comprehensive validation across all components