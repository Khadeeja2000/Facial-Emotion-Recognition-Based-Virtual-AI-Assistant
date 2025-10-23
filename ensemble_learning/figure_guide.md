# Research Paper Figure Guide

## Essential Figures to Include

### 1. **Figure 1: Dataset Distribution (Add this)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\linewidth]{dataset_distribution.png}
    \caption{FER2013 Dataset Distribution by Emotion Class}
    \label{fig:dataset-dist}
\end{figure}
```
**Purpose**: Show class imbalance in FER2013 dataset
**Content**: Bar chart showing number of samples per emotion class

### 2. **Figure 2: Model Architecture Comparison (Add this)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{model_architectures.png}
    \caption{Ensemble Model Architectures: Mini-XCEPTION, MobileNetV2, and EfficientNetB0}
    \label{fig:model-arch}
\end{figure}
```
**Purpose**: Visualize the three models used in ensemble
**Content**: Architecture diagrams of all three models

### 3. **Figure 3: Performance Comparison (Use existing)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{results/ENHANCED_performance_comparison.png}
    \caption{Model Performance Comparison on FER2013 Dataset}
    \label{fig:performance-comp}
\end{figure}
```
**Purpose**: Compare accuracy, F1-score, precision, recall across models
**Status**: ✅ Already created

### 4. **Figure 4: Ensemble Benefits Analysis (Use existing)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{results/ENHANCED_ensemble_benefits.png}
    \caption{Ensemble vs Individual Model Performance}
    \label{fig:ensemble-benefits}
\end{figure}
```
**Purpose**: Show improvement of ensemble over best individual model
**Status**: ✅ Already created

### 5. **Figure 5: Per-Class Performance (Use existing)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\linewidth]{results/ENHANCED_per_class_analysis.png}
    \caption{Per-Class F1-Score Performance}
    \label{fig:per-class}
\end{figure}
```
**Purpose**: Show performance for each emotion class
**Status**: ✅ Already created

### 6. **Figure 6: Confusion Matrix (Add this)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\linewidth]{confusion_matrix.png}
    \caption{Confusion Matrix for Ensemble Model}
    \label{fig:confusion-matrix}
\end{figure}
```
**Purpose**: Show misclassification patterns
**Content**: Heatmap of true vs predicted emotions

### 7. **Figure 7: Training Curves (Add this)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{training_curves.png}
    \caption{Training and Validation Curves for Ensemble Models}
    \label{fig:training-curves}
\end{figure}
```
**Purpose**: Show model convergence and overfitting analysis
**Content**: Loss and accuracy curves over epochs

### 8. **Figure 8: Real-Time System Pipeline (Add this)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{system_pipeline.png}
    \caption{Real-Time Emotion Recognition and Response Pipeline}
    \label{fig:system-pipeline}
\end{figure}
```
**Purpose**: Show the complete system workflow
**Content**: Flowchart from camera input to AI response

### 9. **Figure 9: Performance Table (Use existing)**
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=1\linewidth]{results/ENHANCED_performance_table.png}
    \caption{Comprehensive Performance Comparison Table}
    \label{fig:performance-table}
\end{figure}
```
**Purpose**: Professional results table for publication
**Status**: ✅ Already created

## Tables to Include

### **Table 1: Dataset Statistics**
```latex
\begin{table}[h]
\centering
\caption{FER2013 Dataset Statistics}
\label{tab:dataset-stats}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Emotion} & \textbf{Train} & \textbf{Validation} & \textbf{Test} \\
\hline
Angry & 3,995 & 999 & 491 \\
Disgust & 436 & 109 & 55 \\
Fear & 4,097 & 1,024 & 528 \\
Happy & 7,215 & 1,804 & 879 \\
Sad & 4,830 & 1,207 & 594 \\
Surprise & 3,171 & 793 & 416 \\
Neutral & 4,965 & 1,241 & 626 \\
\hline
\textbf{Total} & \textbf{28,709} & \textbf{7,177} & \textbf{3,589} \\
\hline
\end{tabular}
\end{table}
```

### **Table 2: Model Performance (Use existing results)**
```latex
\begin{table}[h]
\centering
\caption{Performance Comparison of Individual Models and Ensemble}
\label{tab:results}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Macro F1} & \textbf{Precision} & \textbf{Recall} \\
\hline
Mini-XCEPTION & 0.6841 & 0.5661 & 0.6185 & 0.5509 \\
MobileNetV2 & 0.4764 & 0.2935 & 0.3601 & 0.3103 \\
EfficientNetB0 & 0.3970 & 0.2552 & 0.3274 & 0.2873 \\
\textbf{Ensemble} & \textbf{0.7033} & \textbf{0.5771} & \textbf{0.7122} & \textbf{0.5574} \\
\hline
\end{tabular}
\end{table}
```

## Figures You Already Have ✅
1. ENHANCED_performance_comparison.png
2. ENHANCED_ensemble_benefits.png  
3. ENHANCED_per_class_analysis.png
4. ENHANCED_performance_table.png

## Figures You Need to Create
1. Dataset distribution chart
2. Model architecture diagrams
3. Confusion matrix
4. Training curves
5. System pipeline diagram

## Recommended Figure Placement in Paper
- **Figure 1**: After FER2013 Dataset Details section
- **Figure 2**: After Methodology section
- **Figure 3**: In Results section
- **Figure 4**: In Results section
- **Figure 5**: In Results section
- **Figure 6**: In Results section
- **Figure 7**: In Experimental Setup section
- **Figure 8**: In System Design section
- **Figure 9**: In Results section

## Figure Quality Requirements
- Resolution: 300 DPI minimum
- Format: PNG or PDF
- Font size: Readable at publication size
- Colors: Use colorblind-friendly palette
- Labels: Clear, descriptive captions
