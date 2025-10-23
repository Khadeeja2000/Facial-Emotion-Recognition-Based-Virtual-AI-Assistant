# 📊 RESEARCH PAPER IMAGES GUIDE

## 📁 Folder Location
**Path:** `/Users/DELL/Documents/Emotion-recognition/ensemble_learning/paper_images/`

---

## ✅ ALL 11 IMAGES READY TO USE

### **FIGURE 1: Dataset Distribution**
- **File:** `image1.png`
- **Size:** 282 KB
- **Usage:** Section 3 (Dataset and Preprocessing)
- **Description:** Bar and pie charts showing FER2013 class distribution with imbalance visualization
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.9\columnwidth]{image1.png}
\caption{Class distribution in FER2013 dataset showing significant imbalance, with Happy comprising 27.6\% while Disgust represents only 1.7\% of samples.}
\label{fig:dataset-distribution}
\end{figure}
```

---

### **FIGURE 2: System Architecture**
- **File:** `system_architecture.png`
- **Size:** 293 KB
- **Usage:** Section 4.1 (Overall System Framework)
- **Description:** Complete pipeline from video capture to intervention recommendation
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{system_architecture.png}
\caption{Complete system architecture showing the pipeline from live video capture through face detection, emotion recognition, mental health assessment, and adaptive intervention recommendation.}
\label{fig:system-architecture}
\end{figure}
```

---

### **FIGURE 3: Model Architectures**
- **File:** `model_architectures.png`
- **Size:** 309 KB
- **Usage:** After Section 4.4 (EfficientNetB0 Architecture)
- **Description:** Side-by-side comparison of three ensemble models
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{model_architectures.png}
\caption{Architectural comparison of the three ensemble models: Mini-Xception (specialized for FER), MobileNetV2 (transfer learning), and EfficientNetB0 (compound scaling).}
\label{fig:model-architectures}
\end{figure}
```

---

### **FIGURE 4: Performance Comparison**
- **File:** `image9.png`
- **Size:** 305 KB
- **Usage:** Section 6.1 (Overall Performance)
- **Description:** 2×2 grid of bar charts showing accuracy, F1, precision, recall
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{image9.png}
\caption{Comprehensive performance comparison across accuracy, macro F1-score, precision, and recall metrics. The ensemble (red border) consistently outperforms individual models.}
\label{fig:performance-comparison}
\end{figure}
```

---

### **FIGURE 5: Ensemble Benefits**
- **File:** `image10.png`
- **Size:** 178 KB
- **Usage:** Section 6.1 (after discussing ensemble improvements)
- **Description:** Side-by-side comparison and improvement percentages
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{image10.png}
\caption{Ensemble benefits analysis showing percentage improvements over the best individual model (Mini-Xception) across all evaluation metrics.}
\label{fig:ensemble-benefits}
\end{figure}
```

---

### **FIGURE 6: Per-Class Performance**
- **File:** `image11.png`
- **Size:** 171 KB
- **Usage:** Section 6.1 (Per-Class Performance subsection)
- **Description:** Horizontal bar chart of F1-scores by emotion
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.9\columnwidth]{image11.png}
\caption{Per-class F1-scores for the ensemble model. Happy achieves the highest performance (0.806) while Disgust remains most challenging (0.433) due to limited training samples.}
\label{fig:per-class}
\end{figure}
```

---

### **FIGURE 7: Confusion Matrix**
- **File:** `image12.png`
- **Size:** 302 KB
- **Usage:** Section 6.1 (after Per-Class Performance)
- **Description:** Heatmap showing classification patterns
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.9\columnwidth]{image12.png}
\caption{Confusion matrix for ensemble model revealing systematic misclassification patterns: Fear↔Surprise (12.7\%), Disgust→Angry (15.3\%), and Sad→Neutral (11.2\%).}
\label{fig:confusion-matrix}
\end{figure}
```

---

### **FIGURE 8: Grad-CAM Visualizations**
- **File:** `gradcam_visualization.png`
- **Size:** 209 KB
- **Usage:** Section 6.2 (Grad-CAM Visualization Analysis)
- **Description:** 2×4 grid showing attention maps for all 7 emotions
- **LaTeX Code:**
```latex
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{gradcam_visualization.png}
\caption{Grad-CAM visualizations for representative samples across all seven emotions. Heatmaps highlight facial regions contributing to predictions: (a) Happy - focus on eyes and mouth, (b) Sad - attention to inner eyebrows and mouth corners, (c) Angry - concentration on furrowed brows and narrowed eyes, (d) Fear - eye-widening regions, (e) Surprise - eyes and mouth, (f) Disgust - nose and upper lip, (g) Neutral - distributed attention.}
\label{fig:gradcam}
\end{figure*}
```
**Note:** Use `\begin{figure*}` for two-column spanning figure

---

### **FIGURE 9: SOTA Comparison**
- **File:** `image14.png`
- **Size:** 310 KB
- **Usage:** Section 6 (new subsection: Benchmark Comparison)
- **Description:** Bar chart comparing with recent state-of-the-art methods
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{image14.png}
\caption{Comparison with recent state-of-the-art methods on FER2013. Our ensemble (70.3\%) achieves competitive performance, ranking among top-tier approaches.}
\label{fig:sota-comparison}
\end{figure}
```

---

### **FIGURE 10: Latency Breakdown**
- **File:** `latency_breakdown.png`
- **Size:** 275 KB
- **Usage:** Section 6.3 (System Performance)
- **Description:** Pie chart and bar chart of processing times
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{latency_breakdown.png}
\caption{Real-time processing latency breakdown showing 82ms total pipeline latency: face detection (15ms), preprocessing (12ms), ensemble inference (42ms), aggregation (5ms), and visualization (8ms).}
\label{fig:latency}
\end{figure}
```

---

### **FIGURE 11: User Study Results**
- **File:** `user_study_results.png`
- **Size:** 255 KB
- **Usage:** Section 6.3 (after user study discussion)
- **Description:** Three-panel visualization of acceptance rates, engagement, mood
- **LaTeX Code:**
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{user_study_results.png}
\caption{User study results showing (a) intervention acceptance rates by emotion category, (b) average engagement duration, and (c) pre/post-intervention mood ratings demonstrating significant improvement (p<0.01).}
\label{fig:user-study}
\end{figure}
```

---

## 📝 LATEX GRAPHICS PATH SETUP

Add this at the beginning of your LaTeX document (already in your template):

```latex
\graphicspath{{paper_images/}}
```

This tells LaTeX to look for images in the `paper_images/` folder.

---

## 🎯 INSERTION ORDER IN PAPER

```
SECTION 3: Dataset
└── FIGURE 1: image1.png (Dataset Distribution)

SECTION 4: Methodology
├── FIGURE 2: system_architecture.png (System Architecture)
└── FIGURE 3: model_architectures.png (Model Comparison)

SECTION 6: Results
├── FIGURE 4: image9.png (Performance Comparison)
├── FIGURE 5: image10.png (Ensemble Benefits)
├── FIGURE 6: image11.png (Per-Class Performance)
├── FIGURE 7: image12.png (Confusion Matrix)
├── FIGURE 8: gradcam_visualization.png (Grad-CAM)
├── FIGURE 9: image14.png (SOTA Comparison)
├── FIGURE 10: latency_breakdown.png (Latency)
└── FIGURE 11: user_study_results.png (User Study)
```

---

## 🔗 TEXT REFERENCES TO ADD

### In Section 3:
```latex
...with Disgust underrepresented (1.7\% of samples), as illustrated in Fig.~\ref{fig:dataset-distribution}.
```

### In Section 4.1:
```latex
The complete system architecture is depicted in Fig.~\ref{fig:system-architecture}.
```

### After Section 4.4:
```latex
Fig.~\ref{fig:model-architectures} compares the architectural designs of the three ensemble components.
```

### In Section 6.1 (after Table 1):
```latex
Fig.~\ref{fig:performance-comparison} visualizes these performance differences across evaluation metrics, with detailed improvement breakdown shown in Fig.~\ref{fig:ensemble-benefits}.
```

### In Section 6.1 (Per-Class):
```latex
Performance varies significantly across emotion classes (Fig.~\ref{fig:per-class}). The confusion matrix (Fig.~\ref{fig:confusion-matrix}) reveals systematic error patterns.
```

### In Section 6.2:
```latex
Representative Grad-CAM visualizations (Fig.~\ref{fig:gradcam}) demonstrate attention alignment with FACS-defined action units.
```

### In Section 6 (new subsection):
```latex
Our ensemble ranks competitively among recent approaches (Fig.~\ref{fig:sota-comparison}).
```

### In Section 6.3:
```latex
Processing latency breakdown (Fig.~\ref{fig:latency}) confirms real-time capability. User study results (Fig.~\ref{fig:user-study}) demonstrate practical effectiveness.
```

---

## ✨ IMAGE QUALITY SPECIFICATIONS

- **Resolution:** 300 DPI (publication quality)
- **Format:** PNG with transparency
- **Color Space:** RGB
- **File Sizes:** 171 KB - 310 KB (optimized)
- **Dimensions:** Optimized for IEEE single/double column

---

## 🎉 ALL IMAGES READY!

**Total Images:** 11 figures
**Total Size:** ~2.8 MB
**Status:** ✅ All created and ready for LaTeX compilation

---

## 📞 SUPPORT

If any image needs modification:
1. Edit `create_paper_images.py`
2. Run: `python3 create_paper_images.py`
3. Images will be regenerated in `paper_images/`

**Location:** `/Users/DELL/Documents/Emotion-recognition/ensemble_learning/paper_images/`

