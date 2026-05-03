# Predictive-Quality-Control-of-Machined-Shafts
AI-powered bearing fault detection system using  signal processing and machine learning. Classifies  10 bearing fault conditions from vibration signals  with 97.39% accuracy using Time &amp; Frequency Domain  analysis, WPT feature extraction, Random Forest,  SVM and Deep Learning models.

##  Overview

This project builds a complete AI pipeline to detect 
and classify bearing faults in rotating machinery. 
Vibration signals from bearings are analyzed using 
both Time Domain and Frequency Domain approaches. 
The system identifies 10 different bearing conditions 
including normal operation and various fault types 
with their severity levels achieving 97.39% accuracy.

Real world impact:
Detects bearing faults BEFORE machines break down,
preventing unexpected downtime, saving costs and
keeping factory workers safe.
##  Problem Statement

Machined shafts rotate using bearings. When bearing
quality degrades, sudden machine breakdown occurs
causing production loss and worker injuries.
Traditional manual inspection is slow and misses
early stage faults.

Our AI system monitors vibration signals continuously
and detects fault type and severity automatically
in seconds.
##  Dataset

- **Name:** CWRU Bearing Fault Dataset
- **Source:** Case Western Reserve University
- **Samples:** 4600 (perfectly balanced)
- **Classes:** 10 fault conditions
- **Sampling Rate:** 48,000 samples/second

| Class | Description | Samples |
|-------|-------------|---------|
| Normal | Healthy bearing | 460 |
| Ball_007 | Ball fault - small | 460 |
| Ball_014 | Ball fault - medium | 460 |
| Ball_021 | Ball fault - large | 460 |
| IR_007 | Inner Race - small | 460 |
| IR_014 | Inner Race - medium | 460 |
| IR_021 | Inner Race - large | 460 |
| OR_007 | Outer Race - small | 460 |
| OR_014 | Outer Race - medium | 460 |
| OR_021 | Outer Race - large | 460 |

##  Features

-  Complete Time Domain Analysis
-  Complete Frequency Domain Analysis
-  Wavelet Packet Transform (WPT) feature extraction
-  FFT signal conversion
-  34 features extracted per sample
-  3 ML/DL models trained and compared
-  Confusion matrix visualization
-  Time vs Frequency domain comparison
-  Training history visualization


##  Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python 3.8+ |
| Platform | Google Colab |
| DL Framework | TensorFlow / Keras |
| ML Library | Scikit-learn |
| Signal Processing | PyWavelets, NumPy |
| Visualization | Matplotlib, Seaborn |
| Data Handling | NumPy, SciPy |

### Notebook Structure (16 Cells):

| Cell | Description |
|------|-------------|
| Cell 1 | Libraries & Setup |
| Cell 2 | Data Upload & Analysis |
| Cell 3 | Time Domain Feature Extraction |
| Cell 4 | WPT Application |
| Cell 5 | Data Preparation (Split & Scale) |
| Cell 6 | ML Models (RF + SVM) |
| Cell 7 | Deep Learning Model (MLP) |
| Cell 8 | Confusion Matrix Heatmaps |
| Cell 9 | FFT Conversion to Frequency Domain |
| Cell 10 | Frequency Feature Extraction |
| Cell 11 | WPT on Frequency Domain |
| Cell 12 | Frequency Data Preparation |
| Cell 13 | Frequency ML Models |
| Cell 14 | Frequency DL Model |
| Cell 15 | Frequency Confusion Matrix |
| Cell 16 | Final Comparison |

##  How To Run

1. Open `CAPSTONE PROJECT.ipynb` in Google Colab
2. Upload `CWRU_48k_load_1_CNN_data.npz` when prompted
3. Run cells one by one in order
4. View results and visualizations

```python
# Install required library
!pip install vmdpy -q

# All other libraries are 
# pre-installed in Google Colab
```

# Overall Best = Deep Learning on Time Domain
   Accuracy     = 97.39%
   Test Samples = 920

## ⚙️ Complete Pipeline
