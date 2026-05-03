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
- **Samples:** 4600 total
- **Classes:** 10 fault conditions
- **Sampling Rate:** 48,000 samples/second
- **Format:** NPZ file (32×32 per sample)

| Class | Description | Crack Size | Samples |
|-------|-------------|------------|---------|
| Normal | Healthy bearing | None | 460 |
| Ball_007 | Ball fault | 0.007 inch | 460 |
| Ball_014 | Ball fault | 0.014 inch | 460 |
| Ball_021 | Ball fault | 0.021 inch | 460 |
| IR_007 | Inner Race fault | 0.007 inch | 460 |
| IR_014 | Inner Race fault | 0.014 inch | 460 |
| IR_021 | Inner Race fault | 0.021 inch | 460 |
| OR_007 | Outer Race fault | 0.007 inch | 460 |
| OR_014 | Outer Race fault | 0.014 inch | 460 |
| OR_021 | Outer Race fault | 0.021 inch | 460 |

Dataset is perfectly balanced 
No missing values 
No duplicate samples 


##  Exploratory Data Analysis (EDA)

### What I Checked:

**Data Quality:**
- Missing values → None found 
- Duplicate samples → None found 
- Class balance → Perfect 460 each 
- Data type → Float64 correct 

**Signal Analysis:**
- Boxplot → Shows amplitude spread per class
- Histogram → Shows value distribution
- Heatmap → Shows feature correlations

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
-  Boxplot outlier detection
-  Feature correlation heatmap


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
| Cell 3 | Missing Values & Duplicate |
| Cell 4 | Statistical Analysis|
| Cell 5 | Boxplot Outlier Detection |
| Cell 6 | Histogram Distribution|
| Cell 7 | Time Domain Feature Extraction |
| Cell 8 | WPT Application |
| Cell 9 | Data Preparation (Split & Scale) |
| Cell 10 | ML Models (RF + SVM) |
| Cell 11 | Deep Learning Model (MLP) |
| Cell 12 | Confusion Matrix Heatmaps |
| Cell 13| FFT Conversion to Frequency Domain |
| Cell 14 | Frequency Feature Extraction |
| Cell 15 | WPT on Frequency Domain |
| Cell 16 | Frequency Data Preparation |
| Cell 17 | Frequency ML Models |
| Cell 18 | Frequency DL Model |
| Cell 19 | Frequency Confusion Matrix |
| Cell 20 | Final Comparison |

##  Models Used

### Baseline Models:
**Random Forest:**
- 200 decision trees vote together
- Best for tabular feature data
- Explainable predictions
- Fast training (5 seconds)

**SVM:**
- RBF kernel for curved boundaries
- Separates 10 classes using
  One vs Rest strategy
- Proven on bearing fault research

### Advanced Model:
**MLP Deep Learning:**



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
# Summary
#  Domain Comparisons
Time Domain → better for DL
Raw signal is complex
DL finds hidden patterns! 

Frequency Domain → better for RF
FFT cleans and organizes data
RF reads clear patterns directly! 

##  Complete 
