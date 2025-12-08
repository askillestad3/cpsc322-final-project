# Predicting Student Performance in Secondary Education

**Team Members:** Abby Skillestad and Miles Mercer  
**Group:** Group 5  
**Course:** CPSC 322, Fall 2025

## Project Overview

This project uses machine learning to predict whether secondary school mathematics students will pass or fail based on demographic, social, and academic features. We implemented and compared four classifiers (Dummy, k-NN, Decision Tree, Random Forest) using custom from-scratch implementations.

## Key Results

| Classifier | Stratified 10-Fold CV Accuracy |
|------------|-------------------------------|
| Dummy (baseline) | 67.09% |
| k-Nearest Neighbors (k=5) | 68.35% |
| Decision Tree | 58.23% |
| **Random Forest (N=500, M=100, F=6)** | **67.85%** |

**Key Finding:** Past failures is the strongest predictor of student performance.

## Dataset

**Source:** [UCI Machine Learning Repository - Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

**Citation:**  
P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance."  
*Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008)*, pp. 5-12, Porto, Portugal, April 2008.

- **Instances:** 649 students (Mathematics course)
- **Features:** 30 attributes (demographic, social, academic)
- **Target:** Pass (G3 ≥ 10) vs Fail (G3 < 10)
- **Class Distribution:** 73% pass, 27% fail

## Installation

```bash
# Clone repository
git clone https://github.com/[your-username]/CPSC322-FINAL-PROJECT.git
cd CPSC322-FINAL-PROJECT

# Activate the existing virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages (if not already installed)
pip install jupyter matplotlib numpy pytest pytest-cov

# Dataset is already included in data/student-mat.csv
```

## Running the Project

### Run Jupyter Notebook
```bash
# Make sure virtual environment is activated first
jupyter notebook
# Open final.ipynb in your browser and run all cells
```

### Run Unit Tests
```bash
# Make sure virtual environment is activated first
python -m pytest test_myclassifiers.py -v
```

## Project Structure

```
├── final.ipynb                      # Main technical report
├── mysklearn/                       # Custom ML implementations
│   ├── __init__.py
│   ├── myclassifiers.py             # kNN, Decision Tree, Random Forest
│   ├── myevaluation.py              # Cross-validation, metrics
│   ├── mypytable.py                 # Data table class
│   ├── myutils.py                   # Utility functions
│   ├── myrandomforestclassifier.py  # Random Forest implementation
│   └── mysimplelinearregressor.py   # Linear regression
├── test_myclassifiers.py            # Unit tests
├── data/
│   └── student-mat.csv              # Dataset
├── output/                          # Generated visualizations
│   ├── classifier_comparison.png
│   ├── failures_impact.png
│   ├── grade_distribution.png
│   ├── parent_education_impact.png
│   └── studytime_vs_grade.png
└── venv/                            # Virtual environment (already set up)
```
