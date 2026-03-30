# 🛡️ Insurance Fraud Detection System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A comprehensive machine learning pipeline for detecting insurance fraud using advanced ensemble methods, hyperparameter optimization, and MLflow tracking.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Model Performance](#model-performance)
- [MLflow Tracking](#mlflow-tracking)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

Insurance fraud costs the industry billions annually. This project implements a complete ML pipeline to detect fraudulent insurance claims using multiple algorithms, automated hyperparameter tuning with Optuna, and comprehensive experiment tracking with MLflow.

### Key Highlights
- ✅ 7 different ML algorithms tested and compared
- ✅ Automated hyperparameter optimization (30 trials per model)
- ✅ Handles class imbalance with advanced techniques
- ✅ Complete MLflow experiment tracking
- ✅ Production-ready pipeline with preprocessing
- ✅ ROC-AUC optimized for fraud detection

---

## ✨ Features

### 🔍 Data Processing
- Automated data cleaning and validation
- Duplicate detection and removal
- Missing value handling
- Feature encoding (numerical + categorical)
- Stratified train-test splitting

### 🤖 Machine Learning
- **7 Algorithms Evaluated:**
  - XGBoost (with scale_pos_weight)
  - Random Forest (balanced classes)
  - Gradient Boosting
  - Logistic Regression (L2/None penalty)
  - K-Nearest Neighbors
  - Decision Tree (balanced)
  - AdaBoost

### ⚙️ Hyperparameter Optimization
- Optuna-based Bayesian optimization
- 30 trials per model
- Stratified 5-fold cross-validation
- ROC-AUC metric optimization

### 📊 Evaluation Metrics
- ROC-AUC (primary metric)
- Precision & Recall
- F1-Score
- Accuracy
- Confusion Matrix
- Training/Testing Time
- Model Size

### 🔬 Experiment Tracking
- Complete MLflow integration
- Nested run tracking
- Model versioning and registration
- Parameter and metric logging
- Artifact storage

---

## 📊 Dataset

### Expected Format
The system expects a CSV file with the following characteristics:

```
Columns:
- Feature columns (numerical and categorical)
- Target column: 'PotentialFraud' (Yes/No or 1/0)
```

### Data Requirements
- **Format:** CSV file
- **Target Variable:** Binary (Fraud/Non-Fraud)
- **Features:** Mixed (numerical and categorical supported)
- **Size:** No strict limit (tested on datasets with 10K+ rows)

### Example Schema
```csv
Age,Gender,ClaimAmount,PolicyType,Region,PreviousClaims,PotentialFraud
45,M,15000,Comprehensive,North,2,No
32,F,8500,Third Party,South,0,Yes
...
```

> **Note:** Update the `DATA_PATH` variable in the script to point to your dataset.

---

## 🤖 Models

### Algorithm Details

| Model | Key Hyperparameters | Class Imbalance Handling |
|-------|-------------------|------------------------|
| **XGBoost** | n_estimators, learning_rate, max_depth, gamma, subsample | `scale_pos_weight` |
| **Random Forest** | n_estimators, max_depth, min_samples_split, max_features | `class_weight='balanced'` |
| **Gradient Boosting** | n_estimators, learning_rate, max_depth, subsample | Native handling |
| **Logistic Regression** | C, penalty, solver, max_iter | `class_weight='balanced'` |
| **KNN** | n_neighbors, weights, metric, p | Distance weighting |
| **Decision Tree** | criterion, max_depth, min_samples_split | `class_weight='balanced'` |
| **AdaBoost** | n_estimators, learning_rate, base_estimator | Native handling |

### Why These Models?

- **XGBoost:** Industry standard for fraud detection, handles imbalanced data well
- **Random Forest:** Robust to overfitting, provides feature importance
- **Gradient Boosting:** Sequential learning, captures complex patterns
- **Logistic Regression:** Fast, interpretable baseline
- **KNN:** Instance-based learning, good for local patterns
- **Decision Tree:** Interpretable rules, base for ensemble methods
- **AdaBoost:** Adaptive boosting, focuses on hard cases

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/insurance-fraud-detection.git
cd insurance-fraud-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies
Create a `requirements.txt` file with:
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
optuna>=3.0.0
mlflow>=2.0.0
joblib>=1.1.0
```

---

## 💻 Usage

### Quick Start

1. **Prepare Your Data**
   ```python
   # Update the DATA_PATH in the script
   DATA_PATH = "path/to/your/cleaned.csv"
   ```

2. **Run the Pipeline**
   ```bash
   python fraud_detection_complete.py
   ```

3. **Monitor Progress**
   The script will display real-time progress with:
   - Data loading and preprocessing stats
   - Optimization progress for each model
   - Cross-validation scores
   - Final evaluation metrics

### Expected Runtime
- Small dataset (<10K rows): 5-15 minutes
- Medium dataset (10K-100K rows): 15-45 minutes
- Large dataset (>100K rows): 45+ minutes

### Output Files

After execution, you'll find:

```
📁 Project Directory
├── best_fraud_model.pkl                    # Production model
├── fraud_label_encoder.pkl                 # Label encoder
├── fraud_model_comparison_results.csv      # Comparison table
└── mlruns/                                 # MLflow tracking data
    └── [experiment_id]/
        ├── meta.yaml
        └── [run_ids]/
```

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load the model and encoder
model = joblib.load('best_fraud_model.pkl')
encoder = joblib.load('fraud_label_encoder.pkl')

# Prepare new data
new_data = pd.DataFrame({
    'Age': [45],
    'Gender': ['M'],
    'ClaimAmount': [15000],
    # ... other features
})

# Make prediction
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[:, 1]

# Decode prediction
fraud_label = encoder.inverse_transform(prediction)
print(f"Prediction: {fraud_label[0]}")
print(f"Fraud Probability: {probability[0]:.2%}")
```

---

## 📁 Project Structure

```
insurance-fraud-detection/
│
├── fraud_detection_complete.py      # Main pipeline script
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── IMPROVEMENTS_SUMMARY.md           # Technical improvements doc
│
├── data/                            # Data directory (create this)
│   └── cleaned.csv                  # Your dataset (add this)
│
├── models/                          # Saved models (auto-created)
│   ├── best_fraud_model.pkl
│   └── fraud_label_encoder.pkl
│
├── results/                         # Results (auto-created)
│   └── fraud_model_comparison_results.csv
│
├── mlruns/                          # MLflow tracking (auto-created)
│
└── notebooks/                       # Jupyter notebooks (optional)
    ├── 01_EDA.ipynb
    ├── 02_Feature_Engineering.ipynb
    └── 03_Model_Analysis.ipynb
```

---

## 📈 Results

### Sample Performance Metrics

> **Note:** These are example results. Your actual results will vary based on your dataset.

```
🏆 COMPLETE MODEL COMPARISON (ROC-AUC RANKED)
═══════════════════════════════════════════════════════════════════════════

Model                   ROC-AUC    F1-Score  Precision  Recall    Accuracy
────────────────────────────────────────────────────────────────────────────
XGBoost                 0.9542     0.8834    0.8923     0.8746    0.9312
RandomForest            0.9487     0.8756    0.8845     0.8668    0.9267
GradientBoosting        0.9423     0.8698    0.8734     0.8662    0.9234
LogisticRegression      0.9156     0.8334    0.8456     0.8213    0.9023
AdaBoost                0.9089     0.8267    0.8401     0.8134    0.8989
DecisionTree            0.8845     0.7923    0.8123     0.7725    0.8756
KNN                     0.8678     0.7734    0.7989     0.7483    0.8623
```

### Key Insights

📊 **Best Performing Model:** XGBoost
- Highest ROC-AUC: 0.9542
- Best F1-Score: 0.8834
- Excellent balance of precision and recall

🎯 **Why XGBoost Wins:**
- Native handling of class imbalance
- Regularization prevents overfitting
- Captures non-linear patterns
- Fast inference time

⚡ **Speed vs. Performance Trade-off:**
- Logistic Regression: Fastest (good baseline)
- XGBoost: Best performance (recommended for production)
- Random Forest: Good balance

---

## 🎯 Model Performance

### Confusion Matrix (Example - XGBoost)

```
                  Predicted
                No Fraud    Fraud
Actual  No      4523        87
Fraud   Fraud    156       734

Precision: 89.4%  (of predicted fraud, 89.4% are actually fraud)
Recall: 82.5%     (of actual fraud, 82.5% are detected)
F1-Score: 85.8%   (harmonic mean of precision and recall)
```

### ROC Curve Interpretation

- **ROC-AUC > 0.95:** Excellent discrimination
- **ROC-AUC 0.90-0.95:** Very good discrimination  
- **ROC-AUC 0.80-0.90:** Good discrimination
- **ROC-AUC 0.70-0.80:** Fair discrimination
- **ROC-AUC < 0.70:** Poor discrimination

### Feature Importance (Top 10 - Example)

```
1. ClaimAmount           : 0.234
2. PreviousClaims        : 0.187
3. Age                   : 0.156
4. PolicyDuration        : 0.142
5. VehicleAge            : 0.098
6. IncidentSeverity      : 0.067
7. NumberOfCars          : 0.045
8. Region_Urban          : 0.034
9. Gender_M              : 0.021
10. PolicyType_Comprehensive: 0.016
```

---

## 🔬 MLflow Tracking

### View MLflow UI

```bash
mlflow ui
```

Then open your browser to: `http://localhost:5000`

### MLflow Features Used

- **Experiment Tracking:** All 7 models tracked in one experiment
- **Parameter Logging:** Every hyperparameter combination
- **Metric Logging:** 12+ metrics per model
- **Model Registry:** Best models registered for deployment
- **Artifact Storage:** Pipelines, encoders, and feature info

### Querying Best Model via MLflow

```python
import mlflow

# Load best model from registry
model_name = "Fraud_XGBoost_Best"
model_version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

# Use for predictions
predictions = model.predict(test_data)
```

---

## 🛠️ Advanced Configuration

### Customizing the Pipeline

#### Change Number of Optimization Trials
```python
# In fraud_detection_complete.py, line ~440
study.optimize(
    obj_fn, 
    n_trials=50,  # Change from 30 to 50
    ...
)
```

#### Modify Cross-Validation Folds
```python
# In evaluate_model function, line ~270
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# Change from 5 to 10 folds
```

#### Add Custom Models
```python
def objective_custom_model(trial):
    """Your custom model objective"""
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    
    # Define your model and hyperparameters
    model = YourCustomClassifier(
        param1=trial.suggest_int('param1', 1, 10),
        param2=trial.suggest_float('param2', 0.01, 1.0, log=True)
    )
    
    return evaluate_model(create_pipeline(scaler_type, model))

# Add to objectives dictionary
objectives["CustomModel"] = objective_custom_model
```

#### Adjust Class Imbalance Handling
```python
# For XGBoost - modify scale_pos_weight calculation (line ~78)
scale_pos_weight = (neg_count / pos_count) * 2  # Increase penalty
```

---

## 📊 Interpreting Results

### When to Use Each Metric

| Metric | Use Case | Interpretation |
|--------|----------|----------------|
| **ROC-AUC** | Overall model discrimination | How well model separates classes |
| **Precision** | Cost of false positives high | Of flagged cases, how many are actual fraud |
| **Recall** | Cost of false negatives high | Of actual fraud, how many we catch |
| **F1-Score** | Balanced view needed | Harmonic mean of precision/recall |
| **Accuracy** | Balanced dataset only | Overall correct predictions |

### Business Impact Analysis

```python
# Example: Calculate potential savings
total_fraud_value = fraud_cases * avg_claim_amount
detected_fraud = total_fraud_value * recall
false_positives = legitimate_cases * (1 - precision)
investigation_cost = false_positives * cost_per_investigation

net_savings = detected_fraud - investigation_cost
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Error
```
Solution: Reduce n_trials or use a smaller dataset subset
```

#### 2. MLflow Tracking Error
```bash
# Clear mlruns directory
rm -rf mlruns/
# Restart the script
```

#### 3. Model Not Converging
```python
# Increase max_iter for Logistic Regression
max_iter = 5000  # Instead of 2000
```

#### 4. Slow Execution
```python
# Reduce cross-validation folds
n_splits = 3  # Instead of 5

# Reduce trials
n_trials = 10  # Instead of 30
```

---

## 🔄 Future Enhancements

### Planned Features
- [ ] Deep Learning models (Neural Networks)
- [ ] SHAP values for model interpretability
- [ ] Real-time fraud detection API
- [ ] Automated retraining pipeline
- [ ] Feature engineering automation
- [ ] Ensemble stacking methods
- [ ] Threshold optimization tool
- [ ] Production deployment scripts (Docker)
- [ ] Web dashboard for monitoring
- [ ] A/B testing framework

### Research Directions
- Graph-based fraud detection
- Anomaly detection methods
- Time-series patterns
- Network analysis of fraud rings
- Transfer learning from other domains

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Types of Contributions
- 🐛 Bug reports and fixes
- ✨ New features and models
- 📖 Documentation improvements
- 🧪 Test coverage
- 💡 Feature requests

### Contribution Process

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/insurance-fraud-detection.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests if applicable

4. **Commit Your Changes**
   ```bash
   git commit -m "Add: Amazing feature description"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/amazing-feature
   ```

### Code Style
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📧 Contact

**Project Maintainer:** [Your Name]

- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [@yourusername](https://github.com/yourusername)
- 🌐 Portfolio: [yourwebsite.com](https://yourwebsite.com)

**Project Link:** [https://github.com/yourusername/insurance-fraud-detection](https://github.com/yourusername/insurance-fraud-detection)

---

## 🙏 Acknowledgments

- **Scikit-learn:** Foundation for ML pipeline
- **XGBoost:** High-performance gradient boosting
- **Optuna:** Hyperparameter optimization framework
- **MLflow:** Experiment tracking and model registry
- **Pandas & NumPy:** Data manipulation
- Insurance industry datasets and research papers
- Open-source ML community

---

## 📚 References & Resources

### Research Papers
1. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
2. [Fraud Detection: A Systematic Review](link-to-paper)
3. [Handling Imbalanced Datasets](link-to-paper)

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Tutorials
- [Fraud Detection Best Practices](link)
- [MLOps for Production ML](link)
- [Hyperparameter Tuning Guide](link)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/insurance-fraud-detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/insurance-fraud-detection?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/insurance-fraud-detection)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/insurance-fraud-detection)

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star!**

Made with ❤️ for the ML community

[Report Bug](https://github.com/yourusername/insurance-fraud-detection/issues) · [Request Feature](https://github.com/yourusername/insurance-fraud-detection/issues)

</div>
