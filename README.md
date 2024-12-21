# Fraud Detection in Insurance Claims

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![Dataset Size](https://img.shields.io/badge/Dataset-15%2C900%20rows-yellow)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-95.67%25-blue)
![PR-AUC](https://img.shields.io/badge/PR--AUC-73.36%25-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Project Status](https://img.shields.io/badge/Project-Completed-brightgreen)

## 🌟 Overview

Fraud detection is a significant challenge in industries such as insurance, where fraudulent claims result in financial losses and impact operational efficiency. This project demonstrates an end-to-end pipeline for identifying fraudulent insurance claims using a single comprehensive Colab notebook.

### Key Highlights:
- **Domain:** Auto Insurance Fraud Detection  
- **Objective:** Classify claims as fraudulent or legitimate.  
- **Techniques:** Ensemble learning (XGBoost, LightGBM, CatBoost) and stacking ensembles.  
- **Challenges:** Handling imbalanced data, avoiding overfitting, and ensuring interpretability.


---

## 📚 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Dataset](#-dataset)
3. [Methodology](#-methodology)
4. [Key Results](#-key-results)
5. [Tech Stack](#-tech-stack)
6. [How to Run](#-how-to-run)
7. [Known Issues and Future Improvements](#-Known-Issues-and-Future-Improvements)

---

## 🖋️ Problem Statement

Fraudulent activities in insurance claims are often rare and complex to detect. This dataset focuses on auto insurance fraud, with each instance representing an individual claim. The task is to classify claims as either fraudulent or legitimate based on 31 features, including customer demographics, policy details, and claim history.

### Challenges:
- Imbalanced dataset where fraudulent claims are a small fraction of the total data.
- High stakes in both false negatives (missed fraud cases) and false positives (legitimate claims flagged as fraud).
- Ensuring that the results are interpretable for business stakeholders.

---

## 📂 Dataset

The dataset consists of 15,900 claims split into a training set (3,000 instances) and a test set (12,900 instances), with 31 input features and one target variable, `FRAUDFOUND` (Yes/No).

### Feature Overview

| Feature Type         | Examples                                   |
|----------------------|-------------------------------------------|
| **Temporal Features** | MONTH, WEEKOFMONTH, DAYOFWEEK, YEAR        |
| **Categorical Features** | MAKE, SEX, POLICYTYPE, ACCIDENTAREA, BASEPOLICY |
| **Numerical Features** | AGE, DEDUCTIBLE, DRIVERRATING, REPNUMBER  |
| **Target Variable**   | FRAUDFOUND (Yes: Fraudulent, No: Legitimate Claim) |

---

## 📔 Methodology

This project is implemented entirely in a single Colab notebook, which follows the steps below:

### 1. Data Preprocessing
- Handled missing values using simple imputation strategies.
- Encoded categorical variables using label encoding and one-hot encoding.
- Scaled numerical variables to ensure compatibility with tree-based models.
- Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique) and SMOTETomek.

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions and relationships between key features.
- Investigated correlations and identified important patterns indicative of fraud.

### 3. Model Development
- Trained multiple classifiers:
  - XGBoost
  - LightGBM
  - CatBoost
- Built a stacking ensemble for enhanced predictive performance.
- Optimized hyperparameters using Grid Search, Random Search, and Bayesian Optimization.

### 4. Evaluation
- Evaluated models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - PR-AUC
- Plotted ROC and Precision-Recall Curves for visual insights.

### 5. Threshold Adjustment
- Fine-tuned the decision threshold to balance precision and recall for business-specific needs.

---

## 📊 Key Results

| Metric          | Value   |
|------------------|---------|
| **Accuracy**     | 97.10%  |
| **Precision**    | 67.69%  |
| **Recall**       | 69.84%  |
| **F1 Score**     | 68.75%  |
| **ROC-AUC**      | 95.67%  |
| **PR-AUC**       | 73.36%  |

### Visual Results:
- **ROC Curve:**

- **Precision-Recall Curve:**

The stacking ensemble model achieved the best performance, balancing recall (important for fraud detection) with precision (to minimize false positives).

---

## 💻 Tech Stack

- **Platform:** Google Colab
- **Programming Language:** Python
- **Libraries:**
  - **Data Preprocessing:** pandas, numpy, imblearn  
  - **Visualization:** matplotlib, seaborn, shap  
  - **Model Training:** scikit-learn, xgboost, lightgbm, catboost  

---

## ⚙ How to Run

Follow these steps to run the project:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-project.git
   cd fraud-detection-project
   ```

2. **Upload the Dataset:**
   Place the raw dataset files in the `data/raw/` folder or upload them directly to the Colab notebook.

3. **Open the Notebook:**
   Open the `fraud_detection_colab.ipynb` file in Google Colab.

4. **Install Dependencies:**
   Run the following command in a Colab cell:
   ```bash
   !pip install -r requirements.txt
   ```

5. **Run the Notebook:**
   Execute each cell step-by-step to preprocess the data, train models, evaluate metrics, and visualize results.

---

## 🚀 Known Issues and Future Improvements

### Known Issues
- The model's performance may degrade on unseen data without regular updates.
- Current feature engineering is based on assumptions and could be improved with domain expertise.

### Future Improvements
- **Automated Feature Engineering:** Use tools like Featuretools to extract more meaningful features.
- **Anomaly Detection:** Incorporate techniques like Isolation Forest for better fraud pattern recognition.
- **Retraining Pipeline:** Set up an automated pipeline for periodic retraining using fresh data.

---

## 💾 Project Structure

```
fraud-detection-project/
├── data/
│   ├── train.csv                   # Training dataset 
│   ├── test.csv                    # Test datasets
├── notebook/
│   └── fraud_detection_colab.ipynb # Complete project in a single Colab notebook
├── results/
│   ├── roc_curve.png               # ROC Curve Visualization
│   └── pr_curve.png                # Precision-Recall Curve Visualization
├── reports/
│   ├── presentation.pptx           # Presentation slides
│   └── recording.mp4               # Video explanation
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
├── LICENSE                         # License information
└── .gitignore                      # Files to ignore
```
## Folder and File Descriptions

### **Folders**

- `data/`: Contains datasets.
  - `train.csv`: Training dataset.
  - `test.csv`: Test Dataset for predictions.

- `notebook/`: Includes the main Colab notebook with the entire pipeline.
  - `fraud_detection_colab.ipynb`: Comprehensive notebook covering data preprocessing, model training, evaluation, and visualization.

- `results/`: Stores output plots.
  - `roc_curve.png`: ROC Curve visualization for the best model.
  - `pr_curve.png`: Precision-Recall Curve visualization for the best model.

- `reports/`: Contains the final presentation slides.
  - `recording.mp4`: Detailed Video explanation of the project, including methodology and results.
  - `presentation.pptx`: Slide deck summarizing the project.

### **Key Files**

- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: This file, providing an overview of the project and instructions for replication.

---

