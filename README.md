# Credit Risk Classification: SVM vs. Neural Networks

## Overview
This project classifies loan applicants as **good or bad credit risk** using two machine learning approaches — Support Vector Machines (SVM) and Neural Networks. The goal is to compare how each algorithm handles a real-world credit scoring problem with mixed feature types and class imbalance.

This is a two-part analysis completed as coursework for **DATA 440 (Advanced Machine Learning)** at the University of Maryland Global Campus.

## Business Problem
Banks and lending institutions need reliable methods to assess whether a loan applicant is likely to repay or default. Poor credit decisions lead to financial losses, while overly conservative models reject creditworthy applicants and reduce revenue. An effective model must balance both risks.

## Dataset
- **Source:** [OpenML — German Credit Dataset](https://www.openml.org/d/31)
- **Size:** 1,000 loan applicants
- **Features:** 20 attributes including credit history, loan purpose, employment duration, savings, housing status, and existing credits
- **Target:** Binary classification (good credit vs. bad credit)
- **Class distribution:** 70% good credit, 30% bad credit

## Methods

### Part 1: Support Vector Machines (SVM)
- Exploratory data analysis on categorical and numeric features
- Label encoding and feature preprocessing
- SVM classifier with kernel comparison
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix

### Part 2: Neural Networks
- Feed-forward neural network built with TensorFlow/Keras
- One-hot encoding for categorical variables, standard scaling for numeric features
- Architecture tuning (hidden layers, neurons, activation functions)
- Training with early stopping to prevent overfitting
- Direct performance comparison with SVM results

## Key Findings
- Both models performed reasonably well on this relatively small dataset
- The **class imbalance** (70/30 split) affected both models' ability to correctly identify bad credit applicants
- Feature preprocessing choices (encoding strategy, scaling) had a measurable impact on model performance
- Neural networks required more careful tuning but offered flexibility in handling mixed feature types

## Tools & Libraries
- Python (pandas, NumPy, scikit-learn, TensorFlow/Keras, matplotlib, seaborn)
- Jupyter Notebook

## Files
- `Data440_Assignment_1_SVM.ipynb` — SVM classification analysis
- `Data440_Assignment_2_Neural_Networks.ipynb` — Neural network analysis and model comparison

## Author
**Julio Patiño**
- [LinkedIn](https://www.linkedin.com/in/juliopatino88/)
- [GitHub](https://github.com/juliopatino88)
