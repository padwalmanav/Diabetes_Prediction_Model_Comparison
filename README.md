# Diabetes Prediction Models Comparison 

This project involves predicting diabetes outcomes using various machine learning algorithms. The objective was to preprocess the data, build predictive models, and evaluate their performance using key metrics.

## Features
- Data preprocessing with test set splitting and feature scaling.
- Visualization of evaluation metrics using Matplotlib to compare model performances.
- Implementation of multiple machine learning algorithms including:
  - Decision Trees
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
- Performance evaluation using metrics like accuracy, precision, and recall.

## Results
- Achieved an average accuracy of **80%** using the AdaBoost algorithm.
- Comprehensive comparison of models based on training and test datasets.

## Setup and Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/padwalmanav/Diabetes_Prediction_Models_Comparison.git
    cd Diabetes_Prediction_Models_Comparison
    ```

2. Install the required dependencies and libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit:
    ```bash
    streamlit run diabetes_prediction.py
    ```

## Visualizations
Matplotlib was used to plot evaluation metrics, allowing for a detailed comparison of model performances.

## Key Takeaways
- Preprocessing the data, including feature scaling, significantly improved model performance.
- Random Forest outperformed other models, providing a balance between accuracy, precision, and recall.
