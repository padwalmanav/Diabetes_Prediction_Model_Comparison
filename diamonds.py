import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load pickled models
pickle_in = open("log_reg_model.pkl", "rb")
log_reg = pickle.load(pickle_in)

pickle_in = open("knn_model.pkl", "rb")
knn = pickle.load(pickle_in)

pickle_in = open("dtree_model.pkl", "rb")
dtree = pickle.load(pickle_in)

pickle_in = open("rf_model.pkl", "rb")
rf = pickle.load(pickle_in)

# Sidebar for user input on a single page
st.sidebar.header('Input Features')
Pregnancies = st.sidebar.number_input("Enter Pregnancies(0-17)", min_value=0, max_value=17)
Glucose = st.sidebar.number_input("Enter Glucose(0-199)", min_value=0, max_value=199)
BloodPressure = st.sidebar.number_input("Enter BloodPressure(0-122)", min_value=0, max_value=122)
SkinThickness = st.sidebar.number_input("Enter SkinThickness(0-99)", min_value=0, max_value=99)
Insulin = st.sidebar.number_input("Enter Insulin(0-846)", min_value=0, max_value=846)
BMI = st.sidebar.number_input("Enter BMI(0-67.1)", min_value=0.0, max_value=67.1)
DiabetesPedigreeFunction = st.sidebar.number_input("Enter Diabetes Pedigree Function(0.078-2.42)", min_value=0.078, max_value=2.42)
Age = st.sidebar.number_input("Enter Age(21-81)", min_value=21, max_value=81)

# Create DataFrame for model input
user_df = pd.DataFrame({
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
    'Age': [Age]
})

# Add a "Predict" button
if st.sidebar.button("Predict"):
    # Make predictions with each model
    models = {'Logistic Regression': log_reg, 'KNN': knn, 'Decision Tree': dtree, 'Random Forest': rf}
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(user_df)[0]
        predictions[model_name] = "Has Diabetes" if prediction == 1 else "Doesn't have Diabetes"
    
    # Display predictions
    st.write("### Model Predictions for User Input")
    for model_name, prediction in predictions.items():
        st.write(f"{model_name} Prediction: {prediction}")
    
    # Assuming you have accuracy, precision, recall, and F1 score saved or calculated
    model_metrics = {
        'Logistic Regression': {'Accuracy': 0.88, 'Precision': 0.85, 'Recall': 0.86, 'F1 Score': 0.85},
        'KNN': {'Accuracy': 0.83, 'Precision': 0.82, 'Recall': 0.80, 'F1 Score': 0.81},
        'Decision Tree': {'Accuracy': 0.80, 'Precision': 0.78, 'Recall': 0.79, 'F1 Score': 0.78},
        'Random Forest': {'Accuracy': 0.90, 'Precision': 0.88, 'Recall': 0.87, 'F1 Score': 0.88}
    }

    # Display model comparison metrics
    st.write("### Model Comparison Metrics")
    metrics_df = pd.DataFrame(model_metrics).T
    st.write(metrics_df)

    # Plot comparison metrics
    st.write("### Comparison of Model Performance")
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax[row, col])
        ax[row, col].set_title(f'{metric} Comparison')
        ax[row, col].set_ylim(0, 1)

    st.pyplot(fig)
else:
    st.write("Enter values and click 'Predict' to see model predictions.")
