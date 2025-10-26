import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set page configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        text-align: center;
        color: #2e4053;
    }
    .stPlot {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>Employee Salary Prediction 💰</h1>", unsafe_allow_html=True)

# Load and cache the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Salary.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Visualization", "Salary Prediction"])

if page == "Data Overview":
    st.header("📊 Data Overview")
    
    # Display basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        st.metric("Average Salary", f"${df['Salary'].mean():,.2f}")
    with col3:
        st.metric("PhD Holders", f"{df['PhD'].sum()} ({(df['PhD'].sum()/len(df)*100):.1f}%)")
    
    # Display the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    # Display basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

elif page == "Data Visualization":
    st.header("📈 Data Visualization")
    
    # Salary Distribution
    st.subheader("Salary Distribution")
    try:
        fig = px.histogram(df, x='Salary', nbins=30, title="Distribution of Salaries")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating salary distribution plot: {e}")
    
    # Age vs Salary
    st.subheader("Age vs Salary")
    fig = px.scatter(df, x='Age', y='Salary', 
                    color='PhD', symbol='Gender',
                    labels={'PhD': 'Has PhD', 'Gender': 'Gender'},
                    title="Age vs Salary (with Gender and PhD)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Average Salary by Gender and PhD
    st.subheader("Average Salary by Gender and PhD")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.box(df, x='Gender', y='Salary', 
                     labels={'Gender': '', 'Salary': 'Salary'},
                     title="Salary Distribution by Gender")
        fig1.update_xaxes(ticktext=['Female', 'Male'], tickvals=[0, 1])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.box(df, x='PhD', y='Salary',
                     labels={'PhD': '', 'Salary': 'Salary'},
                     title="Salary Distribution by PhD")
        fig2.update_xaxes(ticktext=['No PhD', 'PhD'], tickvals=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

else:  # Salary Prediction
    st.header("🎯 Salary Prediction")
    
    # Load the saved model and scaler
    @st.cache_resource
    def load_model():
        try:
            with open('salary_model.pkl', 'rb') as file:
                model_data = pickle.load(file)
            return model_data['model'], model_data['scaler']
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    
    model, scaler = load_model()
    
    # Input form
    st.subheader("Enter Employee Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=80, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender = 1 if gender == "Male" else 0
    with col3:
        phd = st.selectbox("PhD", ["No", "Yes"])
        phd = 1 if phd == "Yes" else 0
    
    # Make prediction
    if st.button("Predict Salary"):
        # Scale the input data
        input_data = np.array([[age, gender, phd]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.success(f"Predicted Salary: ${prediction:,.2f}")
        
        # Show example predictions
        st.subheader("Prediction Analysis")
        
        # Create example data points
        example_data = pd.DataFrame({
            'Age': [25, 35, 45, 55],
            'Gender': [0, 1, 0, 1],
            'PhD': [0, 1, 1, 0]
        })
        example_scaled = scaler.transform(example_data)
        example_predictions = model.predict(example_scaled)
        
        # Display example predictions
        st.write("Example Predictions:")
        example_data['Predicted Salary'] = example_predictions
        example_data['Gender'] = example_data['Gender'].map({0: 'Female', 1: 'Male'})
        example_data['PhD'] = example_data['PhD'].map({0: 'No', 1: 'Yes'})
        st.dataframe(example_data)
        
        # Plot feature importance
        features = ['Age', 'Gender', 'PhD']
        importance = abs(model.coef_)
        fig = px.bar(x=features, y=importance,
                    title="Feature Importance",
                    labels={'x': 'Features', 'y': 'Absolute Coefficient Value'})
        st.plotly_chart(fig, use_container_width=True)