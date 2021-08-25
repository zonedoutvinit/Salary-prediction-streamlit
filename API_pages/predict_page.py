import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('Add ML Model', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",                          
        "India",                  
        "United Kingdom",         
        "Germany",                
        "Canada",                 
        "France",                 
        "Brazil",                 
        "Australia",              
        "Spain",                  
        "Netherlands",            
        "Poland",                 
        "Russian Federation",     
        "Italy",                  
        "Sweden",                 
        "Israel",                 
        "Turkey",                  
        "Switzerland",             
        "Ukraine",             
        "Mexico",                  
        "Norway",                 
        "Pakistan",                
        "Belgium",                 
        "Austria",                 
        "Romania",                 
        "South Africa",            
        "Czech Republic",          
        "Ireland",                 
        "Denmark",                 
        "Portugal",                
        "Finland",                 
        "Iran",                    
        "Bulgaria",                
        "Argentina",               
        "New Zealand",             
        "Hungary",                 
        "Greece"
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary in dollars is ${salary[0]:.2f}")
        usd = salary
        inr = usd * 73
        st.subheader(f"The estimated salary in rupees is ₹{inr[0]:.2f}")
    
