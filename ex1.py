import streamlit as st
import pandas as pd
import pickle

st.write("""
### Hello!
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')

def get_input():
    #widgets
    v_Sex = st.sidebar.radio('Sex', ['Male','Female','Infant'])
    v_Length = st.sidebar.slider('Length', 0.0750,0.7450,0.5067)
    v_Diameter = st.sidebar.slider('Diameter', 0.0550, 0.6000,0.4006)
    v_Height = st.sidebar.slider('Diameter', 0.0100, 0.2400,0.1388)
    v_Whole_weight = st.sidebar.slider('Whole_weight', 0.0020, 2.5500,0.7851)
    v_Shucked_weight = st.sidebar.slider('Shucked_weight', 0.0010, 1.0705,0.3089)
    v_Viscera_weight = st.sidebar.slider('Viscera_weight', 0.0005, 0.5410,0.1702)
    v_Shell_weight = st.sidebar.slider('Shell_weight', 0.0015, 1.0050,0.2491)
    if v_Sex == 'Male': v_Sex = 'M'
    elif v_Sex == 'Female': v_Sex = 'F'
    else: v_Sex = 'I'

    #dictionary
    data = {'Sex': v_Sex,
            'Length': v_Length,
            'Diameter': v_Diameter,
            'Height': v_Height,
            'Whole_weight': v_Whole_weight,
            'Shucked_weight': v_Shucked_weight,
            'Viscera_weight': v_Viscera_weight,
            'Shell_weight': v_Shell_weight}


    #create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()
st.write(df)

data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)

cat_data = pd.get_dummies(df[['Sex']])

X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)
X_new = X_new.drop(columns=['Sex'])
# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)