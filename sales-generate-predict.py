import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Sales Prediction App")
st.write("This app predicts the **Sales** !")

st.sidebar.header('User Input Parameters')

def user_input_features():
    # label, min, max, default
    tv = st.sidebar.slider('TV', 0.0, 300.0, 74.0)
    radio = st.sidebar.slider('Radio', 0.0, 4.4, 9.9)
    newspaper = st.sidebar.slider('Newspaper', 0.0, 25.0, 1.3)
    data = {'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper,}
    #take the first row
    features = pd.DataFrame(data, index=[0])
    return features

# hantar data yg user pilih ke df
df = user_input_features()

# subheader sama dgn dua ##
st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Sales-model-LR .h5", "rb"))
pred = loaded_model.predict(df)
pred_prob = modelGaussianIris.predict_proba(df)

st.subheader('Sales Prediction')
st.write(pred)

st.subheader('Prediction Probability')
st.write(pred_prob)
