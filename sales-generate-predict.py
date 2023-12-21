import streamlit as st
import pandas as pd
import seaborn as sns

st.write("# Sales Prediction App")
st.write("This app predicts the **Sales** !")

st.sidebar.header('User Input Parameters')

#load model, set cache to prevent reloading
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('models/Sales-model-LR.h5')
    return model

with st.spinner("Loading Model...."):
    model=load_model()

def user_input_features():
    # label, min, max, default
    tv = st.sidebar.slider('TV', 0.0, 300.0, 74.0)
    radio = st.sidebar.slider('Radio', 0.0, 4.4, 9.9)
    newspaper = st.sidebar.slider('Newspaper', 0.0, 25.0, 1.3)
    data = {'tv': tv,
            'radio': radio,
            'newspaper': newspaper,}
    #take the first row
    features = pd.DataFrame(data, index=[0])
    return features

# hantar data yg user pilih ke df
df = user_input_features()

# subheader sama dgn dua ##
st.subheader('User Input parameters')
st.write(df)

prediction = load_model(df)
st.write(prediction)

st.subheader('Prediction')
pred = model.predict(df)
st.write(pred)
