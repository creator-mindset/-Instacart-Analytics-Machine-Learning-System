import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Instacart Analysis and ML System", layout="wide")

regressor = joblib.load("random_forest_regressor.pkl")
classifier = joblib.load("random_forest_classifier.pkl")
kmeans = joblib.load("kmeans_model.pkl")


def get_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return [f"feature_{i+1}" for i in range(model.n_features_in_)]


st.title("Instacart Analysis and Machine Learning System")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Regressor", "Classifier", "KMeans"])

with tab1:
    st.header("Project Overview")
    st.write("This dashboard uses three trained models on Instacart data.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Regressor Trees", regressor.n_estimators)
    col2.metric("Classifier Trees", classifier.n_estimators)
    col3.metric("KMeans Clusters", kmeans.n_clusters)

with tab2:
    st.header("Random Forest Regressor")
    features = get_features(regressor)
    inputs = []
    for f in features:
        inputs.append(st.number_input(f, value=0.0, key="reg_" + f))
    if st.button("Predict", key="reg_btn"):
        X = pd.DataFrame([inputs], columns=features)
        result = regressor.predict(X)[0]
        st.success(f"Prediction: {result:.2f}")

with tab3:
    st.header("Random Forest Classifier")
    features = get_features(classifier)
    inputs = []
    for f in features:
        inputs.append(st.number_input(f, value=0.0, key="clf_" + f))
    if st.button("Predict", key="clf_btn"):
        X = pd.DataFrame([inputs], columns=features)
        result = classifier.predict(X)[0]
        st.success(f"Predicted Class: {result}")

with tab4:
    st.header("KMeans Clustering")
    features = get_features(kmeans)
    inputs = []
    for f in features:
        inputs.append(st.number_input(f, value=0.0, key="km_" + f))
    if st.button("Predict Cluster", key="km_btn"):
        X = pd.DataFrame([inputs], columns=features)
        result = kmeans.predict(X)[0]
        st.success(f"Cluster: {result}")
