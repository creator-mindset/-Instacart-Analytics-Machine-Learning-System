import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Instacart Analytics System",
    layout="wide"
)

st.title("🛒 Instacart Analytics & Machine Learning System")

st.sidebar.title("Navigation")

page=st.sidebar.radio(
"Select Module",
[
"Home",
"Business Dashboard",
"SQL Insights",
"Sales Prediction",
"Reorder Prediction",
"Customer Segmentation",
"About Project"
]
)

if page=="Home":

    st.header("Project Overview")

    st.write("""
This project combines

✅ PostgreSQL

✅ SQL

✅ Random Forest Regressor

✅ Random Forest Classifier

✅ K-Means Clustering

to analyze Instacart customer behaviour.
""")
    
elif page=="Business Dashboard":

    df=pd.read_csv("customer_segments.csv")

    c1,c2,c3=st.columns(3)

    c1.metric("Customers",df.shape[0])

    c2.metric("Average Orders",
              round(df["total_orders"].mean(),2))

    c3.metric("Average Products",
              round(df["total_products"].mean(),2))

    fig=px.histogram(
        df,
        x="total_orders",
        title="Orders Distribution"
    )

    st.plotly_chart(fig)

elif page=="SQL Insights":

    df=pd.read_csv("customer_segments.csv")

    st.subheader("Top Customers")

    top=df.sort_values(
        "total_orders",
        ascending=False
    ).head(20)

    st.dataframe(top)

elif page=="Sales Prediction":

    model=joblib.load(
        "random_forest_regressor.pkl"
    )

    st.subheader("Predict Days Until Next Order")

    user_id=st.number_input("User ID",1)

    order_number=st.number_input("Order Number",1)

    dow=st.slider("Day",0,6)

    hour=st.slider("Hour",0,23)

    if st.button("Predict"):

        pred=model.predict(
        [[
        user_id,
        order_number,
        dow,
        hour
        ]]
        )

        st.success(
        f"Expected Next Order After {round(pred[0],2)} Days"
        )

    elif page=="Reorder Prediction":

    model=joblib.load(
        "random_forest_classifier.pkl"
    )

    user=st.number_input("User",1)

    product=st.number_input("Product",1)

    cart=st.number_input("Cart Position",1)

    order=st.number_input("Order Number",1)

    dow=st.slider("Day",0,6)

    hour=st.slider("Hour",0,23)

    if st.button("Predict Reorder"):

        pred=model.predict(
        [[
        user,
        product,
        cart,
        order,
        dow,
        hour
        ]]
        )

        if pred[0]==1:

            st.success("Customer Will Reorder")

        else:

            st.error("Customer Will Not Reorder")

elif page=="Customer Segmentation":

    df=pd.read_csv("customer_segments.csv")

    model=joblib.load(
        "kmeans_model.pkl"
    )

    df["Cluster"]=model.labels_

    fig=px.scatter(
        df,
        x="total_orders",
        y="total_products",
        color="Cluster",
        title="Customer Segments"
    )

    st.plotly_chart(fig)

elif page=="About Project":

    st.header("Project Information")

    st.write("""

Technology Used

• PostgreSQL

• SQL

• Python

• Pandas

• Random Forest

• K-Means

• Streamlit

Business Goal

Understand customer behaviour and improve business decisions.

""")
