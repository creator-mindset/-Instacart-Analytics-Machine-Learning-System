import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Instacart Analytics & Machine Learning System",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------

st.title("🛒 Instacart Analytics & Machine Learning System")

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🤖 Random Forest Regressor",
        "🛒 Random Forest Classifier",
        "👥 Customer Segmentation",
        "ℹ About Project"
    ]
)

# ===================================================
# HOME PAGE
# ===================================================

if page == "🏠 Home":

    st.header("Project Overview")

    st.write("""
This project analyzes customer purchasing behaviour using the
Instacart Market Basket Analysis Dataset.

### Technologies Used

- PostgreSQL
- SQL
- Python
- Pandas
- Scikit-Learn
- Random Forest
- K-Means
- Plotly
- Streamlit

### Machine Learning Models

- Random Forest Regressor
- Random Forest Classifier
- K-Means Clustering

### Business Objectives

✔ Customer Behaviour Analysis

✔ Product Reorder Prediction

✔ Customer Segmentation

✔ Business Intelligence Dashboard
""")

# ===================================================
# DASHBOARD
# ===================================================

elif page == "📊 Dashboard":

    st.header("Business Dashboard")

    df = pd.read_csv("customer_segments.csv")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        len(df)
    )

    col2.metric(
        "Average Orders",
        round(df["total_orders"].mean(), 2)
    )

    col3.metric(
        "Average Products",
        round(df["total_products"].mean(), 2)
    )

    st.markdown("---")

    fig = px.histogram(
        df,
        x="total_orders",
        nbins=30,
        title="Distribution of Customer Orders"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="total_orders",
        y="total_products",
        title="Orders vs Products"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Dataset")

    st.dataframe(df.head(20))
# ===================================================
# RANDOM FOREST REGRESSOR
# ===================================================

elif page == "🤖 Random Forest Regressor":

    st.header("Random Forest Regressor")

    st.write(
        "Predict the number of days until the customer's next order."
    )

    model = joblib.load("random_forest_regressor.pkl")

    user_id = st.number_input(
        "User ID",
        min_value=1,
        value=1
    )

    order_number = st.number_input(
        "Order Number",
        min_value=1,
        value=1
    )

    order_dow = st.selectbox(
        "Order Day",
        [0,1,2,3,4,5,6]
    )

    order_hour = st.slider(
        "Order Hour",
        0,
        23,
        12
    )

    if st.button("Predict Next Order"):

        prediction = model.predict(
            [[
                user_id,
                order_number,
                order_dow,
                order_hour
            ]]
        )

        st.success(
            f"Expected Next Order After {prediction[0]:.2f} Days"
        )

# ===================================================
# RANDOM FOREST CLASSIFIER
# ===================================================

elif page == "🛒 Random Forest Classifier":

    st.header("Random Forest Classifier")

    st.write(
        "Predict whether the customer will reorder the product."
    )

    model = joblib.load("random_forest_classifier.pkl")

    user_id = st.number_input(
        "User ID ",
        min_value=1,
        value=1
    )

    product_id = st.number_input(
        "Product ID",
        min_value=1,
        value=1
    )

    add_to_cart = st.number_input(
        "Add To Cart Order",
        min_value=1,
        value=1
    )

    order_number = st.number_input(
        "Order Number ",
        min_value=1,
        value=1
    )

    order_dow = st.selectbox(
        "Order Day ",
        [0,1,2,3,4,5,6]
    )

    order_hour = st.slider(
        "Order Hour ",
        0,
        23,
        12
    )

    if st.button("Predict Reorder"):

        prediction = model.predict(
            [[
                user_id,
                product_id,
                add_to_cart,
                order_number,
                order_dow,
                order_hour
            ]]
        )

        if prediction[0] == 1:

            st.success(
                "Customer is likely to reorder this product."
            )

        else:

            st.error(
                "Customer is unlikely to reorder this product."
            )
  
# ===================================================
# CUSTOMER SEGMENTATION
# ===================================================

elif page == "👥 Customer Segmentation":

    st.header("Customer Segmentation using K-Means")

    df = pd.read_csv("customer_segments.csv")

    model = joblib.load("kmeans_model.pkl")

    df["Cluster"] = model.labels_

    segment_map = {
        0: "Loyal Customers",
        1: "High Value Customers",
        2: "Occasional Buyers",
        3: "At Risk Customers",
        4: "Bulk Buyers"
    }

    df["Segment"] = df["Cluster"].map(segment_map)

    st.subheader("Customer Segments")

    st.dataframe(df.head(20))

    fig = px.scatter(
        df,
        x="total_orders",
        y="total_products",
        color="Segment",
        hover_data=["user_id"],
        title="Customer Segmentation"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Distribution")

    fig2 = px.pie(
        df,
        names="Segment",
        title="Customer Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)


# ===================================================
# ABOUT PROJECT
# ===================================================

elif page == "ℹ About Project":

    st.header("About Project")

    st.markdown("""
## Instacart Analytics & Machine Learning System

### Dataset
Instacart Market Basket Analysis Dataset

### Database
- PostgreSQL

### SQL Concepts
- JOIN
- GROUP BY
- HAVING
- CTE
- Window Functions
- Business Analytics

### Machine Learning
- Random Forest Regressor
- Random Forest Classifier
- K-Means Clustering

### Python Libraries
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Joblib
- Streamlit

### Business Objectives

✔ Customer Analytics

✔ Product Analytics

✔ Customer Segmentation

✔ Reorder Prediction

✔ Next Order Prediction

✔ Interactive Dashboard

---

Developed by Prajjwal Bisht
""")

# ===================================================
# FOOTER
# ===================================================

st.markdown("---")

st.caption(
    "Instacart Analytics & Machine Learning System | Data Science Portfolio Project"
)
