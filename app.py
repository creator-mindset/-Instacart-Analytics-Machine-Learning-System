import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Instacart Analytics & Machine Learning System",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}. Make sure it's in the app's working directory.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Make sure it's in the app's working directory.")
        st.stop()
    return joblib.load(path)


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

    df = load_csv("customer_segments.csv")

    required_cols = {"total_orders", "total_products"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"customer_segments.csv is missing expected columns: {missing}")
        st.stop()

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

    model = load_model("random_forest_regressor.pkl")

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
        [0, 1, 2, 3, 4, 5, 6]
    )

    order_hour = st.slider(
        "Order Hour",
        0,
        23,
        12
    )

    if st.button("Predict Next Order"):

        input_df = pd.DataFrame(
            [[user_id, order_number, order_dow, order_hour]],
            columns=["user_id", "order_number", "order_dow", "order_hour"]
        )

        try:
            prediction = model.predict(input_df)
            st.success(
                f"Expected Next Order After {prediction[0]:.2f} Days"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===================================================
# RANDOM FOREST CLASSIFIER
# ===================================================

elif page == "🛒 Random Forest Classifier":

    st.header("Random Forest Classifier")

    st.write(
        "Predict whether the customer will reorder the product."
    )

    model = load_model("random_forest_classifier.pkl")

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
        [0, 1, 2, 3, 4, 5, 6]
    )

    order_hour = st.slider(
        "Order Hour ",
        0,
        23,
        12
    )

    if st.button("Predict Reorder"):

        input_df = pd.DataFrame(
            [[user_id, product_id, add_to_cart, order_number, order_dow, order_hour]],
            columns=["user_id", "product_id", "add_to_cart_order", "order_number", "order_dow", "order_hour"]
        )

        try:
            prediction = model.predict(input_df)

            if prediction[0] == 1:
                st.success(
                    "Customer is likely to reorder this product."
                )
            else:
                st.error(
                    "Customer is unlikely to reorder this product."
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===================================================
# CUSTOMER SEGMENTATION
# ===================================================

elif page == "👥 Customer Segmentation":

    st.header("Customer Segmentation using K-Means")

    df = load_csv("customer_segments.csv")
    model = load_model("kmeans_model.pkl")

    required_cols = {"total_orders", "total_products"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"customer_segments.csv is missing expected columns: {missing}")
        st.stop()

    # BUG FIX: model.labels_ reflects the training data the KMeans model
    # was originally fit on, not this df. It can mismatch in length or
    # simply mislabel rows. Use model.predict() on the actual feature
    # columns instead so clusters are computed for the data being shown.
    feature_cols = ["total_orders", "total_products"]
    try:
        df["Cluster"] = model.predict(df[feature_cols])
    except Exception as e:
        st.error(f"Clustering prediction failed: {e}")
        st.stop()

    segment_map = {
        0: "Loyal Customers",
        1: "High Value Customers",
        2: "Occasional Buyers",
        3: "At Risk Customers",
        4: "Bulk Buyers"
    }

    df["Segment"] = df["Cluster"].map(segment_map).fillna("Unclassified")

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
