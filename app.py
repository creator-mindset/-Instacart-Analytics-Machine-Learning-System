import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(filename):
    """Return the full path to a data/model file, or raise a clear
    Streamlit error (instead of a raw traceback) if it's missing."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(
            f"Required file `{filename}` was not found next to app.py "
            f"(looked in: {BASE_DIR}). Make sure it's committed to the "
            f"repo, the filename/case matches exactly, and it isn't "
            f"excluded via .gitignore or stuck in Git LFS."
        )
        st.stop()
    return path

st.set_page_config(
    page_title="Instacart Analytics & Machine Learning System",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

[data-testid="stSidebar"]{
    background-color:#F4F6F7;
}

.metric-container{
    background:#ffffff;
    padding:15px;
    border-radius:10px;
}

h1{
    color:#1F618D;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
# NOTE: only kmeans_df is kept — it's the source data Segmentation
# clusters and searches against, not just a preview table.
# regressor_dataset.csv and classifier_dataset.csv are no longer
# loaded since they were only used for dataset previews/histograms.

@st.cache_data
def load_kmeans():
    return pd.read_csv(resolve_path("K-means Dataset.csv"))

# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    regressor = joblib.load(
        resolve_path("random_forest_regressor.pkl")
    )

    classifier = joblib.load(
        resolve_path("random_forest_classifier.pkl")
    )

    kmeans = joblib.load(
        resolve_path("kmeans_model.pkl")
    )

    return regressor, classifier, kmeans


kmeans_df = load_kmeans()

regressor_model, classifier_model, kmeans_model = load_models()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Module",

    [
        "🏠 Home",
        "🤖 Random Forest Regressor",
        "🛒 Random Forest Classifier",
        "👥 Customer Segmentation",
        "📄 About"
    ]

)

# ==========================================================
# HOME
# ==========================================================

if page == "🏠 Home":

    st.title("🛒 Instacart Analytics & Machine Learning System")

    st.write("""
Welcome to the end-to-end Data Science Project.

### Technologies

- PostgreSQL
- SQL
- Python
- Pandas
- Scikit-Learn
- Random Forest
- K-Means
- Streamlit

### Project Modules

✔ Business Analytics

✔ Machine Learning

✔ Customer Segmentation

✔ Interactive Dashboard

""")

    st.image(
        "https://images.unsplash.com/photo-1556740749-887f6717d7e4?w=1200",
        use_container_width=True
    )

# ==========================================================
# RANDOM FOREST REGRESSOR
# ==========================================================

elif page == "🤖 Random Forest Regressor":

    st.title("🤖 Random Forest Regressor")

    st.write("Predict the expected number of days until the customer's next order.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        user_id = st.number_input(
            "User ID",
            min_value=1,
            step=1
        )

        order_number = st.number_input(
            "Order Number",
            min_value=1,
            step=1
        )

    with col2:

        order_dow = st.selectbox(
            "Order Day of Week",
            [0, 1, 2, 3, 4, 5, 6]
        )

        order_hour = st.slider(
            "Order Hour",
            0,
            23,
            12
        )

    if st.button("Predict Next Order"):

        input_data = pd.DataFrame({
            "user_id": [user_id],
            "order_number": [order_number],
            "order_dow": [order_dow],
            "order_hour_of_day": [order_hour]
        })

        prediction = regressor_model.predict(input_data)

        st.success(
            f"Predicted Days Until Next Order: {prediction[0]:.2f} Days"
        )

# ==========================================================
# RANDOM FOREST CLASSIFIER
# ==========================================================

elif page == "🛒 Random Forest Classifier":

    st.title("🛒 Random Forest Classifier")

    st.write("Predict whether a customer will reorder a product.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        user_id = st.number_input(
            "User ID",
            min_value=1,
            step=1,
            key="clf_user"
        )

        product_id = st.number_input(
            "Product ID",
            min_value=1,
            step=1,
            key="clf_product"
        )

        add_to_cart_order = st.number_input(
            "Add To Cart Position",
            min_value=1,
            step=1,
            key="clf_cart"
        )

    with col2:

        order_number = st.number_input(
            "Order Number",
            min_value=1,
            step=1,
            key="clf_order"
        )

        order_dow = st.selectbox(
            "Order Day",
            [0, 1, 2, 3, 4, 5, 6],
            key="clf_day"
        )

        order_hour = st.slider(
            "Order Hour",
            0,
            23,
            12,
            key="clf_hour"
        )

    if st.button("Predict Reorder"):

        input_data = pd.DataFrame({
            "user_id": [user_id],
            "product_id": [product_id],
            "add_to_cart_order": [add_to_cart_order],
            "order_number": [order_number],
            "order_dow": [order_dow],
            "order_hour_of_day": [order_hour]
        })

        prediction = classifier_model.predict(input_data)

        probability = classifier_model.predict_proba(input_data)

        if prediction[0] == 1:

            st.success("✅ Customer is likely to reorder this product.")

        else:

            st.error("❌ Customer is unlikely to reorder this product.")

        st.write(
            f"Prediction Confidence: {max(probability[0]) * 100:.2f}%"
        )

    st.markdown("---")

    st.subheader("Feature Importance")

    # Feature names pulled from the trained model itself, so this no
    # longer depends on classifier_dataset.csv being present.
    feature_names = getattr(
        classifier_model,
        "feature_names_in_",
        [f"Feature {i}" for i in range(len(classifier_model.feature_importances_))]
    )

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier_model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig2 = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================================================
# CUSTOMER SEGMENTATION
# ==========================================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.write("Customer Segmentation using K-Means Clustering")

    st.markdown("---")

    features = kmeans_df.drop(columns=["user_id"])

    kmeans_df["Cluster"] = kmeans_model.predict(features)

    cluster_names = {
        0: "Loyal Customers",
        1: "High Value Customers",
        2: "Occasional Buyers",
        3: "At Risk Customers",
        4: "Bulk Buyers"
    }

    kmeans_df["Segment"] = kmeans_df["Cluster"].map(cluster_names)

    st.subheader("Customer Segmentation Dataset")

    st.dataframe(
        kmeans_df.head(20),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Customer Segment Distribution")

    pie = px.pie(
        kmeans_df,
        names="Segment",
        title="Customer Segments"
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Customer Behaviour")

    scatter = px.scatter(

        kmeans_df,

        x="total_orders",

        y="total_products",

        color="Segment",

        size="total_reorders",

        hover_data=["user_id"],

        title="Customer Segmentation Analysis"

    )

    st.plotly_chart(
        scatter,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Average Metrics by Segment")

    summary = kmeans_df.groupby("Segment")[
        [
            "total_orders",
            "total_products",
            "avg_cart_position",
            "avg_days_between_orders",
            "total_reorders"
        ]
    ].mean()

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Search Customer")

    search_user = st.number_input(
        "Enter User ID",
        min_value=1,
        step=1
    )

    if st.button("Search Customer"):

        result = kmeans_df[
            kmeans_df["user_id"] == search_user
        ]

        if len(result) > 0:

            st.success("Customer Found")

            st.dataframe(result)

        else:

            st.error("Customer Not Found")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

elif page == "📄 About":

    st.title("📄 About This Project")

    st.markdown("""
# Instacart Analytics & Machine Learning System

This is an End-to-End Data Science Project built using the
Instacart Market Basket Analysis Dataset.

The project combines SQL, Machine Learning, Data Analytics,
Business Intelligence and Interactive Visualization.

---
""")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🛠 Technologies")

        st.markdown("""
- PostgreSQL
- SQL
- Python
- Pandas
- NumPy
- Scikit-Learn
- Random Forest
- K-Means
- Plotly
- Streamlit
""")

    with col2:

        st.subheader("📚 Machine Learning")

        st.markdown("""
- Random Forest Regressor
- Random Forest Classifier
- K-Means Clustering
""")

    st.markdown("---")

    st.subheader("📊 Business Objectives")

    st.markdown("""
✔ Customer Behaviour Analysis

✔ Product Reorder Prediction

✔ Next Order Prediction

✔ Customer Segmentation

✔ Business Intelligence Dashboard

✔ Business Insights using SQL
""")

    st.markdown("---")

    st.subheader("📈 Dataset Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Customers",
        kmeans_df["user_id"].nunique()
    )

    c2.metric(
        "Orders",
        int(kmeans_df["total_orders"].sum())
    )

    c3.metric(
        "Products",
        int(kmeans_df["total_products"].sum())
    )

    st.markdown("---")

    st.subheader("👨‍💻 Project Workflow")

    st.markdown("""

PostgreSQL Database

⬇

Advanced SQL Queries

⬇

Feature Engineering

⬇

Random Forest Models

⬇

K-Means Clustering

⬇

Streamlit Dashboard

""")

    st.markdown("---")

    st.success("End-to-End Data Science Project Successfully Developed ✅")


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;'>

Made with ❤️ using

<b>Python | PostgreSQL | SQL | Scikit-Learn | Streamlit</b>

</div>
""",
unsafe_allow_html=True
)
