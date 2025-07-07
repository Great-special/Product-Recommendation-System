import streamlit as st
import pandas as pd
from recommendation import load_prepared_data, train_predict_cluster, transform_data, recommend_for_user


# --- Set Streamlit Page Config ---
st.set_page_config(page_title="Recommendation App", layout="wide")

# --- Load Data ---
data_df = pd.read_csv('data/amazon.csv')
data_df.loc[:, 'rating'] =  data_df.loc[:, 'rating'].str.replace('|', '0').astype(float)
data_df.loc[:, 'user_id'] = data_df.loc[:, 'user_id'].str.split(',')
df_expanded = data_df.explode('user_id')

pre_trained_data = load_prepared_data()

# --- Transform Data ---
user_item_matrix = transform_data(df_expanded)


# --- Streamlit UI ---

st.title("Product Recommendation System")
st.sidebar.header("About PRS")
st.sidebar.info(
    "This is a Product Recommendation System that uses clustering to recommend products based on user ratings."
)
st.logo("images/prs-seeklogo.png", size="large")

st.sidebar.subheader("Sample User IDs")
st.sidebar.write("AE22E2AXODSPNK3EBIHNGYS5LOSA, AFLEQIFCKD7EUBQTHJ7T7XF4MWMQ, AFC3FFC5PKFF5PMA52S3VCHOZ5FQ")
st.header("User Input")
user_id = st.text_input("Enter User ID", value="AGYYVPDD7YG7FYNBXNGXZJT525AQ")

products = []

if st.button("Get Recommendations"):
    # --- Get the recommended products ---
    products = recommend_for_user(user_id=user_id, user_matrix=pre_trained_data, original_df=df_expanded)

res = st.container(border=True, key=None)

if st.sidebar.button("Retrain Model"):
    # --- Train and Predict Clusters ---
    clusters = train_predict_cluster(user_item_matrix)
    res.success("Model retrained successfully!")
    user_item_matrix['clusters'] = clusters
    res.write(f"Clusters found: {len(user_item_matrix['clusters'].unique())}")


if len(products) > 1:
    res.subheader(f"Recommendations for User ID: {user_id}")
    res.write(products)
else:
    res.subheader("No recommendations available.")
    res.write("Please ensure the user ID exists in the dataset or retrain the model if necessary.")
