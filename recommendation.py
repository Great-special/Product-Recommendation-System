import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option('future.no_silent_downcasting', True)


def transform_data(df_expanded):
    df_expanded.loc[:, 'simplified_category'] = df_expanded.loc[:, 'category'].apply(simplify_category)
    df_expanded = df_expanded.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    return df_expanded


def train_predict_cluster(df_expanded):
    import joblib
    model = joblib.load('kmeans_recommendation_model.pkl')
    scaler = joblib.load('X-scaler.pkl')
    X_scaled = scaler.fit_transform(df_expanded)
    return model.fit_predict(X_scaled)


def load_prepared_data():
    import joblib
    return joblib.load('user_item_matrix_filled_compressed.pkl')



# Create broader category groups
def simplify_category(category):
    if pd.isna(category):
        return 'Other'
    
    main_cat = category.split('|')[0]
    
    # Group similar categories
    if main_cat == 'Electronics':
        if 'Mobile' in category:
            return 'Mobile & Accessories'
        elif 'Computer' in category or 'Laptop' in category:
            return 'Computing & Accessories'
        elif 'Audio' in category or 'Headphone' in category:
            return 'Audio'
        elif 'Camera' in category:
            return 'Camera & Photography'
        elif 'TV' in category or 'Theater' in category:
            return 'TV & Home Theater'
        else:
            return 'Other Electronics'
    
    elif main_cat == 'Computers&Accessories':
        return 'Computing & Accessories'
    
    elif main_cat == 'Home&Kitchen':
        return 'Home & Kitchen'
    
    elif main_cat == 'OfficeProducts':
        return 'Office & Stationery'
    
    else:
        return main_cat

def recommend_for_user(user_id, user_matrix, original_df, n_recommendations=5):
    if user_id not in user_matrix.index:
        return "User not found."

    # Getting the cluster the user belongs to
    cluster_id = user_matrix.loc[user_id, 'Cluster']
    cluster_users = user_matrix[user_matrix['Cluster'] == cluster_id].drop('Cluster', axis=1)

    # Average rating per product in cluster
    product_scores = cluster_users.mean().sort_values(ascending=False)

    # Products the user has already rated
    user_rated = original_df[original_df['user_id'] == user_id]['product_id'].unique()

    # Recommend top N unrated products
    recommendations = product_scores[~product_scores.index.isin(user_rated)].head(n_recommendations)
    return original_df[original_df['product_id'].isin(recommendations.index)]['product_name'].unique()