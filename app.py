import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import streamlit.components.v1 as components
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="SVD Recommendation System",
    layout="wide",
    page_icon="📖",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar width and other elements
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 230px;
        max-width: 400px;
        resize: horizontal;
        overflow-x: auto;
    }
    /* Hide the resize handle on small screens */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            resize: none;
            max-width: 100%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS for dark mode without Inter font
st.markdown("""
    <style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    .main { 
        background-color: #1e1e2f; 
        color: #f1f5f9; 
    }
    .stButton>button { 
        background-color: #38bdf8; 
        color: #0f172a; 
        border-radius: 8px; 
        padding: 10px 20px;
        font-weight: 500;
        border: none;
    }
    .stButton>button:hover { 
        background-color: #7dd3fc; 
        color: #0f172a; 
    }
    .stSelectbox, .stNumberInput { 
        background-color: #2a2a3d; 
        color: #f1f5f9; 
        border: 1px solid #38bdf8; 
        border-radius: 8px; 
        padding: 8px; 
    }
    .stDataFrame { 
        background-color: #2a2a3d; 
        color: #f1f5f9; 
        border: 1px solid #38bdf8; 
        border-radius: 8px; 
    }
    .sidebar .sidebar-content { 
        background-color: #252537; 
        color: #f1f5f9; 
    }
    .article-card { 
        background-color: #2a2a3d; 
        padding: 20px; 
        margin: 10px 0; 
        border-radius: 10px; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.3); 
        color: #f1f5f9;
    }
    .article-card:hover { 
        transform: translateY(-5px); 
        transition: transform 0.2s;
    }
    .error-message { 
        color: #f38ba8; 
        font-weight: 600; 
    }
    h1, h2, h3, h4 { 
        color: #38bdf8; 
        font-weight: 600;
    }
    .stMarkdown p { 
        color: #cbd5e1; 
    }
    .hero-section {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #f1f5f9;
        padding: 60px 40px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    .hero-section h1 {
        font-size: 2.8rem;
        margin-bottom: 10px;
        color: #38bdf8;
    }
    .hero-section p {
        font-size: 1.2rem;
        max-width: 640px;
        margin: 0 auto;
        color: #cbd5e1;
    }
    .hero-section .cta {
        margin-top: 20px;
        font-size: 1.1rem;
        color: #7dd3fc;
        font-weight: 500;
    }
    @media (max-width: 768px) {
        .hero-section {
            padding: 40px 20px;
        }
        .hero-section h1 {
            font-size: 2rem;
        }
        .hero-section p {
            font-size: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Data loading and preprocessing
@st.cache_data
def load_data():
    """Load and preprocess interaction and article data."""
    try:
        df = pd.read_csv('data/user-item-interactions.csv')
        df_content = pd.read_csv('data/articles_community.csv')
    except FileNotFoundError:
        st.error("Dataset files not found. Ensure 'data/user-item-interactions.csv' and 'resources/data/articles_community.csv' exist.")
        return None, None

    df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    df_content.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    df['article_id'] = df['article_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_content['article_id'] = df_content['article_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_content.drop_duplicates(subset='article_id', keep='first', inplace=True)

    def email_mapper():
        coded_dict = {}
        cter = 1
        email_encoded = []
        for val in df['email']:
            val = 'null_user' if pd.isna(val) else val
            if val not in coded_dict:
                coded_dict[val] = cter
                cter += 1
            email_encoded.append(coded_dict[val])
        return email_encoded

    df['user_id'] = email_mapper()
    df.drop(columns=['email'], inplace=True)

    return df, df_content

# Model preparation
@st.cache_data
def create_user_item_matrix(df):
    """Create a user-item interaction matrix."""
    user_item = df.groupby(['user_id', 'article_id'])['title'].count().unstack().fillna(0)
    user_item = (user_item > 0).astype(int)
    user_item.columns = user_item.columns.astype(str)
    return user_item

@st.cache_data
def train_svd(user_item_matrix, n_components=20):
    """Train SVD model for recommendations."""
    u, s, vt = np.linalg.svd(user_item_matrix, full_matrices=False)
    s = np.diag(s[:n_components])
    u = u[:, :n_components]
    vt = vt[:n_components, :]
    return u, s, vt

@st.cache_data
def prepare_content_recommendations(df_content):
    """Prepare content-based recommendation model."""
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    df_content['doc_description'] = df_content['doc_description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df_content['doc_description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, df_content['article_id'].tolist()

# Recommendation functions
def get_top_articles(n, df):
    """Get top N most interacted articles."""
    article_counts = df.groupby('article_id')['title'].count()
    top_articles = article_counts.sort_values(ascending=False).head(n)
    top_df = df[df['article_id'].isin(top_articles.index)][['article_id', 'title']].drop_duplicates()
    return top_df[['article_id', 'title']].to_dict('records')

def get_user_recommendations(user_id, user_item_matrix, df, n_recommend=5):
    """Get collaborative filtering recommendations for a user."""
    try:
        user_vec = user_item_matrix.loc[user_id].values
        similarity = cosine_similarity([user_vec], user_item_matrix.values)[0]
        similarity_df = pd.Series(similarity, index=user_item_matrix.index).drop(user_id)
        similar_users = similarity_df.sort_values(ascending=False).head(10).index
        articles_seen = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
        similar_articles = user_item_matrix.loc[similar_users].sum()
        recommendations = [aid for aid in similar_articles[similar_articles > 0].index if aid not in articles_seen]

        if not recommendations:
            return get_top_articles(n_recommend, df)

        article_counts = df[df['article_id'].isin(recommendations)].groupby('article_id')['title'].count()
        top_articles = article_counts.sort_values(ascending=False).head(n_recommend)
        top_df = df[df['article_id'].isin(top_articles.index)][['article_id', 'title']].drop_duplicates()
        return top_df[['article_id', 'title']].to_dict('records')
    except KeyError:
        return get_top_articles(n_recommend, df)

def get_svd_recommendations(user_id, user_item_matrix, u, s, vt, df, n_recommend=5):
    """Get SVD-based recommendations for a user."""
    try:
        user_idx = user_item_matrix.index.get_loc(user_id)
        user_pred = np.dot(np.dot(u[user_idx, :], s), vt)
        pred_df = pd.Series(user_pred, index=user_item_matrix.columns)
        articles_seen = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
        pred_df = pred_df.drop(articles_seen, errors='ignore')
        top_articles = pred_df.sort_values(ascending=False).head(n_recommend).index
        top_df = df[df['article_id'].isin(top_articles)][['article_id', 'title']].drop_duplicates()
        return top_df[['article_id', 'title']].to_dict('records')
    except KeyError:
        return get_top_articles(n_recommend, df)

def get_model_recommendations(user_id, model, df, df_content, user_item_matrix, n_recommend=5):
    """Get recommendations from a pre-trained model."""
    try:
        articles_seen = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist() if user_id in user_item_matrix.index else []

        if isinstance(model, dict):
            if user_id not in model:
                return get_top_articles(n_recommend, df)
            rec_ids = [str(aid) for aid in model[user_id] if str(aid) not in articles_seen][:n_recommend]
        elif isinstance(model, list):
            if all(isinstance(item, dict) and 'article_id' in item for item in model):
                rec_ids = [str(item['article_id']) for item in model if str(item['article_id']) not in articles_seen][:n_recommend]
            else:
                rec_ids = [str(aid) for aid in model if str(aid) not in articles_seen][:n_recommend]
        else:
            return get_top_articles(n_recommend, df)

        if not rec_ids:
            return get_top_articles(n_recommend, df)

        top_df = df_content[df_content['article_id'].isin(rec_ids)][['article_id', 'doc_full_name']].rename(columns={'doc_full_name': 'title'})
        if top_df.empty:
            return get_top_articles(n_recommend, df)

        rec_ids_available = [aid for aid in rec_ids if aid in top_df['article_id'].values]
        if rec_ids_available:
            top_df = top_df.set_index('article_id').reindex(rec_ids_available).reset_index().dropna()[['article_id', 'title']]
        else:
            top_df = top_df[['article_id', 'title']]

        return top_df[['article_id', 'title']].to_dict('records')
    except Exception:
        return get_top_articles(n_recommend, df)

def get_article_recommendations(article_id, cosine_sim, article_ids, df_content, n_recommend=3):
    """Get content-based recommendations for an article."""
    try:
        idx = article_ids.index(str(article_id))
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommend+1]
        article_indices = [i[0] for i in sim_scores]
        recommended_ids = [article_ids[i] for i in article_indices]
        top_df = df_content[df_content['article_id'].isin(recommended_ids)][['article_id', 'doc_full_name']].rename(columns={'doc_full_name': 'title'})
        return top_df[['article_id', 'title']].to_dict('records')
    except ValueError:
        return []

# Main application
def main():
    """Main application function."""
    # Load data
    df, df_content = load_data()
    if df is None or df_content is None:
        st.stop()

    user_item_matrix = create_user_item_matrix(df)
    u, s, vt = train_svd(user_item_matrix)
    cosine_sim, article_ids = prepare_content_recommendations(df_content)

    # Load pre-trained models
    try:
        with open('top_5.p', 'rb') as f:
            top_5_model = pickle.load(f)
        with open('top_10.p', 'rb') as f:
            top_10_model = pickle.load(f)
        with open('top_20.p', 'rb') as f:
            top_20_model = pickle.load(f)
    except FileNotFoundError:
        st.warning("Model files not found. Using fallback recommendations.")
        top_5_model = top_10_model = top_20_model = None

    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Home", "Browse Articles", "Notebook Explorer"],
        key="nav_radio"
    )

    # Home Page
    if page == "Home":
        # Add custom CSS for the hero section
        st.markdown("""
        <style>
        /* Hero Section */
        .hero-container {
            background: linear-gradient(90deg, #1e3b70 0%, #2a5a8f 100%);
            padding: 1.5rem 1.5rem;
            border-radius: 8px;
            margin: 0 0 1.25rem 0;
            text-align: center;
            color: #ffffff;
        }
        .hero-container h1 {
            font-size: 2rem;
            margin: 0 0 0.6rem 0;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 1px 3px rgba(0,0,0,0.2);
            letter-spacing: -0.5px;
        }
        .hero-container p {
            font-size: 1.05rem;
            margin: 0 auto;
            line-height: 1.5;
            color: #e0e7ff;
            max-width: 800px;
            font-weight: 400;
            opacity: 0.95;
        }
        
        /* Content Sections */
        .stHeader, .stSubheader {
            margin: 0.5rem 0 0.75rem 0 !important;
        }
        
        /* Cards and Containers */
        .stAlert, .stDataFrame, .stMarkdown, .stContainer {
            margin: 0.5rem 0 1rem 0 !important;
        }
        
        /* Form Controls */
        .stSelectbox, .stSlider, .stButton, .stNumberInput {
            margin: 0.25rem 0 0.75rem 0 !important;
        }
        
        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .hero-container {
                padding: 1.25rem 1rem;
                margin: 0 0 1rem 0;
            }
            .hero-container h1 {
                font-size: 1.6rem;
                margin-bottom: 0.5rem;
            }
            .hero-container p {
                font-size: 0.95rem;
                line-height: 1.4;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        # Hero Section
        st.markdown("""
        <div class="hero-container">
            <h1>Welcome to SVD Recommendation System</h1>
            <p>Advanced article recommendation engine using Singular Value Decomposition. Discover relevant content tailored to your preferences.</p>
        </div>
        """, unsafe_allow_html=True)

        st.header("Your Personalized Recommendations")
        col1, col2 = st.columns([3, 2])
        with col1:
            method = st.selectbox(
                "Recommendation Method:",
                ["Top Articles", "User-User Collaborative Filtering", "SVD-Based", "Top-5 Model", "Top-10 Model", "Top-20 Model"],
                key="user_method"
            )
        with col2:
            if method in ["Top-5 Model", "Top-10 Model", "Top-20 Model"]:
                n_recommend = {"Top-5 Model": 5, "Top-10 Model": 10, "Top-20 Model": 20}[method]
                st.markdown(f"**Number of Recommendations:** {n_recommend}")
            else:
                n_recommend = st.slider("Number of Recommendations:", 1, 20, 5, key="user_n_recommend")

        if method == "Top Articles":
            if st.button("Show Top Articles", key="rank_button"):
                recommendations = get_top_articles(n_recommend, df)
                st.subheader("Most Popular Articles")
                cols = st.columns(min(n_recommend, 3))
                for i, rec in enumerate(recommendations):
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                            <div class="article-card">
                                <h4>ID: {rec['article_id']}</h4>
                                <p>{rec['title']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                csv = pd.DataFrame(recommendations).to_csv(index=False)
                st.download_button(
                    "Download Recommendations",
                    csv,
                    f"top_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="rank_download"
                )
        else:
            user_id = st.number_input("User ID (1-5148):", min_value=1, max_value=5148, value=1, key="user_id_input")
            if st.button("Get Recommendations", key="user_button"):
                try:
                    if method == "User-User Collaborative Filtering":
                        recommendations = get_user_recommendations(user_id, user_item_matrix, df, n_recommend)
                    elif method == "SVD-Based":
                        recommendations = get_svd_recommendations(user_id, user_item_matrix, u, s, vt, df, n_recommend)
                    elif method == "Top-5 Model":
                        if top_5_model is None:
                            raise ValueError("Top-5 model not loaded.")
                        recommendations = get_model_recommendations(user_id, top_5_model, df, df_content, user_item_matrix, n_recommend)
                    elif method == "Top-10 Model":
                        if top_10_model is None:
                            raise ValueError("Top-10 model not loaded.")
                        recommendations = get_model_recommendations(user_id, top_10_model, df, df_content, user_item_matrix, n_recommend)
                    elif method == "Top-20 Model":
                        if top_20_model is None:
                            raise ValueError("Top-20 model not loaded.")
                        recommendations = get_model_recommendations(user_id, top_20_model, df, df_content, user_item_matrix, n_recommend)

                    st.subheader(f"Recommended for You (User {user_id})")
                    cols = st.columns(3)
                    for i, rec in enumerate(recommendations):
                        with cols[i % 3]:
                            st.markdown(
                                f"""
                                <div class="article-card">
                                    <h4>ID: {rec['article_id']}</h4>
                                    <p>{rec['title']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    csv = pd.DataFrame(recommendations).to_csv(index=False)
                    st.download_button(
                        "Download Recommendations",
                        csv,
                        f"recs_user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key="user_download"
                    )
                except ValueError as e:
                    st.error(f"Error: {str(e)}", icon="🚨")

    # Browse Articles Page
    elif page == "Browse Articles":
        st.header("Browse Articles")
        st.markdown("Select an article to view its summary and discover similar articles.")

        article_options = df_content[['article_id', 'doc_full_name']].set_index('article_id')['doc_full_name'].to_dict()
        article_id = st.selectbox("Select an Article:", options=list(article_options.keys()), format_func=lambda x: article_options[x], key="article_select")

        if article_id:
            article_data = df_content[df_content['article_id'] == article_id].iloc[0]
            st.subheader(article_data['doc_full_name'])
            st.markdown(f"**Summary:** {article_data['doc_description']}")

            st.subheader("Recommended Articles")
            recommendations = get_article_recommendations(article_id, cosine_sim, article_ids, df_content, n_recommend=3)
            if recommendations:
                cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                            <div class="article-card">
                                <h4>ID: {rec['article_id']}</h4>
                                <p>{rec['title']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                csv = pd.DataFrame(recommendations).to_csv(index=False)
                st.download_button(
                    "Download Recommendations",
                    csv,
                    f"similar_to_{article_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="article_download"
                )
            else:
                st.info("No recommendations available for this article.", icon="ℹ️")

    # Notebook Explorer Page
    elif page == "Notebook Explorer":
        html_file = "Recommendations_with_IBM.html"
        if os.path.exists(html_file):
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=2000, scrolling=True)
        else:
            st.error(f"HTML file '{html_file}' not found. Ensure 'Recommendations_with_IBM.html' is in the correct directory.", icon="🚨")

    # Footer
    st.markdown("---")
    st.markdown("TechBit | Powered by Streamlit & IBM Watson Studio Data | © 2025", unsafe_allow_html=True)

if __name__ == "__main__":
    main()