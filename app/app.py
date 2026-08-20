# app/streamlit_app_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CineMatch AI | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# IMPROVED COLOR SCHEME - BETTER CONTRAST
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .hero-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 2rem 3rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a8a4e6 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #e0e0e0 !important;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .movie-card {
        background: rgba(30, 30, 50, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(108, 92, 231, 0.3);
        transition: all 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-3px);
        background: rgba(40, 40, 65, 0.9);
        border-color: rgba(108, 92, 231, 0.6);
    }
    
    .movie-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.75rem;
    }
    
    .genre-tag {
        display: inline-block;
        background: rgba(108, 92, 231, 0.25);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #c4b5fd !important;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }
    
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .score-high {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: #ffffff !important;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: #1a1a2e !important;
    }
    
    .score-low {
        background: linear-gradient(135deg, #ff7675, #d63031);
        color: #ffffff !important;
    }
    
    .stat-card {
        background: rgba(30, 30, 50, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(108, 92, 231, 0.3);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c4b5fd, #a8a4e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: #b0b0b0 !important;
        font-size: 0.85rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6c5ce7 0%, #8b7ee6 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 40px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(108, 92, 231, 0.4);
    }
    
    .feature-badge {
        display: inline-block;
        background: rgba(108, 92, 231, 0.2);
        padding: 0.25rem 0.7rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #c4b5fd !important;
        margin-right: 0.5rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888888 !important;
        font-size: 0.8rem;
        border-top: 1px solid rgba(108, 92, 231, 0.2);
        margin-top: 2rem;
    }
    
    /* Form elements */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #d0d0d0 !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(30, 30, 50, 0.9);
        border: 1px solid rgba(108, 92, 231, 0.3);
        border-radius: 12px;
        color: #ffffff !important;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(30, 30, 50, 0.9);
        border: 1px solid rgba(108, 92, 231, 0.3);
        border-radius: 12px;
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 50, 0.85);
        border-radius: 12px;
        color: #e0e0e0 !important;
        border: 1px solid rgba(108, 92, 231, 0.3);
    }
    
    /* Success/Warning/Info */
    .stAlert {
        background: rgba(30, 30, 50, 0.9);
        border: 1px solid rgba(108, 92, 231, 0.3);
        border-radius: 12px;
        color: #e0e0e0 !important;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(30, 30, 50, 0.85);
        border-radius: 12px;
        border: 1px solid rgba(108, 92, 231, 0.3);
        color: #e0e0e0 !important;
    }
    
    .dataframe th {
        background: rgba(108, 92, 231, 0.3);
        color: #ffffff !important;
    }
    
    .dataframe td {
        color: #d0d0d0 !important;
    }
    
    /* Markdown text */
    p, li, span, div:not(.stButton) {
        color: #e0e0e0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #6c5ce7, #a8a4e6);
    }
    
    /* Loading text */
    .loading-text {
        color: #c4b5fd !important;
        font-size: 1.2rem;
        text-align: center;
        padding: 2rem;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        background: rgba(20, 20, 40, 0.9);
        border-radius: 12px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load and preprocess movie data"""
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    
    # Rename columns
    ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
    movies = movies.rename(columns={'movieId': 'movie_id'})
    
    # Clean titles
    movies['title_clean'] = movies['title'].apply(
        lambda x: x.split('(')[0].strip() if '(' in x else x.strip()
    )
    
    # Extract year
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna('N/A')
    
    # Process genres
    movies['genres_list'] = movies['genres'].str.split('|')
    
    # Calculate movie statistics
    movie_stats = ratings.groupby('movie_id').agg({
        'rating': ['mean', 'count']
    }).round(2)
    movie_stats.columns = ['avg_rating', 'rating_count']
    movie_stats = movie_stats.reset_index()
    movies = movies.merge(movie_stats, on='movie_id', how='left')
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    movies['rating_count'] = movies['rating_count'].fillna(0)
    
    return {
        'ratings': ratings,
        'movies': movies,
        'all_titles': movies['title_clean'].tolist(),
        'all_genres': sorted(set(g for genres in movies['genres_list'].dropna() for g in genres))
    }

def calculate_genre_similarity(genres1, genres2):
    """Calculate Jaccard similarity between two genre sets"""
    if pd.isna(genres1) or pd.isna(genres2):
        return 0.0
    
    set1 = set(str(genres1).split('|'))
    set2 = set(str(genres2).split('|'))
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def get_similar_movies(title, data, top_n=10):
    """Get similar movies based on genre overlap"""
    
    # Find the movie
    if title not in data['movies']['title_clean'].values:
        matches = data['movies'][data['movies']['title_clean'].str.contains(title, case=False, na=False)]
        if len(matches) == 0:
            return pd.DataFrame()
        target_movie = matches.iloc[0]
    else:
        target_movie = data['movies'][data['movies']['title_clean'] == title].iloc[0]
    
    target_genres = target_movie['genres']
    target_idx = target_movie.name
    
    # Calculate similarities for all movies
    similarities = []
    for idx, movie in data['movies'].iterrows():
        if idx == target_idx:
            continue
        
        # Calculate genre similarity
        genre_sim = calculate_genre_similarity(target_genres, movie['genres'])
        
        # Calculate rating similarity bonus
        rating_diff = abs(target_movie['avg_rating'] - movie['avg_rating'])
        rating_bonus = max(0, 0.1 - rating_diff * 0.05)
        
        # Final similarity score
        final_score = genre_sim + rating_bonus
        
        similarities.append({
            'index': idx,
            'title': movie['title_clean'],
            'genres': movie['genres'],
            'year': movie['year'],
            'avg_rating': movie['avg_rating'],
            'rating_count': int(movie['rating_count']),
            'similarity': round(final_score, 3),
            'genre_similarity': round(genre_sim, 3)
        })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return pd.DataFrame(similarities[:top_n])

def get_collaborative_recommendations(user_id, data, top_n=10):
    """Simple collaborative recommendations based on similar users"""
    
    # Get user's ratings
    user_ratings = data['ratings'][data['ratings']['user_id'] == user_id]
    
    if len(user_ratings) == 0:
        return pd.DataFrame()
    
    # Get user's highly rated movies (4+ stars)
    highly_rated = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
    
    if len(highly_rated) == 0:
        return pd.DataFrame()
    
    # Find other users who liked similar movies
    other_ratings = data['ratings'][data['ratings']['user_id'] != user_id]
    similar_users = other_ratings[other_ratings['movie_id'].isin(highly_rated)]
    
    if len(similar_users) == 0:
        return pd.DataFrame()
    
    # Get movies liked by similar users that the current user hasn't seen
    watched = user_ratings['movie_id'].tolist()
    candidate_movies = similar_users[~similar_users['movie_id'].isin(watched)]
    
    # Score candidate movies
    movie_scores = candidate_movies.groupby('movie_id').agg({
        'rating': ['mean', 'count']
    }).round(2)
    movie_scores.columns = ['avg_score', 'user_count']
    movie_scores = movie_scores.reset_index()
    movie_scores = movie_scores.sort_values(['avg_score', 'user_count'], ascending=False)
    
    # Get top movies
    top_movies = movie_scores.head(top_n)
    
    results = []
    for _, row in top_movies.iterrows():
        movie = data['movies'][data['movies']['movie_id'] == row['movie_id']]
        if len(movie) > 0:
            movie = movie.iloc[0]
            results.append({
                'title': movie['title_clean'],
                'genres': movie['genres'],
                'year': movie['year'],
                'avg_rating': movie['avg_rating'],
                'rating_count': int(movie['rating_count']),
                'predicted_rating': row['avg_score']
            })
    
    return pd.DataFrame(results)

def get_hybrid_recommendations(user_id, title, data, alpha=0.5, top_n=10):
    """Combine similar movies with collaborative filtering"""
    
    # Get similar movies
    similar = get_similar_movies(title, data, top_n=30)
    
    if len(similar) == 0:
        return pd.DataFrame()
    
    # Add collaborative scores
    for idx, row in similar.iterrows():
        movie = data['movies'][data['movies']['title_clean'] == row['title']]
        if len(movie) > 0:
            movie_id = movie.iloc[0]['movie_id']
            
            # Get average rating from similar users
            movie_ratings = data['ratings'][data['ratings']['movie_id'] == movie_id]
            
            # Get user's rating pattern
            user_ratings = data['ratings'][data['ratings']['user_id'] == user_id]
            
            if len(user_ratings) > 0 and len(movie_ratings) > 0:
                # Simple collaborative score based on movie popularity among similar users
                collab_score = movie_ratings['rating'].mean()
            else:
                collab_score = movie.iloc[0]['avg_rating']
            
            similar.at[idx, 'collab_score'] = collab_score
        else:
            similar.at[idx, 'collab_score'] = 3.0
    
    # Normalize scores
    scaler = MinMaxScaler()
    similar['similarity_norm'] = scaler.fit_transform(similar[['similarity']])
    similar['collab_norm'] = scaler.fit_transform(similar[['collab_score']])
    
    # Hybrid score
    similar['hybrid_score'] = alpha * similar['collab_norm'] + (1 - alpha) * similar['similarity_norm']
    similar = similar.sort_values('hybrid_score', ascending=False)
    
    return similar.head(top_n)

def search_movies(query, data, top_n=20):
    """Search movies by title"""
    if not query:
        return pd.DataFrame()
    
    results = data['movies'][
        data['movies']['title_clean'].str.contains(query, case=False, na=False)
    ].copy()
    
    if len(results) == 0:
        return pd.DataFrame()
    
    return results.head(top_n)[['title_clean', 'genres', 'year', 'avg_rating', 'rating_count']].rename(
        columns={'title_clean': 'title'}
    )

def get_trending_movies(data, top_n=20):
    """Get trending movies"""
    min_ratings = 50
    trending = data['movies'][data['movies']['rating_count'] >= min_ratings].copy()
    
    if len(trending) > 0:
        trending['score'] = trending['avg_rating'] * np.log1p(trending['rating_count'])
        trending = trending.nlargest(top_n, 'score')
    else:
        trending = data['movies'].nlargest(top_n, 'rating_count')
    
    return trending[['title_clean', 'genres', 'year', 'avg_rating', 'rating_count']].rename(
        columns={'title_clean': 'title'}
    )

def get_movies_by_genre(genre, data, top_n=30):
    """Get movies by genre"""
    if genre and genre != "All":
        genre_movies = data['movies'][
            data['movies']['genres'].str.contains(genre, case=False, na=False)
        ].copy()
    else:
        genre_movies = data['movies'].copy()
    
    return genre_movies.nlargest(top_n, 'rating_count')[['title_clean', 'genres', 'year', 'avg_rating', 'rating_count']].rename(
        columns={'title_clean': 'title'}
    )

# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_movie_card(movie, index, score_col=None, score_label="Match Score"):
    """Display movie card with genre tags - LIGHT TEXT VERSION"""
    
    # Get genres as list for tags
    genres_list = str(movie['genres']).split('|') if pd.notna(movie['genres']) else []
    genre_tags = ' '.join([f'<span class="genre-tag">{g.strip()}</span>' for g in genres_list[:5]])
    
    # Determine score class
    score = movie.get(score_col, 0) if score_col else 0
    if score >= 0.6:
        score_class = "score-high"
    elif score >= 0.3:
        score_class = "score-medium"
    else:
        score_class = "score-low"
    
    # Rating display
    rating_stars = "⭐" * min(5, int(round(movie.get('avg_rating', 0))))
    rating_display = f"{movie.get('avg_rating', 0):.1f} {rating_stars}" if movie.get('avg_rating', 0) > 0 else "No ratings"
    review_count = f"({int(movie.get('rating_count', 0))} reviews)" if movie.get('rating_count', 0) > 0 else ""
    
    score_display = f"{score:.3f}" if isinstance(score, float) else str(score)
    
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">
            {index}. {movie['title']}
            <span class="feature-badge">{movie['year']}</span>
        </div>
        <div class="movie-genres">
            {genre_tags}
        </div>
        <div style="margin-top: 0.75rem;">
            <span class="score-badge {score_class}">{score_label}: {score_display}</span>
            <span class="feature-badge">⭐ {rating_display}</span>
            <span class="feature-badge">{review_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_stats(data):
    """Display statistics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(data['movies']):,}</div>
            <div class="stat-label">Movies</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{data['ratings']['user_id'].nunique():,}</div>
            <div class="stat-label">Users</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(data['ratings']):,}</div>
            <div class="stat-label">Ratings</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = data['ratings']['rating'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{avg_rating:.1f}⭐</div>
            <div class="stat-label">Avg Rating</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">CineMatch AI</div>
        <div class="hero-subtitle">Intelligent Movie Recommendations Powered by Machine Learning</div>
        <div style="margin-top: 1rem;">
            <span class="feature-badge">🎯 Genre-Based Matching</span>
            <span class="feature-badge">👥 Collaborative Filtering</span>
            <span class="feature-badge">⚡ Hybrid Recommendations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading movie database..."):
        data = load_data()
    
    # Stats
    display_stats(data)
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["🎯 Hybrid", "🎬 Similar Movies", "👥 For You", "🔍 Search", "📊 Explore"],
        icons=["star", "film", "person", "search", "graph-up"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background": "rgba(255,255,255,0.05)", "border-radius": "50px", "margin-bottom": "2rem"},
            "icon": {"color": "#c4b5fd", "font-size": "1rem"},
            "nav-link": {"font-size": "0.9rem", "text-align": "center", "margin": "0px", "color": "#d0d0d0"},
            "nav-link-selected": {"background": "linear-gradient(135deg, #6c5ce7, #8b7ee6)", "color": "#ffffff"},
        }
    )
    
    # Hybrid Recommendations
    if selected == "🎯 Hybrid":
        st.markdown("### 🎯 Hybrid Recommendations")
        st.markdown("*Combines genre similarity with your taste*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            movie_list = data['all_titles']
            favorite_movie = st.selectbox("🎬 Select a movie you love", movie_list)
        
        with col2:
            user_id = st.number_input("👤 Your User ID", min_value=1, max_value=610, value=1)
        
        col3, col4 = st.columns(2)
        with col3:
            alpha = st.slider("⚖️ Balance", 0.0, 1.0, 0.5, 0.05, 
                             help="0 = Just similar movies, 1 = Just your taste")
        with col4:
            top_n = st.select_slider("📊 Number", options=[5, 10, 15, 20], value=10)
        
        if st.button("✨ Get Recommendations", use_container_width=True):
            with st.spinner("Finding movies you'll love..."):
                results = get_hybrid_recommendations(user_id, favorite_movie, data, alpha, top_n)
            
            if len(results) > 0:
                st.success(f"🎯 Found {len(results)} recommendations based on '{favorite_movie}'")
                for idx, row in results.iterrows():
                    display_movie_card(row.to_dict(), idx+1, 'hybrid_score', "Match")
            else:
                st.warning("Movie not found. Try another title.")
    
    # Similar Movies (Content-Based)
    elif selected == "🎬 Similar Movies":
        st.markdown("### 🎬 Similar Movies")
        st.markdown("*Find movies with similar genres*")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            movie_list = data['all_titles']
            favorite_movie = st.selectbox("🎬 Select a movie", movie_list, key="similar_select")
        
        with col2:
            top_n = st.select_slider("Number", options=[5, 10, 15, 20], value=10, key="similar_n")
        
        if st.button("🔍 Find Similar Movies", use_container_width=True):
            with st.spinner("Finding similar movies..."):
                results = get_similar_movies(favorite_movie, data, top_n)
            
            if len(results) > 0:
                st.success(f"🎬 Movies similar to '{favorite_movie}':")
                for idx, row in results.iterrows():
                    display_movie_card(row.to_dict(), idx+1, 'similarity', "Similarity")
            else:
                st.warning("Movie not found.")
    
    # Collaborative (For You)
    elif selected == "👥 For You":
        st.markdown("### 👥 Personalized Recommendations")
        st.markdown("*Movies recommended based on your taste*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.number_input("👤 Your User ID", min_value=1, max_value=610, value=1, key="collab_user")
        
        with col2:
            top_n = st.select_slider("Number", options=[5, 10, 15, 20], value=10, key="collab_n")
        
        # Show user history
        user_ratings = data['ratings'][data['ratings']['user_id'] == user_id]
        if len(user_ratings) > 0:
            with st.expander(f"📊 Your rated movies ({len(user_ratings)})"):
                user_movies = user_ratings.merge(data['movies'], on='movie_id', how='left')
                for _, row in user_movies.head(10).iterrows():
                    st.write(f"• {row['title_clean']} - {row['rating']}⭐")
        else:
            st.info(f"👤 User {user_id} hasn't rated any movies yet.")
        
        if st.button("🎯 Get Personalized Picks", use_container_width=True):
            with st.spinner("Finding movies for you..."):
                results = get_collaborative_recommendations(user_id, data, top_n)
            
            if len(results) > 0:
                st.success(f"🎯 Recommended for you based on your taste:")
                for idx, row in results.iterrows():
                    display_movie_card(row.to_dict(), idx+1, 'predicted_rating', "Predicted")
            else:
                st.warning("Not enough data. Rate more movies or try a different user ID.")
    
    # Search
    elif selected == "🔍 Search":
        st.markdown("### 🔍 Search Movies")
        st.markdown("*Find movies by title*")
        
        search_query = st.text_input("Enter movie title", placeholder="e.g., Toy Story, Inception, The Dark Knight...")
        
        if search_query:
            with st.spinner("Searching..."):
                results = search_movies(search_query, data, 30)
            
            if len(results) > 0:
                st.success(f"✅ Found {len(results)} movies matching '{search_query}'")
                for idx, row in results.iterrows():
                    row_dict = row.to_dict()
                    genres_list = str(row_dict['genres']).split('|') if pd.notna(row_dict['genres']) else []
                    genre_tags = ' '.join([f'<span class="genre-tag">{g.strip()}</span>' for g in genres_list[:5]])
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            {idx+1}. {row_dict['title']}
                            <span class="feature-badge">{row_dict['year']}</span>
                        </div>
                        <div class="movie-genres">
                            {genre_tags}
                        </div>
                        <div style="margin-top: 0.5rem;">
                            <span class="feature-badge">⭐ {row_dict['avg_rating']:.1f}</span>
                            <span class="feature-badge">({int(row_dict['rating_count'])} reviews)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"No movies found matching '{search_query}'")
    
    # Explore
    elif selected == "📊 Explore":
        st.markdown("### 📊 Explore Movies")
        
        tab1, tab2 = st.tabs(["🔥 Trending Now", "🎭 Browse by Genre"])
        
        with tab1:
            with st.spinner("Loading trending movies..."):
                trending = get_trending_movies(data, 20)
            
            if len(trending) > 0:
                for idx, row in trending.iterrows():
                    row_dict = row.to_dict()
                    genres_list = str(row_dict['genres']).split('|') if pd.notna(row_dict['genres']) else []
                    genre_tags = ' '.join([f'<span class="genre-tag">{g.strip()}</span>' for g in genres_list[:5]])
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">
                            {idx+1}. {row_dict['title']}
                            <span class="feature-badge">{row_dict['year']}</span>
                        </div>
                        <div class="movie-genres">
                            {genre_tags}
                        </div>
                        <div style="margin-top: 0.5rem;">
                            <span class="feature-badge">🔥 {row_dict['avg_rating']:.1f}⭐</span>
                            <span class="feature-badge">({int(row_dict['rating_count'])} reviews)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            genres = ["All"] + data['all_genres']
            selected_genre = st.selectbox("Select Genre", genres)
            top_n = st.select_slider("Movies to show", options=[10, 20, 30, 50], value=20)
            
            if st.button("Browse Movies", use_container_width=True):
                with st.spinner("Loading movies..."):
                    results = get_movies_by_genre(selected_genre, data, top_n)
                
                if len(results) > 0:
                    for idx, row in results.iterrows():
                        row_dict = row.to_dict()
                        genres_list = str(row_dict['genres']).split('|') if pd.notna(row_dict['genres']) else []
                        genre_tags = ' '.join([f'<span class="genre-tag">{g.strip()}</span>' for g in genres_list[:5]])
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">
                                {idx+1}. {row_dict['title']}
                                <span class="feature-badge">{row_dict['year']}</span>
                            </div>
                            <div class="movie-genres">
                                {genre_tags}
                            </div>
                            <div style="margin-top: 0.5rem;">
                                <span class="feature-badge">⭐ {row_dict['avg_rating']:.1f}</span>
                                <span class="feature-badge">({int(row_dict['rating_count'])} reviews)</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>CineMatch AI • Intelligent Movie Recommendation System</p>
        <p>Powered by Machine Learning • Content-Based & Collaborative Filtering</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()