import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import random
import concurrent.futures
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ CONFIG ------------------
st.set_page_config(page_title="🍿 Movie Recommender", layout="wide")
st.title("🍿 Movie Recommendation System")

# Add smooth loading CSS
st.markdown("""
<style>
.movie-card {
    transition: all 0.3s ease;
    border-radius: 10px;
    padding: 10px;
}
.movie-card:hover {
    transform: scale(1.03);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}
[data-testid="stImage"] img {
    border-radius: 8px;
}
.toast {
    padding: 12px;
    color: white;
    border-radius: 4px;
    margin: 10px 0;
    animation: fadeIn 0.5s, fadeOut 0.5s 2.5s;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes fadeOut {
    from {opacity: 1;}
    to {opacity: 0;}
}
</style>
""", unsafe_allow_html=True)

# ------------------ TMDB API INTEGRATION (OPTIMIZED) ------------------
TMDB_API_KEY = "865d428a2c4c544488296053aa37499f"

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_all_ratings(movie_titles):
    """Batch fetch ratings for all movies"""
    ratings = {}
    with st.spinner("Fetching latest ratings from TMDB..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for title in movie_titles:
                url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
                futures.append(executor.submit(requests.get, url, timeout=2))
            
            for future, title in zip(concurrent.futures.as_completed(futures), movie_titles):
                try:
                    response = future.result().json()
                    if response['results']:
                        ratings[title] = round(response['results'][0]['vote_average'], 1)
                    else:
                        ratings[title] = None
                except:
                    ratings[title] = None
    return ratings

# ------------------ DATA LOADING (OPTIMIZED) ------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[df["poster"].notna()].drop_duplicates(subset="title")
    
    # Batch fetch all ratings at once
    ratings = fetch_all_ratings(df['title'].tolist())
    df['tmdb_rating'] = df['title'].map(ratings)
    
    return df

df = load_data()
genres = sorted(set(df['genre'].dropna().unique()))

# ------------------ IMAGE HANDLING (OPTIMIZED) ------------------
def load_image(url):
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.thumbnail((300, 450))
            return img
        return None
    except:
        return None

@st.cache_data(ttl=3600, show_spinner="Loading movie posters...")
def preload_images(poster_urls):
    images = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_url = {executor.submit(load_image, url): url for url in poster_urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            images[url] = future.result()
    return images

# Preload images in background
poster_urls = df['poster'].dropna().unique()
image_cache = preload_images(poster_urls)

# ------------------ RECOMMENDATION ENGINE ------------------
@st.cache_data(ttl=3600)
def prepare_recommendation_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'].fillna(''))
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = prepare_recommendation_model(df)

def get_content_based_recommendations(title, df, cosine_sim=cosine_sim, n=5):
    try:
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        return df.iloc[movie_indices]
    except:
        return pd.DataFrame()

# ------------------ SESSION STATE ------------------
if "wishlist" not in st.session_state:
    st.session_state.wishlist = []
    
if "view_state" not in st.session_state:
    st.session_state.view_state = "recommended"  # 'recommended', 'all', 'genre', 'wishlist', 'search'
    
if "current_genre" not in st.session_state:
    st.session_state.current_genre = "All Genres"

if "show_toast" not in st.session_state:
    st.session_state.show_toast = False
    st.session_state.toast_message = ""
    st.session_state.toast_type = ""

# ------------------ TOAST NOTIFICATION ------------------
def show_toast(message, type="success"):
    st.session_state.show_toast = True
    st.session_state.toast_message = message
    st.session_state.toast_type = type

# ------------------ WISHLIST FUNCTIONS ------------------
def add_to_wishlist(title):
    if title not in st.session_state.wishlist:
        st.session_state.wishlist.append(title)
        show_toast(f"Added '{title}' to wishlist", "success")
    else:
        show_toast(f"'{title}' is already in your wishlist", "info")

def remove_from_wishlist(title):
    if title in st.session_state.wishlist:
        st.session_state.wishlist.remove(title)
        show_toast(f"Removed '{title}' from wishlist", "success")

# ------------------ DISPLAY FUNCTIONS ------------------
def display_movie_card(row, show_remove=False, index=None):
    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        
        # Poster image with placeholder
        if row.poster in image_cache and image_cache[row.poster]:
            st.image(image_cache[row.poster], use_container_width=True)
        else:
            st.image(Image.new('RGB', (300, 450), color='#333'), 
                    use_container_width=True)
        
        # Title
        st.markdown(f'<div class="movie-title">{row.title}</div>', unsafe_allow_html=True)
        
        # TMDB Rating
        if pd.notna(row.tmdb_rating):
            st.markdown(f"""
                <div class="rating-container">
                    <img src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg" 
                         class="tmdb-logo" alt="TMDB" width=80>
                    <span class="tmdb-rating">{row.tmdb_rating}/10</span>
                </div>
            """, unsafe_allow_html=True)
        
        # Description
        st.markdown(f'<div class="movie-description">{row.description[:100]}...</div>', unsafe_allow_html=True)
        
        # Buttons
        if show_remove:
            if st.button("Remove from Wishlist", 
                        key=f"remove_{row.title}_{index}",
                        use_container_width=True,
                        type="primary"):
                remove_from_wishlist(row.title)
                st.rerun()
        else:
            if st.button("Add to Wishlist", 
                        key=f"wish_{row.title}_{index}" if index else f"wish_{row.title}",
                        use_container_width=True):
                add_to_wishlist(row.title)
                st.rerun()
        
        # IMDb Link
        st.markdown(f"[🔗 IMDb Link]({row.imdb_link})")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_movie_grid(movies, n_per_row=5, show_remove=False):
    cols = st.columns(n_per_row)
    for idx, row in enumerate(movies.itertuples()):
        with cols[idx % n_per_row]:
            display_movie_card(row, show_remove=show_remove, index=idx)

def display_wishlist():
    if not st.session_state.wishlist:
        st.markdown("""
            <div style="text-align: center; padding: 40px;">
                <h3>Your wishlist is empty</h3>
                <p>Start adding movies to see them here!</p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.subheader(f"⭐ Your Wishlist ({len(st.session_state.wishlist)} movies)")
    wishlist_df = df[df['title'].isin(st.session_state.wishlist)]
    display_movie_grid(wishlist_df, n_per_row=3, show_remove=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Filters")
    search_query = st.text_input("🔍 Search for a Movie")
    selected_genre = st.selectbox("🎬 Select Genre", ["All Genres"] + genres)
    
    # Reset view state if filters change
    if (search_query and st.session_state.view_state != "search") or \
       (selected_genre != st.session_state.current_genre):
        st.session_state.view_state = "recommended"
        st.session_state.current_genre = selected_genre
    
    # Wishlist section
    st.subheader("Wishlist")
    st.markdown(f'<div class="wishlist-count">{len(st.session_state.wishlist)} movies</div>', 
               unsafe_allow_html=True)
    
    if st.button("⭐ View Wishlist" if st.session_state.view_state != "wishlist" else "🎬 Back to Movies",
               use_container_width=True):
        st.session_state.view_state = "wishlist" if st.session_state.view_state != "wishlist" else "recommended"
        st.rerun()

# ------------------ TOAST DISPLAY ------------------
if st.session_state.show_toast:
    toast_color = "#4CAF50" if st.session_state.toast_type == "success" else "#2196F3" if st.session_state.toast_type == "info" else "#f44336"
    st.markdown(
        f'<div class="toast" style="background-color: {toast_color}">{st.session_state.toast_message}</div>',
        unsafe_allow_html=True
    )
    time.sleep(3)
    st.session_state.show_toast = False
    st.rerun()

# ------------------ MAIN CONTENT ------------------
if st.session_state.view_state == "wishlist":
    display_wishlist()
else:
    # Apply filters
    if search_query:
        filtered = df[df["title"].str.contains(search_query, case=False)]
        st.session_state.view_state = "search"
    elif selected_genre != "All Genres":
        filtered = df[df["genre"] == selected_genre]
    else:
        filtered = df

    # Handle view states
    if st.session_state.view_state == "recommended":
        if selected_genre == "All Genres" and not search_query:
            st.subheader("🎯 Recommended for You")
            display_movie_grid(df.sample(n=min(5, len(df))))
            
            if st.button("🎬 Explore All Movies", use_container_width=True):
                st.session_state.view_state = "all"
                st.rerun()
        
        elif selected_genre != "All Genres":
            st.subheader(f"🎬 Top {selected_genre} Movies")
            display_movie_grid(filtered.sample(n=min(5, len(filtered))))
            
            if st.button(f"🎬 Explore All {selected_genre} Movies", use_container_width=True):
                st.session_state.view_state = "genre"
                st.rerun()
        
        elif search_query:
            st.subheader(f"🔍 Search Results for '{search_query}'")
            display_movie_grid(filtered, n_per_row=3)
    
    elif st.session_state.view_state == "all":
        st.subheader("🎞 All Movies")
        page_size = 15
        page = st.number_input("Page", min_value=1, max_value=len(df)//page_size + 1, value=1)
        display_movie_grid(df.iloc[(page-1)*page_size : page*page_size], n_per_row=3)
    
    elif st.session_state.view_state == "genre":
        st.subheader(f"🎞 All {selected_genre} Movies")
        page_size = 15
        page = st.number_input("Page", min_value=1, max_value=len(filtered)//page_size + 1, value=1)
        display_movie_grid(filtered.iloc[(page-1)*page_size : page*page_size], n_per_row=3)
    
    elif st.session_state.view_state == "search":
        st.subheader(f"🔍 Search Results for '{search_query}'")
        display_movie_grid(filtered, n_per_row=3)

    # Recommendations based on wishlist
    if st.session_state.wishlist and st.session_state.view_state != "wishlist":
        st.subheader("✨ Recommendations Based on Your Wishlist")
        try:
            recommended = get_content_based_recommendations(
                random.choice(st.session_state.wishlist), 
                df, 
                cosine_sim,
                n=5
            )
            if not recommended.empty:
                display_movie_grid(recommended)
        except:
            pass