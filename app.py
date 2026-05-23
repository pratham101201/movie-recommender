

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.markdown("**Content-Based Filtering · TF-IDF + Cosine Similarity**")
st.markdown("---")

@st.cache_data
def load_and_process():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df = df[["title", "overview", "genres", "vote_average"]].copy()

    def clean(text):
        if pd.isnull(text): return ""
        text = text.lower()
        text = re.sub(r"[^a-z\\s]", "", text)
        words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
        return " ".join(words)

    df["clean_text"] = df["overview"].apply(clean)
    tfidf  = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    matrix = tfidf.fit_transform(df["clean_text"])
    sim    = cosine_similarity(matrix, matrix)
    idx    = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return df, sim, idx

df, cosine_sim, indices = load_and_process()

def recommend(name, n=5):
    if name not in indices: return None
    i      = indices[name]
    scores = sorted(enumerate(cosine_sim[i]), key=lambda x: x[1], reverse=True)[1:n+1]
    idxs   = [s[0] for s in scores]
    vals   = [round(s[1], 4) for s in scores]
    res    = df[["title", "vote_average"]].iloc[idxs].copy()
    res["Similarity"] = vals
    res.columns       = ["🎬 Movie Title", "⭐ Rating", "📊 Similarity"]
    res.index         = range(1, len(res) + 1)
    return res

col1, col2 = st.columns([3, 1])
with col1:
    movie = st.selectbox("Select a Movie", sorted(df["title"].dropna().unique().tolist()))
with col2:
    top_n = st.number_input("Top N", min_value=1, max_value=20, value=5)

if st.button("🔍 Get Recommendations", use_container_width=True):
    with st.spinner("Finding similar movies..."):
        results = recommend(movie, n=top_n)
    if results is not None:
        st.success(f"Top {top_n} movies similar to **{movie}**")
        st.dataframe(results, use_container_width=True)
    else:
        st.error("Movie not found. Try another title.")

st.markdown("---")
st.caption("Dataset: TMDB 5000 Movies · Kaggle | Built with Streamlit · Deployed on Render")
