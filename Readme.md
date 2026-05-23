# Movie Recommendation System

Content-based filtering using TF-IDF + Cosine Similarity.  
Dataset: TMDB 5000 Movies — https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Render.  
Build command : `pip install -r requirements.txt`  
Start command : `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Live URL
https://your-app-name.onrender.com
