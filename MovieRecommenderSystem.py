import pandas as pd
import requests
import json
import time 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


API_KEY = 'APIKEY'

def buscar_pelicula_id(titulo):
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {'api_key': API_KEY, 'query': titulo, 'language': 'es-MX'}
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            return data['results'][0]['id']
    except requests.exceptions.RequestException:
        return None
    return None

def get_movie_details(movie_id):
    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {'api_key': API_KEY, 'language': 'es-MX', 'append_to_response': 'keywords,credits'}
    try:
        response = requests.get(details_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def procesar_detalles(detalles):
    if not detalles: return ""
    generos = [d['name'] for d in detalles.get('genres', [])]
    palabras_clave = [d['name'] for d in detalles.get('keywords', {}).get('keywords', [])]
    actores = [d['name'] for d in detalles.get('credits', {}).get('cast', [])[:3]]
    director = ""
    for persona in detalles.get('credits', {}).get('crew', []):
        if persona.get('job') == 'Director':
            director = persona['name']
            break
    caracteristicas = generos + palabras_clave + actores + [director]
    caracteristicas_limpias = [str(item).replace(" ", "").lower() for item in caracteristicas if item]
    return " ".join(caracteristicas_limpias)


def get_top_rated_movies(num_pages=5):
    """
    Obtiene una lista de las películas mejor valoradas de TMDb.
    """
    all_movies = []
    top_rated_url = "https://api.themoviedb.org/3/movie/top_rated"
    
    for page in range(1, num_pages + 1):
        params = {'api_key': API_KEY, 'language': 'es-MX', 'page': page}
        try:
            response = requests.get(top_rated_url, params=params)
            response.raise_for_status()
            data = response.json()
            all_movies.extend(data.get('results', []))
            print(f"Página {page} obtenida exitosamente.")
        except requests.exceptions.RequestException as e:
            print(f"Error obteniendo la página {page}: {e}")
            break 
    return all_movies

def get_recommendations(title, df, cosine_sim_matrix):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    if title not in indices:
        return f"La película '{title}' no se encuentra en nuestra base de datos."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]



print("Obteniendo lista de películas mejor valoradas...")
lista_peliculas = get_top_rated_movies(num_pages=10) # 200 películas

movie_data_list = []

print("\nProcesando cada película para obtener sus detalles...")
for movie in lista_peliculas:
    movie_id = movie.get('id')
    movie_title = movie.get('title')
    
    if movie_id and movie_title:
        print(f"Procesando: {movie_title}")
        detalles = get_movie_details(movie_id)
        texto_procesado = procesar_detalles(detalles)
        
        movie_data_list.append({
            'id': movie_id,
            'title': movie_title,
            'features': texto_procesado
        })
        #Pausa para no saturar api
        time.sleep(0.1)

df = pd.DataFrame(movie_data_list)


tfidf = TfidfVectorizer(stop_words='english')
df['features'] = df['features'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("\n--- DataFrame Creado Exitosamente ---")
print(df.head())
print("\n--- Información del DataFrame ---")
df.info()

pelicula_ejemplo = "El Padrino"

recomendaciones = get_recommendations(pelicula_ejemplo, df, cosine_sim)

print(f"Recomendaciones para '{pelicula_ejemplo}':")
print(recomendaciones)