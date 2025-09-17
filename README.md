
# Sistema de Recomendación de Películas con TMDb API

Este es un sistema de recomendación de contenido que sugiere películas similares a una película dada por el usuario. El sistema obtiene los datos en tiempo real desde la API de The Movie Database (TMDb) y utiliza un modelo de Machine Learning para encontrar similitudes.

## Características

- **Obtención de Datos Dinámica:** Utiliza la API de TMDb para obtener un listado de las películas mejor valoradas.
- **Ingeniería de Características:** Extrae y procesa características clave de cada película como género, palabras clave, director y actores principales.
- **Modelo de Similitud:** Implementa un vectorizador TF-IDF y la similitud del coseno para calcular qué tan parecidas son las películas entre sí.

## Tecnologías Utilizadas

- **Python**
- **Pandas:** Para la manipulación de datos.
- **Scikit-learn:** Para la implementación del modelo TF-IDF y la similitud del coseno.
- **Requests:** Para realizar las llamadas a la API de TMDb.



