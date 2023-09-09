import numpy as np
import pandas as pd
import nltk
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


# # Function to convert the string representation to NumPy array
# def convert_to_array(vector_str):
#     return np.array([float(num) for num in vector_str.replace('[','').replace(']','').split()])


# # Load the dataset
# df = pd.read_excel('./preprocessed_imdb.xlsx')
# # Use the 'convert_to_array' function to convert the 'vector' column
# df['vector'] = df['vector'].apply(convert_to_array)



# # nltk.download('punkt')
# # nltk.download('stopwords')

# # Load the Word2Vec model from the local file
model = Word2Vec.load("Word2Vec_imdb.bin")

# Function to preprocess the user query
def preprocess_query(query):
    # Convert to lowercase and tokenize
    tokens = nltk.word_tokenize(str(query).lower())

    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

embedding_size = 100


# Function to calculate the average vector for a movie's text
def get_average_vector(text):
    vectors = [model.wv[token] for token in text if token in model.wv]
    if len(vectors) > 0:
        return sum(vectors) / len(vectors)
    return [0] * embedding_size  # Return a zero vector if the token is not in the vocabulary


# Function to find the indices of the top 'n' nearest vectors in df['vector'] to the input vector
def find_top_n_nearest_vector_indices(user_query_vector, df_vectors, n=5):
    similarity_scores = cosine_similarity([user_query_vector], df_vectors)
    similarity_scores = similarity_scores[0]  # Get the similarity scores as a 1D array
    top_n_indices = np.argsort(similarity_scores)[-n:][::-1]  # Indices of top 'n' most similar vectors
    top_n_scores = similarity_scores[top_n_indices]  # Get the similarity scores for top 'n' movies
    return top_n_indices, top_n_scores
