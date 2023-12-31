from flask import Flask, render_template, request, url_for, redirect, session, jsonify
import mysql.connector
import os
import traceback
import pandas as pd
import numpy as np
import nltk
from math import sqrt
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.neighbors import NearestNeighbors
from  recommendation_part.recommend import preprocess_query, get_average_vector, find_top_n_nearest_vector_indices

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math  

nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__)
app.secret_key = os.urandom(24)

def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="flask"
    )

@app.route('/fetch_datasets', methods=['GET'])
def fetch_datasets():
    try:
        # Fetch the datasets from the SQL database using a SELECT query
        query = "SELECT id, rating, ruser FROM ratings"

        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute(query)
        datasets = cursor.fetchall()

        # Convert the datasets into a list of dictionaries
        fetched_data = []
        for dataset in datasets:
            id,  rating, ruser = dataset
            data_dict = {
                'id': id,
                'ruser': ruser,
                'rating': rating,
            }
            fetched_data.append(data_dict)
            print(data_dict)
            fetched_data_df = pd.DataFrame(fetched_data)
            print(fetched_data_df)
            fetched_data_df.shape
            fetched_data_df[fetched_data_df['ruser'] == 5]
            input_movie_title = "title"
            input_movie = pd.DataFrame(fetched_data)  # Convert fetched_data to a DataFrame
            input_movie = input_movie[input_movie['movie_title'] == input_movie_title]
            
            userSubsetGroup = users.groupby(['ruser'])
            userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]) >= 5, reverse=True)
            userSubsetGroup

        # Apply filtering and merging
        

        # Check if the code is running within a request context
        with app.test_request_context():
            if 'user_id' in session:
                user_id = session['user_id']
                users = pd.DataFrame(fetched_data)
                users = users[users['ruser'] == user_id]
                userSubsetGroup = users.groupby(['ruser'])
                userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]) >= 5, reverse=True)

                pearsonCorDict = {}
                for name, group in userSubsetGroup:
                    group = group.sort_values(by='id')
                    input_movie = input_movie.sort_values(by='id')
                    n = len(group)
                    temp = input_movie[input_movie['id'].isin(group['id'].tolist())]
                    tempRatingList = temp['rating'].tolist()
                    tempGroupList = group['rating'].tolist()
                    Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(n)
                    Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(n)
                    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(
                        tempGroupList) / float(n)

                    if Sxx != 0 and Syy != 0:
                        pearsonCorDict[name] = Sxy / sqrt(Sxx * Syy)
                    else:
                        pearsonCorDict[name] = 0

                    print(f"Pearson correlation for user {name}: {pearsonCorDict[name]}")

                pearsonDF = pd.DataFrame.from_dict(pearsonCorDict, orient='index')
                pearsonDF.columns = ['similarityIndex']
                pearsonDF['ruser'] = pearsonDF.index

                topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)
                topUsers = topUsers.reset_index()

                print(topUsers.head(5))
                # Fetch the rating data from the SQL database using a SELECT query
                rating_query = "SELECT ruser, rating FROM ratings"
                cursor.execute(rating_query)
                rating_data = cursor.fetchall()
                rating_columns = ['ruser', 'rating']
                rating = pd.DataFrame(rating_data, columns=rating_columns)

                print("topUsers DataFrame:")
                print(topUsers.head(5))
                print("Column names of topUsers DataFrame:")
                print(topUsers.columns)

                rating = pd.DataFrame(rating_data, columns=rating_columns)
                print("rating DataFrame:")
                print(rating.head(5))
                print("Column names of rating DataFrame:")
                print(rating.columns)

                # Convert 'ruser' column in topUsers DataFrame to integer
               # Convert 'ruser' column in topUsers DataFrame to integer
                # Convert 'ruser' column in topUsers DataFrame to integer
                topUsers['ruser'] = topUsers['ruser'].astype(int)

                rating['ruser'] = rating['ruser'].astype(int)

                # Merge topUsers with rating data
                topUsersRating = topUsers.merge(rating, on='ruser', how='inner')

                print(topUsers['ruser'].unique())
                print(rating['ruser'].unique())
                print("Column names of topUsers DataFrame:")
                print(topUsers.columns)

                print("Column names of rating DataFrame:")
                print(rating.columns)

                # Check if the 'id' column exists in topUsersRating DataFrame
                if 'ruser' in topUsersRating.columns:
                    # Group by 'id' and calculate the sum of 'similarityIndex' and 'rating'
                    tempTopUsersRating = topUsersRating.groupby('ruser').sum()[['similarityIndex', 'rating']]
                    tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
                    print(tempTopUsersRating.head())
                else:
                    print("The 'id' column is not present in the topUsersRating DataFrame.")

        # Convert the pearsonDF DataFrame to a string
        pearsonDF_str = pearsonDF.to_string(index=False)
        print("PearsonDF as string:", pearsonDF_str)

        # Return the fetched datasets and pearsonDF as string response
        return jsonify({'fetched_data': fetched_data, 'pearsonDF': pearsonDF_str})

    except Exception as e:
        # Handle any exceptions that occur during fetching
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500


# Define the number of movies per page
MOVIES_PER_PAGE = 20

@app.route('/')
def hello_world():
    # Get the page number from the request's query parameters
    page = request.args.get('page', default=1, type=int)
    
    cnx = create_connection()
    cursor = cnx.cursor()

    try:
        # Calculate the offset based on the page number
        offset = (page - 1) * MOVIES_PER_PAGE

        # Fetch movies for the current page using LIMIT and OFFSET
        cursor.execute("SELECT Genre, Title, Image_URL, Rating FROM movies LIMIT %s OFFSET %s", (MOVIES_PER_PAGE, offset))
        movies = cursor.fetchall()

        # Calculate the total number of movies (needed for pagination)
        cursor.execute("SELECT COUNT(*) FROM movies")
        total_movies = cursor.fetchone()[0]

        # Calculate the total number of pages
        total_pages = math.ceil(total_movies / MOVIES_PER_PAGE)

        # You can adjust your other queries here (e.g., categories)

        if 'user_id' in session:
            user_id = session['user_id']
            email_query = "SELECT email FROM users WHERE id = %s"
            cursor.execute(email_query, (user_id,))
            user_email = cursor.fetchone()[0]
            cursor.close()
            cnx.close()
            return render_template('index.html', user_email=user_email, movies=movies, total_pages=total_pages, current_page=page)
        else:
            cursor.close()
            cnx.close()
            return render_template('index.html', movies=movies, total_pages=total_pages, current_page=page)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while retrieving movies"

# @app.route('/')
# def hello_world():
#     cnx = create_connection()
#     cursor = cnx.cursor()

#     try:
#         # cursor.execute("SELECT Genre, Title, Image_URL, Rating FROM movies LIMIT 20")
#         cursor.execute("SELECT Genre, Title, Image_URL, Rating FROM movies LIMIT 100")
#         movies = cursor.fetchall()
#         print("Movies : ", movies)

#         category_query = "SELECT DISTINCT Genre FROM movies"
#         cursor.execute(category_query)
#         categories = cursor.fetchall()
#         print("Categories : ", categories)

#         if 'user_id' in session:
#             user_id = session['user_id']
#             email_query = "SELECT email FROM users WHERE id = %s"
#             cursor.execute(email_query, (user_id,))
#             user_email = cursor.fetchone()[0]
#             cursor.close()
#             cnx.close()
#             return render_template('index.html', user_email=user_email, movies=movies, categories=categories)
#         else:
#             cursor.close()
#             cnx.close()
#             return render_template('index.html', movies=movies, categories=categories)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         return "Error occurred while retrieving movies"



@app.route('/rating/<title>')
def movie_rating(title):
    # Fetch movie details from the database
    cnx = create_connection()
    cursor = cnx.cursor()
    print("Title : ", title)
    query = "SELECT * FROM movies WHERE Title = %s"
    cursor.execute(query, (title,))
    movie = cursor.fetchone()
    cursor.close()
    cnx.close()
    print("Movie : ", movie)
    movie_id = movie[4]
    print("Movie ID : ", movie_id)
    
    try:
        user_email = session.get('user_email')
    
        return render_template('ratings.html', movie_title=title, id=movie_id, user_email=user_email )
    except:
        return render_template('ratings.html', movie_title=title, id=movie_id )



@app.route('/genre/<genre>')
def show_genre(genre):
    image_url = url_for('static', filename='logo.jpg')
    try:
        user_email = session.get('user_email')
        return render_template('genre.html', selected_genre=genre, image_url=image_url, user_email=user_email)
    except:
        return render_template('genre.html', selected_genre=genre, image_url=image_url)



@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    # Get the submitted form data
    movie_id = request.form.get('id')
    movie_title = request.form.get('title')
    rating = int(request.form.get('rating'))
    user_id = session.get('user_id')  # Get the user ID from the session

    cnx = create_connection()
    cursor = cnx.cursor()

    try:
        if movie_title is not None and movie_title.strip() != "":
            # Save the rating to the database
            insert_query = "INSERT INTO newrating (user_id, movie_id, ratings) VALUES (%s, %s, %s)"
            # insert_query = "INSERT INTO ratings (movie_id, movie_title, rating, ruser) VALUES (%s, %s, %s, %s)"

            if user_id is not None:  # Check if the user ID exists
                cursor.execute(insert_query, (user_id, movie_id, rating))
                # cursor.execute(insert_query, (movie_id, movie_title, rating, user_id))
                cnx.commit()
                return redirect('/')
                # return "Rating submitted successfully"

            else:
                return "User ID not found"
        else:
            return "Movie title is missing or empty"

    except Exception as e:
        cnx.rollback()
        traceback.print_exc()
        error_message = f"Error occurred while submitting the rating: {str(e)}"
        return error_message

    finally:
        cursor.close()
        cnx.close()

def trending_movies():
    cnx = create_connection()
    cursor = cnx.cursor()

    try:
        cursor.execute("SELECT * FROM movies ORDER BY Rating DESC LIMIT 10")
        movies = cursor.fetchall()
        return movies
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while retrieving movies"
    finally:
        cursor.close()
        cnx.close()

@app.route('/popular_movies')
def popular_movies():
    image_url = url_for('static', filename='logo.jpg')
    trending_movie_list = trending_movies()  # Call Function for trending movies
    print("Trending Movies : ", trending_movie_list)
    try:
        user_email = session.get('user_email')
        return render_template('popular_movies.html', image_url=image_url, user_email=user_email, popular_movies = trending_movie_list)            
    except:
        return render_template('popular_movies.html', image_url=image_url, popular_movies=trending_movie_list)


@app.route('/search', methods=['POST'])
def search():
    user_input = request.form.get('query')

    print("User Input : ", user_input)

    try:
        cnx = create_connection()
        cursor = cnx.cursor()

        # SQL query to search for movies based on user input
        query = "SELECT * FROM movies WHERE Title LIKE %s"
        cursor.execute(query, ("%" + user_input + "%",))
        movies = cursor.fetchall()
        print("Movies : ", movies)

        cursor.close()
        cnx.close()

        try:
            user_email = session.get('user_email')
            return render_template('search.html', movies=movies, user_input=user_input, user_email=user_email)
        except:
            return render_template('search_result.html', movies=movies, user_input=user_input)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while retrieving movies"




@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    cnx = create_connection()
    cursor = cnx.cursor()

    try:
        # Execute a SELECT query to fetch the user based on email and password
        query = "SELECT * FROM `users` WHERE `email` = %s AND `password` = %s"
        values = (email, password)
        cursor.execute(query, values)
        user = cursor.fetchone()  # Fetch the first row
        print("User : ", user)

        if user is not None:
            # User found, store user details in session
            session['user_id'] = user[0]
            session['user_email'] = user[1]
            return redirect('/')
        else:
            return render_template('login.html', error="Invalid email or password")
    except Exception as e:
        # Handle any exceptions that might occur
        traceback.print_exc()  # Print the traceback to see the specific error
        return "Error occurred while validating login"
    finally:
        cursor.close()
        cnx.close()


# @app.route('/login_validation', methods=['POST'])
# def login_validation():
#     email = request.form.get('email')
#     password = request.form.get('password')

#     cnx = create_connection()
#     cursor = cnx.cursor()

#     try:
#         # Execute a SELECT query to fetch the user based on email and password
#         query = "SELECT * FROM `users` WHERE `email` = %s AND `password` = %s"
#         values = (email, password)
#         cursor.execute(query, values)
#         user = cursor.fetchone()
#         user = cursor.fetchone()

#         if user is not None:
#             # User found, store user details in session
#             session['user_id'] = user[0]
#             session['user_email'] = user[1]
#             return redirect('/')
#         else:
#             return render_template('login.html', error="Invalid email or password")
#     except Exception as e:
#         # Handle any exceptions that might occur
#         traceback.print_exc()  # Print the traceback to see the specific error
#         return "Error occurred while validating login"
#     finally:
#         cursor.close()
#         cnx.close()


@app.route('/add_user', methods=['POST'])
def add_user():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    cnx = create_connection()
    cursor = cnx.cursor()

    try:
        query = "INSERT INTO `users` (`name`, `email`, `password`) VALUES (%s, %s, %s)"
        values = (name, email, password)
        cursor.execute(query, values)
        cnx.commit()

        return redirect('/')
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while adding user"
    finally:
        cursor.close()
        cnx.close()


@app.route('/logout')
def logout():
    if 'user_id' in session:
        session.pop('user_id')
    return redirect('/login')


def load_ratings_data():
    cnx = create_connection()
    cursor = cnx.cursor()

    try:
        cursor.execute("SELECT * FROM newrating")
        ratings = cursor.fetchall()
        return ratings
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while retrieving ratings"
    finally:
        cursor.close()
        cnx.close()

def knn_recommendation(user_id_input):
    # Load the ratings data
    ratings = load_ratings_data()
    print("Ratings : ", ratings)
    # Convert the ratings data into a DataFrame
    ratings_df = pd.DataFrame(ratings, columns=['rating_id', 'user_id', 'movie_id', 'ratings', 'timestamp'])
    print("Ratings Dataframe : ", ratings_df)
    # Check for and handle duplicate entries by averaging ratings
    ratings_df = ratings_df.groupby(['user_id', 'movie_id'])['ratings'].mean().reset_index()
    print("Ratings Dataframe  without Duplicate: ", ratings_df)

    # Fill missing values with zeros
    user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='ratings')
    print("User Item Matrix : ", user_item_matrix)

    user_item_matrix = user_item_matrix.fillna(0)  # Fill missing values with zeros

    print("USer Id from KNN : ", user_id_input)
    print("User Item Matrix : ", user_item_matrix)

    # Computet number of neighbors as a percentage of the number of users
    percentage_of_users = 0.5  
    k = max(1, int(percentage_of_users * len(user_item_matrix)))  

    # Create the NearestNeighbors model and fit it to the user-item matrix 
    metric = 'cosine'  
    knn = NearestNeighbors(n_neighbors=k, metric=metric)

    # Fit the model to the user-item matrix
    knn.fit(user_item_matrix.values)

    # User Id for which we need to generate the recommendations
    user = int(user_id_input)

    # Find the k-nearest neighbors of the user
    distances, neighbor_indices = knn.kneighbors([user_item_matrix.loc[user].values], n_neighbors=k+1)  

    # Extract movie recommendations from the neighbors
    recommended_movie_indices = [i for i in neighbor_indices[0] if i != user]  # Exclude the user
    recommended_movies = user_item_matrix.columns[recommended_movie_indices]

    recommended_movies_data = []

    cnx = create_connection()
    cursor = cnx.cursor()

    # Print the recommended movies
    print("Recommended Movies:")
    for movie_id in recommended_movies:
        query = "SELECT * FROM movies WHERE id = %s"
        cursor.execute(query, (movie_id,))
        movie = cursor.fetchone()
        recommended_movies_data.append(movie)

    cursor.close()
    cnx.close()

    print("Recommendation Movies Data : ", recommended_movies_data)
    return recommended_movies_data

    

@app.route('/movie_details/<title>')
def movie_details(title):
    print("Title : ", title)
    try:
        cnx = create_connection()
        cursor = cnx.cursor()
        
        # Fetch movie details by title
        query = "SELECT * FROM movies WHERE Title = %s"
        cursor.execute(query, (title,))
        movie = cursor.fetchone()
        print("Movie in Details : ", movie)

        # Close the database connection
        cursor.close()
        cnx.close()

        print("Movie : ", movie)

        # Check if a user is logged in
        user_id = session.get('user_id')
        print("User ID : ", user_id)
        user_email = session.get('user_email')
        print("User Email : ", user_email)

        if user_id is not None:
            print("User ID : ", user_id)
            # Get recommended movies for the logged-in user
            recommended_movies_data = knn_recommendation(user_id)
            print("Recommended Movies Data : ", recommended_movies_data)
            
            # Get user's email (if available)
            user_email = session.get('user_email')
            
            # Render the movie details page with recommendations
            return render_template('movie_details.html', movie=movie, recommended_movies=recommended_movies_data, user_email=user_email)
        
        else:
            print("False Condition")
            # If no user is logged in, render the movie details page without recommendations
            return render_template('movie_details.html', movie=movie)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Return an error response if an exception occurs
        return "Error occurred while retrieving movie details"



# Function to convert the string representation to NumPy array
def convert_to_array(vector_str):
    return np.array([float(num) for num in vector_str.replace('[','').replace(']','').split()])


# Load the dataset
df = pd.read_excel('./preprocessed_imdb.xlsx')
# Use the 'convert_to_array' function to convert the 'vector' column
df['vector'] = df['vector'].apply(convert_to_array)


nltk.download('punkt')
nltk.download('stopwords')



# Define the API endpoint for movie recommendations
@app.route('/recommend', methods=['GET'])
def recommend_movies():

    user_input = request.args.get('user_input')
    page = request.args.get('page', default=1, type=int)
    in_page = request.args.get('in_page', default=10, type=int)

    image_url = url_for('static', filename='logo.jpg')

    if user_input is None:
        print("From Initial")
        try:
            user_email = session.get('user_email')
            return render_template('recommend.html', image_url=image_url, user_email=user_email)
        except:
            return render_template('recommend.html', image_url=image_url)
    else:
        print("Hello")
        user_query = preprocess_query(user_input)  # Call function to preprocess user input
        print("User query : ", user_query)
        # function to get the average vector for the user query
        user_query_vector = get_average_vector(user_query)

        print("User query vector : ", user_query_vector)
        
        # Convert df['vector'] to a 2D array (required for cosine_similarity)
        df_vectors = np.array(df['vector'].tolist())

        # Find the indices and similarity scores of the top 'n' nearest vectors in df['vector'] to the user_query_vector
        top_n_indices, top_n_scores = find_top_n_nearest_vector_indices(user_query_vector, df_vectors, n=page * in_page)

        # Get the details of the top 'n' nearest movies and their genres
        top_n_movies = df.loc[top_n_indices]

        # Add the similarity scores to the DataFrame
        top_n_movies['Similarity Score'] = top_n_scores

        top_n_movies = top_n_movies[['Title', 'Genre', 'Similarity Score']]     

        top_n_movies_list = top_n_movies.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
    
        print("Top n movies : ", top_n_movies_list)

        try:
            user_email = session.get('user_email')
            return render_template('recommend.html', image_url=image_url, user_input=user_input, top_n_movies=top_n_movies_list, user_email=user_email)
        except:    
            return render_template('recommend.html', image_url=image_url, user_input=user_input, top_n_movies=top_n_movies_list)




if __name__ == "__main__":
    app.debug = True
    app.run()
