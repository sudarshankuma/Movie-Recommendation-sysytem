from flask import Flask, render_template, request, url_for, redirect, session, jsonify
import mysql.connector
import os
import traceback
import pandas as pd
import numpy as np
from math import sqrt
from werkzeug.security import generate_password_hash, check_password_hash

from  recommendation_part.recommend import preprocess_query, convert_to_array, get_average_vector, find_top_n_nearest_vector_indices


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






@app.route('/')
def hello_world():
    cnx = create_connection()
    cursor = cnx.cursor()

    try:
        cursor.execute("SELECT Genre, Title, Image_URL, Rating FROM movies LIMIT 20")
        movies = cursor.fetchall()

        category_query = "SELECT DISTINCT Genre FROM movies"
        cursor.execute(category_query)
        categories = cursor.fetchall()
        print("Categories : ", categories)

        if 'user_id' in session:
            user_id = session['user_id']
            email_query = "SELECT email FROM users WHERE id = %s"
            cursor.execute(email_query, (user_id,))
            user_email = cursor.fetchone()[0]
            cursor.close()
            cnx.close()
            return render_template('index.html', user_email=user_email, movies=movies, categories=categories)
        else:
            cursor.close()
            cnx.close()
            return render_template('index.html', movies=movies, categories=categories)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error occurred while retrieving movies"


@app.route('/rating/<title>')
def movie_rating(title):
    return render_template('ratings.html', movie_title=title)



@app.route('/genre/<genre>')
def show_genre(genre):
    image_url = url_for('static', filename='logo.jpg')
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
            insert_query = "INSERT INTO ratings (movie_id, movie_title, rating, ruser) VALUES (%s, %s, %s, %s)"

            if user_id is not None:  # Check if the user ID exists
                cursor.execute(insert_query, (movie_id, movie_title, rating, user_id))
                cnx.commit()
                return "Rating submitted successfully"
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



@app.route('/popular_movies.html')
def popular_movies():
    image_url = url_for('static', filename='logo.jpg')
    return render_template('popular_movies.html', image_url=image_url)


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
        user = cursor.fetchone()

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
    return redirect('/login.html')


# Load the dataset
df = pd.read_excel('./preprocessed_imdb.xlsx')


# Define the API endpoint for movie recommendations
@app.route('/recommend', methods=['GET'])
def recommend_movies():

    user_input = request.args.get('user_input')
    page = request.args.get('page', default=1, type=int)
    in_page = request.args.get('in_page', default=10, type=int)

    image_url = url_for('static', filename='logo.jpg')

    if user_input is None:
        print("From Initial")
        return render_template('recommend.html', image_url=image_url)
    else:
        user_query = preprocess_query(user_input)  # Call function to preprocess user input

        # function to get the average vector for the user query
        user_query_vector = get_average_vector(user_query)

        print("User query vector : ", user_query_vector)

        # Use the 'convert_to_array' function to convert the 'vector' column
        df['vector'] = df['vector'].apply(convert_to_array)

        # Convert df['vector'] to a 2D array (required for cosine_similarity)
        df_vectors = np.array(df['vector'].tolist())

        # Find the indices and similarity scores of the top 'n' nearest vectors in df['vector'] to the user_query_vector
        top_n_indices, top_n_scores = find_top_n_nearest_vector_indices(user_query_vector, df_vectors, n=page * in_page)

        # Get the details of the top 'n' nearest movies and their genres
        top_n_movies = df.loc[top_n_indices]

        # Add the similarity scores to the DataFrame
        top_n_movies['Similarity Score'] = top_n_scores

        top_n_movies = top_n_movies[['Title', 'Genre', 'Similarity Score']]        
        
        print("Top n movies : ", top_n_movies)
        return render_template('recommend.html', image_url=image_url, user_input=user_input, top_n_movies=top_n_movies)





# @app.route('/recommend')
# def recommend():
#     image_url = url_for('static', filename='logo.jpg')
    
#     return render_template('recommend.html', image_url=image_url)


if __name__ == "__main__":
    app.debug = True
    app.run()
