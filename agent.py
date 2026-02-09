from pydantic import BaseModel
import random
import string
import dspy
import os
from api_keys import TOGETHER_API_KEY
import util
import numpy as np
from synthetic_users import SYNTHETIC_USERS

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

## load ratings matrix and convert user ratings to binary
titles, ratings_matrix = util.load_ratings('data/ratings.txt')
user_ratings_dict = {user: np.zeros(len(titles)) for user in SYNTHETIC_USERS}
for user, movies in SYNTHETIC_USERS.items():
    for movie in movies:
        user_ratings_dict[user][titles.index(movie)] = 1

class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`, so
    # we define a custom class to represent the date.
    year: int
    month: int
    day: int
    hour: int
    minute: int

class UserProfile(BaseModel):
    name: str
    email: str
    balance: float

class Movie(BaseModel):
    title: str
    start_time: Date
    price: float

class Ticket(BaseModel):
    user_name: str
    movie_title: str
    time: Date

class Request(BaseModel):
    user_request: str
    user_name: str

user_database = {
    "peter": UserProfile(name="Peter", email="peter@gmail.com", balance=42),
    "emma": UserProfile(name="Emma", email="emma@gmail.com", balance=79),
    "jake": UserProfile(name="Jake", email="jake@gmail.com", balance=13),
    "sarah": UserProfile(name="Sarah", email="sarah@gmail.com", balance=36),
    "michael": UserProfile(name="Michael", email="michael@gmail.com", balance=8),
    "lisa": UserProfile(name="Lisa", email="lisa@gmail.com", balance=97),
    "marcus": UserProfile(name="Marcus", email="marcus@gmail.com", balance=59),
    "sophia": UserProfile(name="Sophia", email="sophia@gmail.com", balance=25),
    "chris": UserProfile(name="Chris", email="chris@gmail.com", balance=63),
    "amy": UserProfile(name="Amy", email="amy@gmail.com", balance=91),
}

showtime_database = {
    "Back to the Future": Movie(title="Back to the Future (1985)", start_time=Date(year=2025, month=11, day=13, hour=10, minute=0), price=15.0),
    "Speed": Movie(title="Speed (1994)", start_time=Date(year=2025, month=11, day=13, hour=11, minute=30), price=20.0),
    "Star Wars: Episode VI - Return of the Jedi": Movie(title="Star Wars: Episode VI - Return of the Jedi (1983)", start_time=Date(year=2025, month=11, day=15, hour=13, minute=0), price=18.0),
    "Terminator": Movie(title="Terminator, The (1984)", start_time=Date(year=2025, month=11, day=15, hour=18, minute=0), price=14.0),
    "Star Wars: Episode V - The Empire Strikes Back": Movie(title="Star Wars: Episode V - The Empire Strikes Back (1980)", start_time=Date(year=2025, month=11, day=15, hour=20, minute=0), price=16.5),
    "Matrix": Movie(title="Matrix, The (1999)", start_time=Date(year=2025, month=11, day=15, hour=22, minute=0), price=19.0),
    "Silence of the Lambs": Movie(title="Silence of the Lambs, The (1991)", start_time=Date(year=2025, month=11, day=16, hour=10, minute=15), price=17.0),
    "Fight Club": Movie(title="Fight Club (1999)", start_time=Date(year=2025, month=11, day=16, hour=12, minute=45), price=18.5),
    "Lord of the Rings: The Two Towers": Movie(title="Lord of the Rings: The Two Towers, The (2002)", start_time=Date(year=2025, month=11, day=16, hour=15, minute=0), price=17.5),
    "Lord of the Rings: The Fellowship of the Ring": Movie(title="Lord of the Rings: The Fellowship of the Ring, The (2001)", start_time=Date(year=2025, month=11, day=16, hour=17, minute=30), price=17.0),
    "Pulp Fiction": Movie(title="Pulp Fiction (1994)", start_time=Date(year=2025, month=11, day=16, hour=19, minute=45), price=15.5),
    "Star Wars: Episode IV - A New Hope": Movie(title="Star Wars: Episode IV - A New Hope (1977)", start_time=Date(year=2025, month=11, day=16, hour=22, minute=0), price=16.0),
    "Titanic": Movie(title="Titanic (1997)", start_time=Date(year=2025, month=11, day=15, hour=10, minute=0), price=20.0)
}

ticket_database = {}
request_database = {}


## defining tools and helper functions for the tools

def _generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))

def binarize(ratings, threshold=2.5):
    """
    Return a binarized version of the given matrix.
    To binarize a matrix, replace all entries above the threshold with 1.
    and replace all entries at or below the threshold with a -1.
    Entries whose values are 0 represent null values and should remain at 0.

    :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
    :param threshold: Numerical rating above which ratings are considered positive
    :returns: a binarized version of the movie-rating matrix
    """
    
    ########################################################################
    # TODO: Binarize the supplied ratings matrix.                          #
    #                                                                      #
    # WARNING: Do not use self.ratings directly in this function.          #
    ########################################################################

    # The starter code returns a new matrix shaped like ratings but full of
    # zeros.
    binarized_ratings = np.zeros_like(ratings)

    ########################################################################
    #                        END OF YOUR CODE                              #
    ########################################################################
    return binarized_ratings

def similarity(u, v):
    """
    Calculate the cosine similarity between two vectors.
    You may assume that the two arguments have the same shape.
    :param u: one vector, as a 1D numpy array
    :param v: another vector, as a 1D numpy array
    :returns: the cosine similarity between the two vectors
    """
    ########################################################################
    # TODO: Compute cosine similarity between the two vectors.             #
    ########################################################################
    similarity = 0
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################
    return similarity

def recommend_movies(user_name: str, k=3):
    """
    Generate a list of indices of movies to recommend using collaborative
        filtering.

    You should return a collection of `k` indices of movies recommendations.

    As a precondition, user_ratings have been loaded for you based on the provided user_name.

    Remember to exclude movies the user has already rated!

    :returns: a list of k movie titles corresponding to movies in
    ratings_matrix, in descending order of recommendation.
    """
    user_profile = user_database[user_name.lower()]
    user_name = user_profile.name
    user_ratings = user_ratings_dict[user_name]

    ########################################################################
    # TODO: Implement collaborative filtering to generate a list of movie indices to recommend to the user.
    ########################################################################
    # Populate this list with k movie indices to recommend to the user.
    recommendations = []
    
    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################

    ## convert the movie indices to movie titles
    result_titles = [titles[movie_index] for movie_index in recommendations]
    return result_titles

def general_qa(user_request: str):
    """
    Answer a general question about the airline by making an LLM call.
    """
    lm = dspy.LM("together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1")
    dspy.configure(lm=lm)

    response = lm(messages=[{"role": "user", "content": user_request}])
    return response

def book_ticket(user_name: str, movie_title: str):
    """
    Book a ticket for the given user and movie title. Tile must be one of: 
    [Back to the Future, Speed, Star Wars: Episode VI - Return of the Jedi, 
    Terminator, Star Wars: Episode V - The Empire Strikes Back, Matrix, 
    Silence of the Lambs, Fight Club, Lord of the Rings: The Two Towers, 
    Lord of the Rings: The Fellowship of the Ring, Pulp Fiction, 
    Star Wars: Episode IV - A New Hope, Titanic]
    """
   
    ########################################################################
    ## TODO: Implement the book_ticket tool                                #
    ########################################################################

    ########################################################################
    #                          END OF YOUR CODE                            #
    ########################################################################
    return f"Ticket booked successfully for {user_name} for the movie {movie_title}. The ticket number is {ticket_number}. Your new balance is {user_profile.balance}."


## TODO: implement other tools for your agent

## Integrating tools into an LLM agent

class MovieTicketAgent(dspy.Signature):
    """
    You are a movie ticket agent that helps user book and manage movie tickets.

    You are given a list of tools to handle user request, and you should 
    decide the right tool to use in order to fulfill users' request.
    """
    
    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc="Message that summarizes the process result, and the information users need, e.g., the ticket number if a new ticket is booked."
    )

dspy.configure(lm=dspy.LM("together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"))
react_agent = dspy.ReAct(
    MovieTicketAgent,
    tools = [
        recommend_movies,
        general_qa,
        ########################################################################
        ## TODO: add other tools for your agent here
        ########################################################################

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
    ]
)