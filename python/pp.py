# %matplotlib inline

import pandas
from sklearn.model_selection import train_test_split
import numpy as np
# import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation

def music():
    triplets_file = './10000.txt'
    songs_metadata_file = './song_data.csv'
    
    song_df_1 = pandas.read_table(triplets_file,header=None)
    song_df_1.columns = ['user_id', 'song_id', 'listen_count']
    
    song_df_2 =  pandas.read_csv(songs_metadata_file)
    
    song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
    
    song_df = song_df.head(10000)
    
    song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
    
    users = song_df['user_id'].unique()
    
    train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
    # print(train_data.head(5))
    
    #Recommenders.item_similarity_recommender_py
    
    is_model = Recommenders.item_similarity_recommender_py()
    is_model.create(train_data, 'user_id', 'song')
    
    #Print the songs for the user in training data
    user_id = users[5]
    user_items = is_model.get_user_items(user_id)
    
    print("------------------------------------------------------------------------------------")
    print("Training data songs for the user userid: %s:" % user_id)
    print("------------------------------------------------------------------------------------")
    
    for user_item in user_items:
         print(user_item)
    
    print("----------------------------------------------------------------------")
    print("Recommendation process going on:")
    print("----------------------------------------------------------------------")
    
    #Recommend songs for the user using personalized model
    is_model.recommend(user_id)
    
    #n = int(input("Which song do you like?\n1.U Smile - Justin Bieber\n2.Paper Planes - M.I.A.\n3.Here Without You - 3 Doors Down\n"))
    #
    #l = ['U Smile - Justin Bieber', 'Paper Planes - M.I.A.', 'Here Without You - 3 Doors Down']
    
    
    print(is_model.get_similar_items([l[1]]))
music()