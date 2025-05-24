# Imports

import pickle
import streamlit as st
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Introduction

st.title('Genre Classification Model')
st.text('The following linear regression model aims to classify media genres by their descriptions.')

# The dataset

df = pd.read_csv('netflix_dataset10.csv')
 
st.header('The Dataset')
st.write(df.head())
st.text('The final dataset features are genre and description with the following four potential classes: Children & Family, Documentaries, International, & Stand-Up Comedy')

# Bar chart

genre_groupby = df.groupby('listed_in').count()
st.bar_chart(genre_groupby)
#
random_genre = st.selectbox('See a description based on the available genres', ('Children & Family', 'Documentaries', 'International', 'Stand-Up Comedy'),
                             placeholder='Select a genre')
 
if random_genre == 'Children & Family':
   st.write(df[df['listed_in'] == 'Children & Family'].sample(1)['description'])
if random_genre == 'Documentaries':
   st.write(df[df['listed_in'] == 'Documentaries'].sample(1)['description'])
if random_genre == 'International':
   st.write(df[df['listed_in'] == 'International'].sample(1)['description'])
if random_genre == 'Stand-Up Comedy':
   st.write(df[df['listed_in'] == 'Stand-Up Comedy'].sample(1)['description'])
 
# Most populated words by genre graphs (no stop words)
 
nltk.download('stopwords')
stop = stopwords.words('english')
df['des_nostop'] = df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
counts = df.set_index('listed_in')['des_nostop'].str.split().explode().groupby(level=0).apply(pd.value_counts)
 
st.header('Most Frequent Words by Genre (no stop words)')
 
# Children & Family
st.text('Children & Family')
random_count1 = st.selectbox('Select word count', [10, 15, 20, 25, 30])

if random_count1 == 10:
   st.bar_chart(counts['Children & Family'][:10])
elif random_count1 == 15:
   st.bar_chart(counts['Children & Family'][:15])
elif random_count1 == 20:
   st.bar_chart(counts['Children & Family'][:20])
elif random_count1 == 25:
   st.bar_chart(counts['Children & Family'][:25])
elif random_count1 == 30:
   st.bar_chart(counts['Children & Family'][:30])
 
# Documentaries
st.text('Documentaries')
random_count2 = st.selectbox('Select word count', [10, 15, 20, 25, 30], key = 'sb_d')

if random_count2 == 10:
   st.bar_chart(counts['Documentaries'][:10])
elif random_count2 == 15:
   st.bar_chart(counts['Documentaries'][:15])
elif random_count2 == 20:
   st.bar_chart(counts['Documentaries'][:20])
elif random_count2 == 25:
   st.bar_chart(counts['Documentaries'][:25])
elif random_count2 == 30:
   st.bar_chart(counts['Documentaries'][:30])
 
# International
 
st.text('International')
random_count3 = st.selectbox('Select word count', [10, 15, 20, 25, 30], key = 'sb_i')

if random_count3 == 10:
   st.bar_chart(counts['International'][:10])
elif random_count3 == 15:
   st.bar_chart(counts['International'][:15])
elif random_count3 == 20:
   st.bar_chart(counts['International'][:20])
elif random_count3 == 25:
   st.bar_chart(counts['International'][:25])
elif random_count3 == 30:
   st.bar_chart(counts['International'][:30])
 
# Stand-Up Comedy
 
st.text('Stand-Up Comedy')
random_count4 = st.selectbox('Select word count', [10, 15, 20, 25, 30], key = 'sb_suc')

if random_count4 == 10:
   st.bar_chart(counts['Stand-Up Comedy'][:10])
elif random_count4 == 15:
   st.bar_chart(counts['Stand-Up Comedy'][:15])
elif random_count4 == 20:
   st.bar_chart(counts['Stand-Up Comedy'][:20])
elif random_count4 == 25:
   st.bar_chart(counts['Stand-Up Comedy'][:25])
elif random_count4 == 30:
   st.bar_chart(counts['Stand-Up Comedy'][:30])
 
# Load model
 
st.header('Logistic Regression Model with GloVe')
st.text('The final model implements a pre-trained embedding technique (GloVe) to predict one of the four mentioned genres.')

# User input

with open('model.pkl', 'rb') as file:
   model = pickle.load(file)


user_des = st.text_input('Enter a genre description')
user_des = [user_des]

# Creating the vectorizer

vectorizer = CountVectorizer(stop_words='english')

# Converting the text to numeric data

new_des = vectorizer.fit_transform(user_des)

CountVectorizedData= pd.DataFrame(new_des.toarray(), columns=vectorizer.get_feature_names_out())

# Defining an empty dictionary to store the values

GloveWordVectors = {}

# Reading Glove Data

model_path = 'glove.6B.50d.txt'

with open('glove.6B.50d.txt', 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], "float")
        GloveWordVectors[word] = vector

# Creating the list of words which are present in the Document term matrix

WordsVocab=CountVectorizedData.columns[:]

# Function to encode data

def FunctionText2Vec(inpTextData):

    # Converting the text to numeric data
    X = vectorizer.transform(inpTextData)
    CountVecData=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Creating empty dataframe to hold sentences
    W2Vec_Data=pd.DataFrame()

    # Looping through each row for the data
    for i in range(CountVecData.shape[0]):

        # initiating a sentence with all zeros
        Sentence = np.zeros(50)

        # Looping thru each word in the sentence and if it's present in the Glove model then storing its vector
        for word in WordsVocab[CountVecData.iloc[i, : ]>=1]:

            #print(word)
            if word in GloveWordVectors.keys():
                Sentence=Sentence+GloveWordVectors[word]

        # Appending the sentence to the dataframe
        W2Vec_Data = pd.concat([W2Vec_Data, pd.DataFrame([Sentence])], ignore_index=True)
    return(W2Vec_Data)

# Predict button

if st.button('Predict genre'):
   W2Vec_Data= FunctionText2Vec(user_des)
   pred = model.predict(W2Vec_Data)
   st.write(f"The predicted genre is: {pred}")
