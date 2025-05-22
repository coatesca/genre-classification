# Imports

import pickle
import os

# Changing directory

# Creating app

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 
# Introduction
df = pd.read_csv('netflix_dataset10')
# 
st.title('Genre Classification Model')
st.text('The following linear regression model aims to classify media genres by their descriptions.')
# 
# The dataset
# 
st.header('The Dataset')
st.write(df.head())
st.text('The final dataset features are genre and description with the following four potential classes: Children & Family, Documentaries, International, & Stand-Up Comedy')
# 
# Bar chart
# 
genre_groupby = df.groupby('listed_in').count()
st.bar_chart(genre_groupby)
# 
random_genre = st.selectbox('See a description based on the available genres', ('Children & Family', 'Documentaries', 'International', 'Stand-Up Comedy'),
                             placeholder='Select a genre')
# 
if random_genre == 'Children & Family':
    st.write(df[df['listed_in'] == 'Children & Family'].sample(1)['description'])
if random_genre == 'Documentaries':
   st.write(df[df['listed_in'] == 'Documentaries'].sample(1)['description'])
if random_genre == 'International':
   st.write(df[df['listed_in'] == 'International'].sample(1)['description'])
if random_genre == 'Stand-Up Comedy':
   st.write(df[df['listed_in'] == 'Stand-Up Comedy'].sample(1)['description'])
# 
# Most populated words by genre graphs (no stop words)
# 
nltk.download('stopwords')
stop = stopwords.words('english')
df['des_nostop'] = df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
counts = df.set_index('listed_in')['des_nostop'].str.split().explode().groupby(level=0).apply(pd.value_counts)
# 
st.header('Generate a Graph: Most Frequent Words by Genre (no stop words')
# 
# Children & Family
st.text('Children & Family')
random_count1 = st.selectbox('Select word count', (10, 15, 20, 25, 30),
                             placeholder=10)
if random_count1 == 10:
   st.bar_chart(counts['Children & Family'][:10])
if random_count1 == 15:
   st.bar_chart(counts['Children & Family'][:15])
if random_count1 == 20:
   st.bar_chart(counts['Children & Family'][:20])
if random_count1 == 25:
   st.bar_chart(counts['Children & Family'][:25])
if random_count1 == 30:
    st.bar_chart(counts['Children & Family'][:30])
# 
# Documentaries
st.text('Documentaries')
random_count2 = st.selectbox('Select word count', (10, 15, 20, 25, 30),
                             placeholder=10)
if random_count2 == 10:
   st.bar_chart(counts['Documentaries'][:10])
if random_count2 == 15:
   st.bar_chart(counts['Documentaries'][:15])
if random_count2 == 20:
   st.bar_chart(counts['Documentaries'][:20])
if random_count2 == 25:
   st.bar_chart(counts['Documentaries'][:25])
if random_count2 == 30:
    st.bar_chart(counts['Documentaries'][:30])
# 
# International
# 
st.text('International')
random_count3 = st.selectbox('Select word count', (10, 15, 20, 25, 30),
                            placeholder=10)
if random_count3 == 10:
   st.bar_chart(counts['International'][:10])
if random_count3 == 15:
   st.bar_chart(counts['International'][:15])
if random_count3 == 20:
   st.bar_chart(counts['International'][:20])
if random_count3 == 25:
    st.bar_chart(counts['International'][:25])
if random_count3 == 30:
   st.bar_chart(counts['International'][:30])
# 
# Stand-Up Comedy
# 
st.text('Stand-Up Comedy')
random_count4 = st.selectbox('Select word count', (10, 15, 20, 25, 30),
                             placeholder=10)
if random_count4 == 10:
   st.bar_chart(counts['Stand-Up Comedy'][:10])
if random_count4 == 15:
   st.bar_chart(counts['Stand-Up Comedy'][:15])
# if random_count4 == 20:
#   st.bar_chart(counts['Stand-Up Comedy'][:20])
# if random_count4 == 25:
#   st.bar_chart(counts['Stand-Up Comedy'][:25])
# if random_count4 == 30:
#   st.bar_chart(counts['Stand-Up Comedy'][:30])
# 
# 
# # Load model
# 
# st.header('Logistic Regression Model with GloVe')
# st.text('The final model implements a pre-trained embedding technique (GloVe) to predict one of the four previously mentioned genres.')
# 
# user_des = st.text_input()
# 
# with open('model.pkl', 'rb') as file:
#   model = pickle.load(file)
# 
# if st.form_submit_button('Predict genre'):
#   data = pd.DataFrame(user_des)
