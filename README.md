# Project Description 

The goal of the following model explorations is to predict the genre of various media types (films, TV shows) based on their descriptions. This could be useful for identifying potential strengths and weaknesses in written marketing materials. 

The data is compromised of over 8,000 TV shows, films, and documentaries taken from Netflix in 2021 and sourced from Datacamp.com. Each piece of media has a variety of information, but for the purposes of the models only genre (listed_in) and the description are needed. 

As discovered in the EDA phase, there are multiple genres listed for some media. Therefore media was duplicated based on the number of genres listed so that each genre would be faithfully represented. Still, some genres were either vague (e.g. classics) or had few data points (less than 1,000), so these genres were dropped. Some genres were also quite similar so they were joined (e.g. Anime Features and Anime Series). 

Several models were explored, including logistic regression, SVM, and random forest. However, the accuracy scores were quite low for all of them (less than 40%). Even with the implementation of a pre-trained embedding technique like GloVe, the accuracy improved the most for the logistic regression model but was still very low (less than 50%). 

Moving forward, more pre-processing techniques were explored with the logistic regression model as the primary focus. This included removing stop words, lemmatizing the data, and even trying to predict genres based on only nouns. However all of these methods never improved the accuracy, sometimes the accuracy even decreased. Choosing the right combination of genres improved the accuracy significantly. 

By disregarding sample counts for each genre and focusing solely on accuracy scores, the final model was a logistic regression model trained with GloVe and had an accuracy of 73%. It was able to predict on the following four genres: Children & Family, Documentaries, International, and Stand-Up Comedy. Although Stand-Up Comedy had one of the lowest sample counts, it seemed to be a very cohesive genre in terms of language. Compared to Drama, which was one of the most populated genres, Stand-Up Comedy demonstrated consistent accuracy improvement with each model iteration, while Drama consistently suffered or brought all of the other genre accuracies down. 

# Project Files 

- Notebook with all relevent code and detailed model processes
- Dataset
- Powerpoint presentation
- App
- Requirements file

# App URL

genre-classification-9b448xtahkz7kqdebryoyf.streamlit.app
