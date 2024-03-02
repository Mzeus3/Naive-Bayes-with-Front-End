import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
#-----------------------------------------------------------------------------#

#--Front End Interaction--#
st.set_page_config(page_title="Naive Bayes", page_icon="⚡",layout="centered")
with st.container():
    st.title("Movie Review Classifier")
    st.subheader("Predicting the Outcome Using Naive Bayes")
    st.subheader("Mahad Zubair - 039")
    st.subheader("Ahmed Moeez Khan - 004")
    st.write("----------")

text = st.text_area("Enter Review Here")
#-----------------------------------------------------------------------------#
if st.button("Classify"):
    data = pd.read_csv("IMDBDataset.csv")  

#--Implementing the Naive Bayes classifier--#
    X = data['review']
    y = data['class']
    vector = CountVectorizer()
    X_vectorized = vector.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=0)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    review = vector.transform([text])
    
 #-----------------------------------------------------------------------------#   
#--Predicting the class of the input review--#
    pred = classifier.predict(review)
    prediction = "Positive" if pred[0] == 1 else "Negative"
    st.write("Sentiment:", prediction)
    st.write("Predicted class:", pred[0])
#-----------------------------------------------------------------------------#
#--Calculating & Displaying the accuracy score and confusion matrix--#
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    confusion_matrix = confusion_matrix(y_test, classifier.predict(X_test))
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write(confusion_matrix)
