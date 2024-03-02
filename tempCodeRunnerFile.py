import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load the IMDB dataset
def load_data():
    data = pd.read_csv('IMDBDataset.csv')
    return data

# Train Naive Bayes classifier
def train_classifier(data):
    X = data['review']
    y = data['class']
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X, y)
    return model

# Main function
def main():
    # Load data
    data = load_data()
    
    # Train the classifier
    model = train_classifier(data)

    # Frontend
    st.title("IMDB Movie Review Sentiment Analysis")
    review = st.text_area("Enter your movie review here:")
    
    if st.button("Classify"):
        if review:
            prediction = model.predict([review])
            if prediction[0] == 'positive':
                st.success("This review is Positive!")
            else:
                st.error("This review is Negative!")
        else:
            st.warning("Please enter a movie review.")

if __name__ == '__main__':
    main()
