# spam_ham_classifier.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # Load Dataset
    df = pd.read_csv(r"C:\Users\lenovo\Downloads\SMSSpamCollection", sep='\t', header=None, names=['label', 'text'])

    # Show first few rows
    print(df.head())

    # Plot the count of spam vs ham
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label', palette='Set2')
    plt.title('Number of Spam vs Ham Messages')
    plt.show()

    # Generate word cloud for Ham
    ham_words = ' '.join(df[df['label'] == 'ham']['text'])
    ham_wc = WordCloud(width=500, height=300, background_color='white').generate(ham_words)

    plt.figure(figsize=(7, 5))
    plt.imshow(ham_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Ham Messages')
    plt.show()

    # Generate word cloud for Spam
    spam_words = ' '.join(df[df['label'] == 'spam']['text'])
    spam_wc = WordCloud(width=500, height=300, background_color='white', colormap='Reds').generate(spam_words)

    plt.figure(figsize=(7, 5))
    plt.imshow(spam_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Spam Messages')
    plt.show()

    # Data Preprocessing
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    X = df['text']
    y = df['label_num']

    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Model Training
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Test on a New Sample
    new_email = ["Congratulations! You won a free cruise ticket. Call now!"]
    new_email_tfidf = vectorizer.transform(new_email)
    prediction = model.predict(new_email_tfidf)

    print("Prediction for New Email:", "Spam" if prediction[0] == 1 else "Ham")

if __name__ == "__main__":
    main()
