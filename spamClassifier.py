import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

dataSet = pd.read_csv("emails.csv")

nltk.download("stopwords")


def extract_punctuations(text):
    text_without_punctuation = [character for character in text if character not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)

    return text_without_punctuation


def extract_stopwords(text):
    text_without_stop_words = [word for word in extract_punctuations(text).split() if word.lower() not in stopwords.words("english")]

    return text_without_stop_words


print(dataSet["text"].head().apply(extract_stopwords))

message = CountVectorizer(analyzer=extract_stopwords).fit_transform(dataSet["text"])


xtrain, xtest, ytrain, ytest = train_test_split(message, dataSet['spam'], test_size=0.20, random_state=0)

# setting model Decision Tree Classifier

classifier_DT = DecisionTreeClassifier()
classifier_DT.fit(xtrain, ytrain)

# Setting up the model's report

print("Decision Tree Classifier Classification Report")

predicateTraining_DT = classifier_DT.predict(xtrain)
print(classification_report(ytrain, predicateTraining_DT))

predicateTest_DT = classifier_DT.predict(xtest)
print(classification_report(ytest, predicateTest_DT))

