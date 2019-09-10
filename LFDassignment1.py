#!/usr/bin/python3
# Author 1: J.F.P. (Richard) Scholtens
# Studentnr.: s2956586
# Author 2: Remy Wang
# Studentnr.: s2212781
# Date: 10/09/2019


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

# COMMENT THIS
def read_corpus(corpus_file, use_sentiment):
    """This function opens a file and retrieves all lines in this file.
    It then removes all whitespace from this line and then creates a list with 
    where the first item is genre, the second item is the sentiment, and the
    third is the id number of the review. Everything after this are the words
    of the review. To retrieve sentiment the variable use_sentiment must be True.
    To use genre's the variable use_sentiment must be False. One of these
    variables will be used as labels. It then returns the documents and labels. """

    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels
    
# a dummy function that just returns its input
def identity(x):
    return x

# COMMENT THIS
X, Y = read_corpus('trainset.txt', use_sentiment=True)
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)

# combine the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )


# COMMENT THIS
classifier.fit(Xtrain, Ytrain)

# COMMENT THIS  
Yguess = classifier.predict(Xtest)

# COMMENT THIS
print(accuracy_score(Ytest, Yguess))

