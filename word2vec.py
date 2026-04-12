from gensim.models import Word2Vec, keyedvectors
import numpy as np
import gensim.downloader as api

# train your own word2vec model
# Training data
# sentences = [
#     ["i", "love", "sip"],
#     ["sip", "is", "good", "investment"],
#     ["stock", "market", "is", "risky"],
#     ["i", "like", "cricket"]
# ]

# model = Word2Vec(sentences, vector_size=300, window=5, min_count=2)
# # get the vector for a word
# vector = model.wv['sip']
# most_similar = model.wv.most_similar('sip')
# print("Vector for 'sip':", vector)
# print("Most similar words to 'sip':", most_similar)


# skip-gram model
"""
Word2Vec Model Training and Analysis Module
This module demonstrates training a Word2Vec skip-gram model using the text8 dataset
from gensim's pretrained models. It loads a large text corpus, trains a Word2Vec model
with specified parameters, and retrieves word vectors and semantically similar words.
The model is trained every time the script runs because:
1. No pre-trained model is being loaded from disk
2. The model is instantiated fresh with Word2Vec(dataset, ...) on each execution
3. The trained model is not saved to a file for later reuse
To avoid retraining:
- Save the model after training: model.save('word2vec_model.bin')
- Load it on subsequent runs: model = Word2Vec.load('word2vec_model.bin')
Parameters:
    vector_size (int): Dimensionality of word vectors (300)
    window (int): Context window size for training (5 words on each side)
    min_count (int): Minimum word frequency threshold (2 occurrences)
Returns:
    None: Prints word vectors and most similar words to 'king'
"""

dataset  = api.load("text8")    
# length of the dataset
dataset = list(dataset)
print("Length of the dataset:", len(dataset))
print("Dataset loaded successfully.")
model_path = "word2vec_model.bin"
try:
    model = Word2Vec.load(model_path)
    print("Model loaded from file.")
except FileNotFoundError:
    model = Word2Vec(dataset, vector_size=300, window=5, min_count=2)
    model.save(model_path)
    print("Model trained and saved to file.")

if __name__ == "__main__":
    while True:
        word = input("Enter a word to get its vector and most similar words (or 'exit' to quit): ")
        if word.lower() == 'exit':
            break
        if word in model.wv:
            vector = model.wv[word]
            most_similar = model.wv.most_similar(word)
            print(f"Vector for '{word}':", vector)
            print(f"Most similar words to '{word}':", most_similar)
        else:
            print(f"'{word}' not found in the vocabulary.")