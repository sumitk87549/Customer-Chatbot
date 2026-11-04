import random
import json
import numpy as np
import pickle
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("Data/intents.json").read()) # intent.json loaded in variable intents

words = []
classes = []
documents = []
ignoreLetters = ["?","!",".",","]

for intent in intents['intents']: # intent dictionary (from intent.json)
    for pattern in intent['patterns']: # patterns key in intent dict (greetings in json file)
        wordList = nltk.word_tokenize(pattern) # tokenized (broken into base words) patters (greetings) in WordList
        words.extend(wordList) # words list now contains tokenized patterns (greetings)
        documents.append((wordList, intent['tag'])) # documents list now contains tuples of wordList (tokenized greetings) and intent['tag'] - main heading in each sub item of dict
        if intent['tag'] not in classes: # classes contains all tags which has been processed till now...
            classes.append(intent['tag'])  

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters] # take words(tokenized patterns/greetings) - remove ignore characters(.,?!) - lemmatize remaining words

words = sorted(set(words)) # remove repeated, sort and then put the sorted/cleaned tokenized patterns/greetings

classes = sorted(set(classes)) # unique classes, sort them and put back into classes

pickle.dump(words, open("words.pkl", "wb")) # saves in local memory for later use
pickle.dump(classes, open("classes.pkl", "wb"))
training = []
outputEmpty = [0] * len(classes) # [0,0,0.... len(classes)]

for document in documents:
    bag = []
    wordPatterns = documents[0] # wordPatterns = tokenized greetings from (tokenized greetings, tag)
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns] # lowerCase each greeting token then lemmatize and put back into wordPatterns
    for word in words: # if our unique-sorted-tokenized-patterns/greetings in wordPatterns (lemmatized-lowercase-tokenized-greetings/patterns) then put it in bag list
        bag.append(1) if words in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty) # outputRow = List ( [0,0,0.... len(classes)] )
    outputRow[classes.index(document[1])] = 1 # outputRow 
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:,: len(words)]
trainY = training[:,len(words)]
