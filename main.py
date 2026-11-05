import random
import json
import numpy as np
import pickle
import tensorflow as tf
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
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
        if intent['tag'] not in classes: # classes contains all tags which has been processed till now, tags-order-as-first_match...
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
    wordPatterns = document[0] # wordPatterns = tokenized greetings from (tokenized greetings, tag)
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns] # lowerCase each greeting token then lemmatize and put back into wordPatterns
    for word in words: # if our unique-sorted-tokenized-patterns/greetings in wordPatterns (lemmatized-lowercase-tokenized-greetings/patterns) then put it in bag list
        bag.append(1) if words in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty) # outputRow = List ( [0,0,0.... len(classes)] ) - make a fresh COPY of the blank answer sheet (copy, so we don't edit the original blank one)
    outputRow[classes.index(document[1])] = 1 # document[1] is the tag of this pattern, classes.index(...) finds its position, and we flip that slot to 1 -> "this pattern belongs to THIS class" (one-hot encoding)
    training.append(bag + outputRow) # glue the bag (input part) and outputRow (answer part) into one long row and add it to the dataset
 
random.shuffle(training) # mix up the rows randomly so the model doesn't see all "greeting" rows first, then all "goodbye" rows etc.
training = np.array(training) # convert the list of lists into a numpy 2D array (rows = patterns, columns = bag + output) so we can slice it easily
 
trainX = training[:,:len(words)] # INPUT data: all rows, first len(words) columns = just the bag-of-words part
trainY = training[:,len(words):] # OUTPUT data: all rows, but ONLY the single column at index len(words)
# ^ HEADS UP (bug): this picks just ONE column (the first class slot). You want ALL the output columns, so it should be training[:, len(words):] (note the colon after the comma-part) 

model = tf.keras.Sequential() # create a base model
model.add(tf.keras.layers.Input(shape=(len(trainX[0]),)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov= True)

model.compile(loss="categorical_crossentropy", optimizer= sgd, metrics=["accuracy"])

hist = model.fit(
    np.array(trainX), np.array(trainY), epochs=250, batch_size=5, verbose=1
)


model.save("chatbot_model.keras")
print("Done")


