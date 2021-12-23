import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# load intents in json file
intents = json.loads(open('intents.json').read())

# containers to store information of intents
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# retrieve information of intents and store them in the container
for intent in intents['intents']:
    # store tags in classes
    classes.append(intent['tag'])
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        # store words in patterns to words
        words.extend(word_list)
        # store (word_list, tag) in documents
        documents.append((word_list, intent['tag']))

# lemmatize words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]

# remove duplicates
words = sorted(set(words))
classes = sorted(set(classes))

# dump words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# define training list
training = []
# empty output list to be copied later
output_empty = [0] * len(classes)

# transform documents from string to numerical values (i.e. 0 or 1) for training
for document in documents:
    bag = [0] * len(words)
    # lemmatize word patterns in document
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    # for each word in words
    for i, word in enumerate(words):
        # set 1 if current word in word patterns, 0 otherwise
        if word in word_patterns:
            bag[i] = 1

    # copy empty output list
    output_row = list(output_empty)
    # set to 1 where value at index of classes is the tag
    output_row[classes.index(document[1])] = 1
    # append bag and output_row to training list
    training.append([bag, output_row])

# randomly reorder training list
random.shuffle(training)
# transform training to numpy array
training = np.array(training, dtype=object)

# get list of list that stores bag of words
train_x = list(training[:, 0])
# get list of list that stores output tags
train_y = list(training[:, 1])

# create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# create sgd optimizer
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# fit the model with our training data
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# save the model
model.save('chatbot_model.h5', hist)
print('Done!')
