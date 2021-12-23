import random
import json
import pickle
import numpy as np
import wikipedia
from termcolor import colored
import os
from google_speech import Speech

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from tensorflow.keras.models import load_model

ERROR_THRESHOLD = 0.9
EVENT_QUIT = 'quit'
EVENT_WIKI = 'wiki'

os.system('color')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
stop_words = set(stopwords.words('english'))


def clean_up_sentence(sentence):
    """
    tokenize the sentence and lemmatize it
    :param sentence: input message
    :return: list of words in sentence
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    """
    First call method clean_up_sentence,
    then create a bag of words representation
    :param sentence: input message
    :return: numpy array of bag of words
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for i, word in enumerate(words):
        if word in sentence_words:
            bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    """
    First call method bag_of_words,
    then use the model to predict the tags related to the sentence.
    sort the tags in descending order of probability
    :param sentence: input message
    :return: list of dictionaries with tag and probability
    """
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'tag': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(tags_list, intents_json):
    """
    Get a random response from the most possible tag.
    :param tags_list: list of dictionaries with tag and probability
    :param intents_json: intents from json file
    :return: list containing response and event
    """
    response_event_list = ["Sorry, I don't understand", ""]
    if not tags_list:
        return response_event_list
    tag = tags_list[0]['tag']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            response_event_list[0] = random.choice(intent['responses'])
            if 'event' in intent.keys():
                response_event_list[1] = intent['event']
            break
    return response_event_list


def speak(text):
    """
    Use google speech to speak text
    See details by following url below
    https://pypi.org/project/google-speech/
    :param text: text to speak
    """
    lang = 'en'
    text += ''
    speech = Speech(text, lang)
    speech.play()


if __name__ == '__main__':
    print('\nChatbot activated!\n')
    while True:
        message = input(colored('You: ', 'green'))
        curr_tags_list = predict_class(message)
        response, event = get_response(curr_tags_list, intents)
        # print response
        print(colored('Chatbot:', 'red'), response)
        # speak response
        try:
            speak(response)
        except Exception as e:
            print('Google speech not working')
            print(e)

        # quit
        if event == EVENT_QUIT:
            quit()
        # search wikipedia
        if event == EVENT_WIKI:
            word_tokens = nltk.word_tokenize(message)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            try:
                summary = wikipedia.summary(''.join(filtered_sentence), sentences=2)
            except Exception as e:
                print('No result found')
                print(e)
            else:
                print(summary)
