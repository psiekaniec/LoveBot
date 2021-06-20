import json
import nltk
import numpy as np
import os
import pickle
import random
import socket
import sys
import threading
import Training

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
os.chdir(script_dir)

MODEL_PATH = 'weights.h5'

nltk.download('all')  # need to download all packages

stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

HOST = '127.0.0.1'
PORT = 8181
BUFSIZE = 1024

labels = []
words = []
docs_x = []
docs_y = []
clients = []
nicknames = []
training = []
output = []
model = None
encoding = 'utf-8'

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))

server.listen()


def bag_of_words(s):
    bag = [0 for _ in range(len(words))]
    s_words = []
    try:
        s_string = str(s, encoding)  # without that line code is crashing every time - required conversion bytes-object to string
        s_words = nltk.word_tokenize(s_string)
        s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]
    except Exception as ex:
        print(ex.args)

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    responses = []
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results_index = np.argmax(res)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)


def handler(client):
    while True:
        if client in clients:
            try:
                msg = client.recv(BUFSIZE)
                print(f"{nicknames[clients.index(client)]} says {msg}")
                broadcast(msg)
                broadcast(predict_class(msg))
            except OSError as oserror:
                print(oserror.args)
                index = clients.index(client)
                clients.remove(client)
                client.close()
                nick = nicknames[index]
                nicknames.remove(nick)
            except ValueError as verror:
                print(verror.args)
                index = clients.index(client)
                clients.remove(client)
                client.close()
                nick = nicknames[index]
                nicknames.remove(nick)
            except Exception as exc:
                print(exc.args)
                index = clients.index(client)
                clients.remove(client)
                client.close()
                nick = nicknames[index]
                nicknames.remove(nick)


def broadcast(message):
    try:
        for client in clients:
            if type(message) == str:
                message = message.encode(encoding)
            client.send(message + b'\n')
    except Exception as ex:
        print(ex.args)


def receive():
    while True:
        client, address = server.accept()
        print(f"Connected to {str(address)}")

        try:
            client.send("NICK".encode(encoding))
            nick = client.recv(BUFSIZE)

            nicknames.append(nick)
            clients.append(client)

            print(f'Nickname of the client is {nick}')
            broadcast(f"{nick} connected to the chat\n".encode(encoding))
            client.send("Connected to the server".encode(encoding))

            thread = threading.Thread(target=handler, args=(client,))
            thread.start()
        except ConnectionAbortedError as cae:
            print(cae.args)
            index = clients.index(client)
            clients.remove(client)
            client.close()
            nick = nicknames[index]
            nicknames.remove(nick)


with open("intents.json", encoding=encoding) as file:
    data = json.load(file)

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != '?']
words = sorted(list(set(words)))
pickle.dump(words, open('words.pkl', 'wb'))
words = pickle.load(open('words.pkl', 'rb'))

labels = sorted(labels)

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [lemmatizer.lemmatize(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

model = Training.create_model(training, output)
print("Server is running")
receive()
