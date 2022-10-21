from asyncio.windows_events import NULL
from flask import Flask, request, render_template, jsonify, url_for
from flask_cors import CORS
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K
import os
from semantic_similarity import match_answer_with_key

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.seterr(divide="ignore", invalid="ignore")

import matplotlib.pyplot as plt
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image


def getTextFromImage():
    rcParams["figure.figsize"] = 10, 10

    reader = easyocr.Reader(["en"])
    output = reader.readtext(cv2.imread(r"./image.jpg"))

    return "".join(i[1] + " " for i in output)


def sent2word(x):
    stop_words = set(stopwords.words("english"))
    x = re.sub("[^A-Za-z]", " ", x)
    x.lower()
    words = x.split()
    return [w for w in words if w not in stop_words]


def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    raw = tokenizer.tokenize(essay)
    return [sent2word(i) for i in raw if len(i) > 0]


def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.0
    # index2word_set = set(model.wv.index2word)
    index2word_set = set(model.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[i])
    vec = np.divide(vec, noOfWords)
    return vec


def getVecs(essays, model, num_features):
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for c, i in enumerate(essays):
        essay_vecs[c] = makeVec(i, model, num_features)
    return essay_vecs


def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True,))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="relu"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["mae"])
    model.summary()
    return model


def convertToVec(text):
    content = text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)
        clean_test_essays = [sent2word(content)]
        testDataVecs = getVecs(clean_test_essays, model, num_features)
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model("final_lstm.h5")
        preds = lstm_model.predict(testDataVecs)
        return str(round(preds[0][0]))


app = Flask(__name__)
CORS(app)
# app.run(use_reloader=False)


@app.route("/", methods=["GET"])
def create_task():
    return render_template("index.html")


@app.route("/upload", methods=["GET"])
def go_to_upload_page():
    return render_template("mainpage.html")


@app.route("/grade", methods=["POST"])
def gradeAnswer():
    K.clear_session()
    # extract image and save to images folder
    # print(request.files)
    # image = request.files["image"]
    # image.save("./image.jpg")
    # print(image)
    # print(request.get_data("image")["image"])
    final_text = request.get_json("text")["text"]
    semantic_strictness = request.get_json("semantic_strictness")["semantic_strictness"]
    answer_key = request.get_json("answer_key")["answer_key"]
    print(semantic_strictness, type(semantic_strictness))
    print(answer_key, type(answer_key))
    score = convertToVec(final_text)
    semantic_score = 0
    if answer_key != "":
        semantic_score = match_answer_with_key(final_text, answer_key, float(semantic_strictness))
    K.clear_session()
    return jsonify({"score": score, "semantic_score": semantic_score}), 201


@app.route("/ocr", methods=["POST"])
def performOCR():
    # K.clear_session()
    # extract image and save to images folder
    print(request.files)
    image = request.files["image"]
    image.save("./image.jpg")
    ocr_text = getTextFromImage()
    # print(image["image"])
    # imagefile = request.files.get("image", "")
    # print(imagefile)
    # print(request.get_data("image")["image"])
    # final_text = request.get_json("text")["text"]
    # score = convertToVec(final_text)
    # K.clear_session()
    return jsonify({"text": ocr_text}), 201
    # return jsonify({"score": score}), 201


if __name__ == "__main__":
    app.run(debug=True)
