#!/usr/bin/env python3

import nltk
from sentence_transformers import SentenceTransformer, util
nltk.download('punkt')
nltk.download('stopwords')
SentenceTransformer("all-mpnet-base-v2")