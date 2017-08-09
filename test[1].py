from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# Set up environment
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import pickle
import os
import re

from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords
# nltk.download("stopwords")

with open ("ln_clin_nursing_final.txt", "r") as myfile:
    data=myfile.read()
# print data
flatlist=[word for word in data.split(' ') ]
import collections
count_dict=collections.Counter(flatlist)
vocablist = [key for key in count_dict if nltk.pos_tag([key])[0][1]not in ["CC","DT","TO","PRP", "IN" ,"RB"] and count_dict[key] > 100]
print vocablist
vocablist.append("hours of sleep")

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=vocablist, ngram_range=(1,3))
X=vectorizer.transform([data])
print X



# print nltk.pos_tag(['not'])[0][1]
#nltk.download()




