import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import EnglishStemmer
import scipy.io
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
from KPlusPlus import DetK



pdfFileObj = open('nursing.pdf','rb')     #'rb' for read binary mode
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
content=pdfReader.getPage(18).extractText()
for i in range(pdfReader.numPages)[19:-3]:
    pageObj = pdfReader.getPage(i)         #'9' is the page number
    content+=pageObj.extractText()
# print (content)

chapter_dico={}
l=content.split("\n")[0]
for line in content.split("\n")[1:]:
    if "CHAPTER" in line:
        l=line
        chapter_dico[l]=""
    else:
        chapter_dico[l]+=line

# print (chapter_dico)
stemmer=EnglishStemmer()

analyzer = CountVectorizer(min_df=0.1, max_df=1.0, ngram_range=(1, 3), analyzer='word',encoding='string',stop_words='english', max_features=70).build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
vectorizer=CountVectorizer(min_df=0.1, max_df=1.0, ngram_range=(1, 3), analyzer=stemmed_words,encoding='string',stop_words='english', max_features=50)
vectorizer.fit_transform(chapter_dico.values())
X=vectorizer.get_feature_names()
X=[x.encode('UTF8') for x in X]

def upper_words(sentence):
    upper=""
    list= sentence.split(" ")
    for word in list:
        if word.isupper():
            upper+=" "+word
    return upper

Y=[upper_words(x.encode('UTF8')) for x in chapter_dico.keys()]
tetha=vectorizer.transform(chapter_dico.values())
beta=tetha.toarray()
columns=list(Y)
columns.insert(0,'words')
columns.append('classes')
data=pd.DataFrame(columns=columns)
data['words']=X
for i in range(len(Y)+1)[1:]:
    for j in range(len(X)):
        data.iloc[j,i]=int(beta[i-1][j])


inputs=np.array(data.drop(['words','classes'],1))
inputs=preprocessing.scale(inputs)

# def number_clusters(inputs):
#     range_n_clusters=range(len(inputs))[2:-1]
#     max=0
#     for n_clusters in range_n_clusters:
#         clusterer = KMeans(n_clusters=n_clusters)
#         cluster_labels = clusterer.fit_predict(inputs)
#         silhouette_avg = silhouette_score(inputs, cluster_labels)
#         if silhouette_avg>=max:
#             max=silhouette_avg
#             clusters=n_clusters
#     print clusters
#     return clusters

clf=DetK()
num_clusters=clf.num_clusters(inputs, 20)
print num_clusters
clf=DetK(n_clusters=num_clusters)
classes=clf.fit_predict(inputs)

for i in range(len(classes)):
    data.iloc[i,len(Y)+1]=classes[i]
print data['words'][data['classes']==0]
print data['words'][data['classes']==1]

