# the function for sentiment analysis is refering to the tutorial in this webpage
# https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
import stanfordnlp
import requests
import json

# Make sure to downloaded the StanfordNLP English model and other essential tools using,
stanfordnlp.download('en')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# Getting data for game id= 271590 (GTAV)
response = requests.get("http://store.steampowered.com/appreviews/271590?json=1")

# Reading the content
content = response.json()


# From language preprocessing to sentiment analysis
def aspect_sentiment_analysis(txt, stop_words, nlp):
    
    txt = txt.lower() # LowerCasing the given Text
    sentList = nltk.sent_tokenize(txt) # Splitting the text into sentences

    fcluster = []
    totalfeatureList = []
    finalcluster = []
    dic = {}

    for line in sentList:
        newtaggedList = []
        txt_list = nltk.word_tokenize(line) # Splitting up into words
        taggedList = nltk.pos_tag(txt_list) # Doing Part-of-Speech Tagging to each word

        newwordList = []
        flag = 0
        for i in range(0,len(taggedList)-1):
            if(taggedList[i][1]=="NN" and taggedList[i+1][1]=="NN"): # If two consecutive words are Nouns then they are joined together
                newwordList.append(taggedList[i][0]+taggedList[i+1][0])
                flag=1
            else:
                if(flag==1):
                    flag=0
                    continue
                newwordList.append(taggedList[i][0])
                if(i==len(taggedList)-2):
                    newwordList.append(taggedList[i+1][0])

        finaltxt = ' '.join(word for word in newwordList) 
        new_txt_list = nltk.word_tokenize(finaltxt)
        wordsList = [w for w in new_txt_list if not w in stop_words]
        taggedList = nltk.pos_tag(wordsList)

        doc = nlp(finaltxt) # Object of Stanford NLP Pipeleine
        
        # Getting the dependency relations betwwen the words
        dep_node = []
        for dep_edge in doc.sentences[0].dependencies:
            dep_node.append([dep_edge[2].text, dep_edge[0].index, dep_edge[1]])

        # Coverting it into appropriate format
        for i in range(0, len(dep_node)):
            if (int(dep_node[i][1]) != 0):
                dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]

        featureList = []
        categories = []
        for i in taggedList:
            if(i[1]=='JJ' or i[1]=='NN' or i[1]=='JJR' or i[1]=='NNS' or i[1]=='RB'):
                featureList.append(list(i)) # For features for each sentence
                totalfeatureList.append(list(i)) # Stores the features of all the sentences in the text
                categories.append(i[0])

        for i in featureList:
            filist = []
            for j in dep_node:
                if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
                    if(j[0]==i[0]):
                        filist.append(j[1])
                    else:
                        filist.append(j[0])
            fcluster.append([i[0], filist])
            
    for i in totalfeatureList:
        dic[i[0]] = i[1]
    
    for i in fcluster:
        if(dic[i[0]]=="NN"):
            finalcluster.append(i)
        
    return(finalcluster)

nlp = stanfordnlp.Pipeline()
stop_words = set("english")



review1 = content["reviews"][10]["review"]
print("review1" + review1)
print(aspect_sentiment_analysis(review1, stop_words, nlp))
print("review2")
review2 = content["reviews"][5]["review"]
print(aspect_sentiment_analysis(review2, stop_words, nlp))
print("review3")
review3 = content["reviews"][1]["review"]
print(aspect_sentiment_analysis(review3, stop_words, nlp))
print("review4")
review4 = "Great base game, online is super toxic"
print("review4")
print(aspect_sentiment_analysis(review4, stop_words, nlp))

review5 = "This game is the ultimate version of one of the greatest games in history and the best open world game that you will ever play!"
print("review5")
print(aspect_sentiment_analysis(review5, stop_words, nlp))

review6 = "nice story mode, nice multi and enjoyable with friends, too funny if u are playing with friends, boring online mode after u purchased wtever u want"
print("review6")
print(aspect_sentiment_analysis(review6, stop_words, nlp))

review7 = "Best open world game!"
print("review7")
print(aspect_sentiment_analysis(review7, stop_words, nlp))

review8 = "I think GTA V is an excellent game and i love this game so much"
print("review8")
print(aspect_sentiment_analysis(review8, stop_words, nlp))
# print(aspect_sentiment_analysis(txt, stop_words, nlp))

review9 = "Nice game"
print("review9")
print(aspect_sentiment_analysis(review9, stop_words, nlp))

review10 = "of course the story was good. the online also good, but hackers and modders ruin online experience which is pain in the ass"
print("review10")
print(aspect_sentiment_analysis(review10, stop_words, nlp))


