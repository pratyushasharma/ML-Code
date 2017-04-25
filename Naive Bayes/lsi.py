# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import re
pattern = re.compile(r'\W+')
from scipy import sparse as sp
from collections import Counter
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import glob, os
#import joblib
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-z', type=int)
parser.add_argument('-k', type=int)
parser.add_argument('--dir', type=str)
parser.add_argument('--doc_in', type=str)
parser.add_argument('--doc_out', type=str)
parser.add_argument('--term_in', type=str)
parser.add_argument('--term_out', type=str)
parser.add_argument('--query_in', type=str)
parser.add_argument('--query_out', type=str)

args = parser.parse_args()
# k = args.k
# z = args.z
# Direct = args.dir
doc_in = args.doc_in
f_doc_in = open(doc_in)
# doc_out = args.doc_out
# term_in = args.term_in
# term_out = args.term_out
# query_in = args.query_in
# query_out = args.query_out

# def get_params():
#     parser = argparse.ArgumentParser(description = 'LSI')    
#     parser.add_argument('-z', dest="dim", type=int)
#     parser.add_argument('-k', dest="numSimilarDocs", type=int)
#     parser.add_argument('-dir', dest="inputDir", type=str)
#     parser.add_argument('-doc_in', dest="inputDoc", type=str)
#     parser.add_argument('-doc_out', dest="outputDoc", type=str)
#     parser.add_argument('-term_in', dest="inputTerm", type=str)
#     parser.add_argument('-term_out', dest="outputTerm", type=str)
#     parser.add_argument('-query_in', dest="inputQuery", type=str)
#     parser.add_argument('-query_out', dest="outputQuery", type=str)
#     # parser.add_argument('-train', action="store", default=1, dest="train", type=int)

#     opts = parser.parse_args(sys.argv[1:])
#     return opts


# def filter_fn(letter):
#     return (letter.isalnum() and not letter.isdigit()) or letter == " "

# ''' the basic version just removes blank words. 
#     Can also remove common words by using a lexicon of common words.
#     Common words should be gotten rid of first or just before performing an svd
# '''
# def removeNoise(word, stop_words):
#     return word not in stop_words and len(word.strip()) != 0

# ''' Gets rid of punctuations, and numbers. Also converts to lowercase'''
# def preprocess_pipeline(article, stop_words):
#     # convert all chars to lowercase
#     article = article.lower()
#     # get rid of non alphabets except while spaces
#     article_filtered = filter(filter_fn, article)
#     words = filter(lambda word: removeNoise(word, stop_words), article_filtered.split(" "))
#     return words

#Enters frequencies of various words in a matrix getting rid of common words

def getWordCounts(words):
    #print type(words)
    word_counts = Counter(words)
    return word_counts


def get_articles(args):
    #reads all text files in a directory
    #stop_words = set(map(lambda line: line.strip(),f.readlines()))
    currDir = os.getcwd()
    # print args.dir
    os.chdir(args.dir)
    # all the words in all the articles, as a set
    total_words = set()
    # all the articles as a list of lists
    articles = []
    fileNames = []
    counter = 0
    for file in glob.glob("*.txt"):
        print "\r>> Done with %d articles" %(counter+1),
        sys.stdout.flush()

        counter += 1
        #Opens a file and reads it
        f = open(file, "r")
        article = f.readlines() #a list containing all the sentences. Each sentence ends with a "\n"
        article = map(lambda line: line.strip(), article)
        fileNames.append(article[0])
        #print(article[0])
        article = " ".join(article)
        #words = preprocess_pipeline(article, stop_words)
        words = pattern.split(article.lower())
        # accumulate
        total_words |= set(words)
        # total_words = sorted(set(words+total_words))
        articles.append(words)
    print "Glob ended"
    total_words = sorted(total_words)
    #fileNametoId = {id : name for (id, name) in enumerate(fileNames)}
    os.chdir(currDir)
    return articles, total_words, fileNames#, fileNametoId

'''
returns the tf-idf weighted document matrix
'''
def convertArticlesToVectors(articles, total_words, vect_dim):
    print "convertArticlesToVectors running"
    #wordsToIds = {word: i for i, word in enumerate(total_words)}
    #IdsToWords = {i: word for i, word in enumerate(total_words)}

    total_words_counts = []
    for article in articles:
        word_counts= getWordCounts(article)
        total_words_counts.append(word_counts)

    vocabSize = len(total_words)
    #print total_words
    print vocabSize
    numArticles = len(articles)
    print len(total_words_counts)
    print numArticles
    #docMatrix = np.zeros((numArticles, vocabSize), dtype="float32") 
    #docMatrix = sp.csc_matrix((numArticles, vocabSize))
    docMatrixList = dict((el,sp.lil_matrix((1,numArticles))) for el in total_words) 
    for i in xrange(numArticles):
        print "\r>> Done with %d/%d articles" %(i+1, numArticles),
        sys.stdout.flush()
        #docSize = len(total_words_counts[i])
        for j in total_words_counts[i]:
            #index = total_words.index(j)
            #currWord = IdsToWords[j]
            #currWordId = WordsToIds[((total_words[i]).keys()).[j]]
            #if (currWord in total_words_counts[i]):
            #docMatrix[i,index] += (total_words_counts[i])[j]
            docMatrixList[j][0,i] = (total_words_counts[i])[j]
    docMatrix = (sp.vstack(docMatrixList.values())).T 
    total_words = docMatrixList.keys()
    # tf = (term Frequency of article)/(total words in article)
    #tf = docMatrix/np.sum(docMatrix, axis= 1, keepdims=True)

    ''' 
        This computes the number of articles in which a particular word 
        has a non zero frequency. 
    '''
    #docFrequencies = np.sum(docMatrix > 0, axis=0, keepdims=True, dtype='float32')
    #idf = np.log(1.0 + numArticles/docFrequencies)
    #return svds(sparse.csc_matrix(tf*idf), vect_dim, which='LM')
    #return sparse.csc_matrix(tf*idf), wordsToIds
    return docMatrix, total_words

def similarDoc(docVector, docMatrix, args, numDoc):
    #minDist = cosine(docVector, docMatrix[0,])
    scores = []
    #minId = 0
    for i in xrange(numDoc):
        currDist = cosine(docVector, docMatrix[i,])
        scores.append((currDist, i))

    scores.sort()
    # indices = []
    # for i in range(opts.numSimilarDocs):
    #     presIndex = np.argmax(scores)
    #     indices.append(presIndex)
    #     scores[presIndex] = 0
    return scores[1:args.k+1]

def getdata(fileName):
    f = open(fileName)
    inputDocs = f.readlines()
    return map(lambda line: line.strip(), inputDocs)



#return articles, total_words, set(articleNames), fileNames
# opts = get_params()
articles, total_words, fileNames = get_articles(args)#, fileNametoId = get_articles()
#f = open("vocab.txt", "w")
#for word in total_words:
#    f.write(word + "\n")
#f.close()

#if (opts.train):
docMatrix, total_words = convertArticlesToVectors(articles, total_words, args.z)
#    joblib.dump(docMatrix,"docMatrix.joblib")
#else:
#    docMatrix = joblib.load("docMatrix.joblib")

# print opts.dim
# print docMatrix.shape
u, s, vts= svds(docMatrix, args.z, which='LM') #doc vectors are u*s ???
docVectors = u*s
wordVectors = vts.T*s

# inputDocs = getdata(args.doc_in)
# f = open(doc_in)
print total_words
inputDocs = f_doc_in.readlines()
inputDocs = map(lambda line: line.strip(), inputDocs)
# print(inputDocs)
# given a doc return sim docs.
f = open(args.doc_out, "w")
for inputDoc in inputDocs:
    docVector = docVectors[fileNames.index(inputDoc)]
    #print docVector
    closestNeighbors = similarDoc(docVector, docVectors, args,len(fileNames))
    f.write(inputDoc + ";\t")
    for dist, neighbor in closestNeighbors:
        f.write(fileNames[neighbor] + ";\t")
    f.write("\n")
print "part 1 done!!"

f.close()
# # implement this
inputTerms = getdata(args.term_in) 
# print inputTerms
f = open(args.term_out,"w")
for inputTerm in inputTerms:
    vt = wordVectors[total_words.index(inputTerm),]
    wordclosestNeighbors = similarDoc(vt, wordVectors, args,len(total_words))
    f.write(inputTerm + ";\t")
    for dist, neighbor in wordclosestNeighbors:
        f.write(total_words[neighbor] + ";\t")
    f.write("\n")

f.close()
print "part 2 done!!"

inputQuerys = getdata(args.query_in)    
f = open(args.query_out,"w")
for inputQuery in inputQuerys:
    wordsInQuery = pattern.split(inputQuery)
    queryVec = np.zeros((1,len(total_words)))
    for w in wordsInQuery:
        i = total_words.index(w)
        queryVec[0,i] += 1
    qdoc = similarDoc(np.dot(queryVec,vts.T), docVectors, args,len(fileNames))
    # qdoc = np.dot(np.dot(queryVec,vts.T),docVector.T)
    for dist, neighbor in qdoc:
        f.write(fileNames[neighbor] + ";\t")
    f.write("\n")

f.close()

