#Code for Part1
#   Standard Naive bayes code

import math
import argparse
import sys


# testfile, modelfile and labelfile



# parser = argparse.ArgumentParser()

# parser.add_argument('--labelfile', type=str)
# parser.add_argument('--modelfile', type=str)
# parser.add_argument('--testfile', type=str)
# args = parser.parse_args()
trainfile = sys.argv[2]
f_trainfile = open(trainfile)
labelfile = sys.argv[3]
f_labelfile = open(labelfile)
testfile = sys.argv[1]
f_testfile = open(testfile)



fdt = {} 
titles = {}
total_classes = 0

def update_fdt(word, title):   
    if word not in fdt:
        fdt[word] = {}
    if title not in (fdt[word]):
        fdt[word][title] = 0
    fdt[word][title] = fdt[word][title] + 1

def get_articles():
    f = open(trainfile, "r")

    article = f.readlines() #a list containing all the sentences. Each sentence ends with a "\n"
    article = map(lambda line: line.strip(), article)
   
    for lines in article: 
        words = lines.lower().rsplit(" ")
        title = words[0]

        words.pop(0)
        words = set(words)

        for word in words:
            update_fdt(word, title)

        if title not in titles:    titles[title] = 0
        titles[title] = titles[title] + 1
    total_classes = len(titles)

get_articles()

# print titles


global_prob = []

def predict(article):
    words = article.lower().rsplit(" ")
    #words = set(words)
    no_of_words = len(words)
    prob = {}

    for title in titles:
        y = titles[title]
        c = len(titles)
        prob[title] = math.log (y)- no_of_words * math.log(1.0 * y+c)

    for word in words:
        if word in fdt:
            for title in fdt[word]:
                prob[title] = prob[title] + math.log(fdt[word][title] * 100000 + 10000)

    return max(prob, key=prob.get)


def go_test():
    f = open(testfile, "r")
    g = open(labelfile, "w")

    articles = f.readlines() #a list containing all the sentences. Each sentence ends with a "\n"
    articles = map(lambda line: line.strip(), articles)

    for article in articles: 
        ans = predict(article)
        g.write("%s\n" % ans)

go_test()