import sys

trainfile = sys.argv[1]
modelfile = sys.argv[2]

f = open (trainfile, "r")
g = open(modelfile, "w")

f = f.read()
g.write(f)