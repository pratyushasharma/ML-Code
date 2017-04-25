#!/usr/bin/env Rscript
# your code must have this first line.

# Test code for linear regression part goes here
args <- commandArgs(trailingOnly = TRUE)
testfile <- args[1]
modelfile <- args[2]
labelfile <- args[3]

B = read.csv(testfile,header = FALSE)

B <- t(matrix(unlist(B), ncol = dim(B)[1], byrow = TRUE))

D = as.matrix(read.csv(modelfile, header = FALSE))

#D <- t(matrix(unlist(D), ncol = dim(D)[1], byrow = TRUE))

featurecreation <- function(A){
	
	features <- matrix(c(0),nrow(A),10)
	
	counter <- 1
	for (i in 1:4){
		
		for(t in i:4){
		
		features[,counter] =  	A[,i]*A[,t]
		counter <- counter+1	
		}
	}
	features <- cbind(features,A)
	return(features)
}



forlabelfile <- function(B,D){
	B <- featurecreation(B)
	unity <- matrix(c(1), nrow(B),1)
	B <- cbind(B,unity)
	yd <- B%*%D
	return(yd)	
	}

write.table(forlabelfile(B,D), file=labelfile, row.names=FALSE, col.names=FALSE)

rm(B)
rm(D)
