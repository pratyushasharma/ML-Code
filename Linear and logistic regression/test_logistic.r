#!/usr/bin/env Rscript
# your code must have this first line.

# Test code for logistic regression part goes here
message("yay")
args <- commandArgs(trailingOnly = TRUE)
testfile <- args[1]
modelfile <- args[2]
labelfile <- args[3]

B = read.csv(testfile, header = FALSE)

B <- t(matrix(unlist(B), ncol = dim(B)[1], byrow = TRUE))

D = as.matrix(read.csv(modelfile, header = TRUE,sep=','))

#D <- t(matrix(unlist(D), ncol = dim(D)[1], byrow = TRUE))

info <- t(D[(nrow(D)-1):nrow(D),])
D<- D[1:(nrow(D)-2),]


featurecreation <- function(A){
	
	features <- matrix(c(0),nrow(A),55)
	
	counter <- 1
	for (i in 1:10){
		
		for(t in i:10){
		
		features[,counter] =  	A[,i]*A[,t]
		counter <- counter+1	
		}
	}
	features <- cbind(features,A)
	return(features)
}
trainNormalize <- function(A) {
	featureVector <- A[, 1: ncol(A)-1]
	meanData <- colMeans(featureVector)
	centeredData <- t(t(featureVector) - colMeans(featureVector))
	stddev <- sqrt(colSums(centeredData*centeredData))
	return(matrix(c(meanData, stddev), ncol(featureVector), 2))
}



fitNormalize <- function(A, info) {
	featureVector <- A[, 1: ncol(A)]
	centeredData <- t(t(featureVector) - info[,1])
	normalizedFeatures <- t(t(centeredData) / info[,2])
	return(normalizedFeatures)
}




labelfile <- function(B,D){
			
	pp <-	D%*%t(B[,1:ncol(B)])

	solution <- apply(pp,2,which.max)
	
	return(solution)	
	}


p <- matrix(1:7,1,7)
#fb <- featurecreation(B)
#trainData <- fitNormalize(fb, info)
trainData <- fitNormalize(B, info)

#rm(fb)
rm(B)

write.table(labelfile(trainData,D), file=labelfile, row.names=FALSE, col.names=FALSE)
