#!/usr/bin/env Rscript
# your code must have this first line.

# Train code for logistic regression part goes here

args <- commandArgs(trailingOnly = TRUE)
trainfile <- args[1]
modelfile <- args[2]

A = read.csv(trainfile,header= FALSE)
#B = read.csv("files1/public_test2.csv", header= FALSE)
A <- t(matrix(unlist(A), ncol = dim(A)[1], byrow = TRUE))
#B <- t(matrix(unlist(B), ncol = dim(B)[1], byrow = TRUE))


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


prob <- function(A,m,p) {	
	fix <- length(p)
	#probabilitymartix[i,j] = P(y_j=i | x_j)
	probabilitymatrix <- matrix(0,fix,nrow(A)) 
		
	pp <-	m%*%t(A[,1:(ncol(A)-1)])
	probabilitymatrix <- exp(t(t(pp) - apply(pp,2,max)))
	probabilitymatrix <- t(probabilitymatrix) /colSums(probabilitymatrix)
	probabilitymatrix <- t(probabilitymatrix)
	subtractionmatrix <- matrix(0,fix,nrow(A))

	for( i in 1:nrow(A)){
		subtractionmatrix[A[i,ncol(A)],i] = 1
	}
	probabilitymatrix = subtractionmatrix - probabilitymatrix 
	
	return(probabilitymatrix)
}

GraDes <- function(A,w,p, lr){
	probmat <- prob(A,w,p)
	for(i in 1:length(p)){ # ith class
		descentterm <- colSums(probmat[i,]*A[,(1:ncol(A)-1)])		
		w[i,] <- w[i,]+ lr*descentterm
	}

	return(w)
}


meancentre <- function(A){ 
	mean <- colMeans(A)
	for(i in 1:(ncol(A)-1)){
	A[,i] <- A[,i] - mean[i]
	}
	return(A)

}


normalize <- function(A){
	# A is a matrix of size N x (d+1) where d is the number of features. The last column contains the labels
	featureVector <- A[, 1: ncol(A)-1]
	labels <- A[, ncol(A)]
	# centeredData is a matrix of size N x d
	centeredData <- t(t(featureVector) - colMeans(featureVector))
	# stddev is a vector of size d
	stddev <- sqrt(colSums(centeredData*centeredData))
	normalizedFeatures <- t(t(centeredData) / stddev)
	return(cbind(normalizedFeatures, labels))
}

trainNormalize <- function(A) {
	featureVector <- A[, 1: ncol(A)-1]
	meanData <- colMeans(featureVector)
	centeredData <- t(t(featureVector) - colMeans(featureVector))
	stddev <- sqrt(colSums(centeredData*centeredData))
	return(matrix(c(meanData, stddev), ncol(featureVector), 2))
}

fitNormalize <- function(A, info) {
	featureVector <- A[, 1: ncol(A)-1]
	labels <- A[, ncol(A)]
	centeredData <- t(t(featureVector) - info[,1])
	normalizedFeatures <- t(t(centeredData) / info[,2])
	return(cbind(normalizedFeatures, labels))
}


#numClass = length(p)
#numFeatures= ncol(x1)-1
#w <- matrix(rnorm(N*M,mean=0,sd=1), numClass, numFeatures) 


trainClassifier <- function(trainData, p, itr, lr) {
	numClass <- length(p)
	goldLabels <- trainData[, ncol(trainData)]
	dim <- ncol(trainData)-1
	w <- matrix(rnorm(numClass*dim,mean=0,sd=1),numClass,dim) 
	bestW <- w
	bestAcc <- 0.0
	for (i in 1:itr) {
		lr <- lr/(1 + 1e-6*i)
		w <- GraDes(trainData, w, p, lr)
		currAcc <- accuracy(goldLabels, w, trainData)
		if (currAcc > bestAcc) {
			bestW <- w
			bestAcc <- currAcc
		}
	}	

	return(bestW)
}


accuracy <- function(goldLabels, w, testData ){
	pp <-	w%*%t(testData[,1:(ncol(testData)-1)])
	preds <- apply(pp,2,which.max)
	c <- matrix(preds==goldLabels)
	return(colSums(c)/nrow(c))
}



p <- matrix(1:7,1,7)
#fa <- featurecreation(A)
#rm(A)
#info <- trainNormalize(fa)
#trainData <- fitNormalize(fa, info)
#rm(fa)
info <- trainNormalize(A)
trainData <- fitNormalize(A, info)
rm(A)

#write.table(convertspacebacktonormal(GradConverge(x1,w,p),A), file="modelfile_train_Lo.csv", row.names=FALSE, col.names=FALSE)
write.table(rbind(trainClassifier(trainData,p,20,0.4),t(info)), file=modelfile, row.names=FALSE,sep=',')
