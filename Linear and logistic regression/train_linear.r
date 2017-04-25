#!/usr/bin/env Rscript
# your code must have this first line.

args <- commandArgs(trailingOnly = TRUE)
trainfile <- args[1]
modelfile <- args[2]

B = read.csv(trainfile,header = FALSE)
    
A <- t(matrix(unlist(B), ncol = dim(B)[1], byrow = TRUE))

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

meancentre <- function(A){ 
	mean <- colSums(A)/nrow(A)
	for(i in 1:(ncol(A)-1)){
	A[,i] <- A[,i] - mean[i]
	}
	return(A)
}



normalize <- function(A){

	
	M <- t(A)%*%A
	for (i in 1:(ncol(A)-1)){
	if(sqrt((M[i,i])/nrow(A)) != 0){
	A[,i] <- A[,i]/sqrt((M[i,i])/nrow(A))
	}
	}
	return(A)
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



# Train code for linear regression part goes here


# Y = AX + b
# Y = AX
# X = inv(A'A)A'Y

mpsi <- function(X){
	X1 <- solve(t(X)%*%X)	
	return(X1)
	}

GiveY <- function(A){
	Y <- A[,ncol(A)]
	return(Y)
	}

GiveX <- function(A){
	X <- A[, 1:(ncol(A)-1)]
	return(X)
	}

SolveLinReg <- function(A){
	
	if (det(t(GiveX(A))%*%(GiveX(A))) != 0){
	Xsol <- mpsi(GiveX(A))%*%t(GiveX(A))%*%GiveY(A)
	return(Xsol)
	}
	else{return(0)}
	}

highesteigenvector <- function(z){
	
	v <- matrix(c(1),ncol(z),1)
	
# zv = lambda v
	k <- (z)%*%(v)
	vdash <- k/matrix(sqrt(t(k)%*%k),dim(k),1)
	v_dash_old = matrix(c(0),length(vdash),1)
	while( ((sqrt(t(v_dash_old-vdash)%*%(v_dash_old-vdash))) > 1e-3) && ((sqrt(t(v_dash_old+vdash)%*%(v_dash_old+vdash))) > 1e-3)){
		v_dash_old = vdash
		k <- (z)%*%(vdash)
		vdash <- k/matrix(sqrt(t(k)%*%k),dim(k),1)
	}
	return(vdash)
}



Findfirstkeigenvectors <- function(A,k){
	z <- t(GiveX(A))%*%(GiveX(A))
	collection <- matrix(c(1),ncol(z),k)
	count <- 1
	currenteigenvectoron <- z 
	while(count < k+1){
		v <- highesteigenvector(currenteigenvectoron)
		collection[,count] <- v
		for(i in 1:ncol(z)){
			z[,i] <- z[,i] - ((t(z[,i])%*%v)[1])*v
		}
	currenteigenvectoron <- z
	count<- count+1		
	}		
	return(collection)
}

mypca <- function(A){
	
	L <- Findfirstkeigenvectors(A,10)
	L1 <- GiveX(A)%*%L
	unitmat <- matrix(1,nrow(A),1)
	A <- cbind(L1,unitmat,A[,ncol(A)])
	return(A)
}

convertspacebacktonormal <- function(X,A){
	L <- Findfirstkeigenvectors((normalize(meancentre(featurecreation(A)))),10)
	Am <- featurecreation(A)
	mean <- colSums(Am)/nrow(Am)
	Am <- meancentre(Am)
	normal <- matrix(c(1), ncol(Am)-1,1)
	M <- t(Am)%*%Am
	for (i in 1:(ncol(Am)-1)){
	if(sqrt(M[i,i]) != 0){
	normal[i,1] <- sqrt(M[i,i]/nrow(A))
	}
	}
	
	X1 <-  L%*%X[1:(nrow(X)-1),]
	constant <- X[nrow(X),] - sum(mean[1:(length(mean)-1)]*X1/normal)
	final <- rbind(X1/normal,constant)
	return(final)
}
#fa <- featurecreation(A)
#info <- trainNormalize(fa)
#trainData <- fitNormalize(fa, info)
write.table(convertspacebacktonormal(SolveLinReg(mypca(normalize(meancentre(featurecreation(A))))),A),file = modelfile, row.names=FALSE, col.names=FALSE)
rm(A)

