############################################
#####OBSG-SVM with online pass for training and batch testing
############################################

library("pracma", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")

onlineBilevelSvm <- function(xtrain,ytrain,xtest,ytest,w0,c){
  streamData <- as.matrix(xtrain)
  y <- as.vector(ytrain)
  
  Ncols <- ncol(streamData)
  Nrows <- nrow(streamData)
  i <- 1
  #w <- w0
  w <- w0/Norm(w0)
  b <- 0
  #c <- 3
  cmin <- 0
  cmax <- 10^5
  yii <- y[1]
  xii <- streamData[1,]
  prediction <- NULL
  
  ###training part
  for (i in 1:Nrows){

    if( y[i]*(streamData[i,] %*% w+b) <1){
      dw <- w - c*y[i]*streamData[i,]
      stepw <-  1/(i)
      w = w - stepw*dw
      b <- b - c*stepw*(-y[i])  # the result is better without updating b
      dc <- -t(y[i]*streamData[i,])%*%(yii*xii)
      stepc <- 1/(i)   
      c <- c - stepc*dc
      if(c < cmin)
        c <- 0.00001
      if (c>cmax)
        c <- cmax
      
      yii <- y[i]
      xii <- streamData[i,]
    }
  }
  
  ## testing result
  y_pred <- sign(as.matrix(xtest)%*% w+as.matrix(b)[1,1]*matrix(1,nrow = nrow(as.matrix(xtest)),ncol = 1))
  #a <- length(which(prediction==ytest[1:nrow(prediction)]))/nrow(prediction)
  a <- length(which(y_pred==ytest[1:nrow(y_pred)]))/nrow(y_pred)
  return(a)
}

########################################################
##### SVM with online pass for training and batch testing
##########################################################
onlineSvm_vt <- function(xtrain,ytrain,xtest,ytest,w0,c){
  streamData <- as.matrix(xtrain)
  y <- as.vector(ytrain)
  
  Ncols <- ncol(streamData)
  Nrows <- nrow(streamData)
  i <- 1
  w <- w0
  w <- w/Norm(w)
  b <- 0
  prediction <- NULL
  
  # training part
  for(i in 1:Nrows){ 
    
    stepw <-   1/(i)
    
    if( y[i]*(streamData[i,] %*% w+b) < 1){
      dw <- w - c*y[i]*streamData[i,]
      w = w - stepw*dw
      b <- b - stepw*(-y[i])  # the result is better without updating b
    }else{
      dw <- w 
      w = w - stepw*dw
      b <- b 
    }
  }
  
  y_pred <- sign(as.matrix(xtest)%*% w+b*matrix(1,nrow = nrow(as.matrix(xtest)),ncol = 1))
  a <- length(which(y_pred==ytest))/nrow(y_pred)
  #a <- length(which(prediction==ytest[1:nrow(prediction)]))/nrow(prediction)
  return(a)
}

###############################################
## 20 runs 
#shuffle before each run
#average prediction accuracy over 20 runs for each c 
###############################################
w0 <- as.vector(runif(ncol(x), min = -1,max = 1))
prediction_BOSVM <- matrix(0,nrow = 10,ncol = 20)
prediction_OSVM <- matrix(0,nrow = 10,ncol = 20)
n <- nrow(Datay)
for (i in 1:10){
  c <- 5^(5-i)
  for (j in 1:20){
    int <- sample.int(n,n)
    Datax <- Datay[int,]
    xtrain <- Datax[1:floor(0.75*n),-1]
    ytrain <- Datax[1:floor(0.75*n),1]
    xtest <- Datax[(floor(0.75*n)+1):n,-1]
    ytest <- Datax[(floor(0.75*n)+1):n,1]
    prediction_BOSVM[i,j] <- onlineBilevelSvm(xtrain,ytrain,xtest,ytest,w0,c)
    prediction_OSVM[i,j] <- onlineSvm(xtrain,ytrain,xtest,ytest,w0,c)
  }
}

predB <- rowMeans(prediction_BOSVM)
predO <- rowMeans(prediction_OSVM,2)


indB <- which(predB==max(predB))
max(predB)
5^(5-indB[1])
mean(prediction_BOSVM[indB[1],])
var(prediction_BOSVM[indB[1],])


indO <- which(predO==max(predO))
max(predO)
5^(5-indO[1])
mean(prediction_OSVM[indO[1],])
var(prediction_OSVM[indO[1],])


plot(predB,type="l",ylim = c(0,1),xlab = "Different C values 5^(5-x)",ylab = "Prediction accuracy",main = "Covtype")
lines(predO,col="red")
abline(h=max(predB),lty=2,col="blue")
abline(h=max(predO),lty=2,col="red")
legend(1, 0.3, legend=c("OBSG-SVM", "OSVM"), col=c("blue","red"), lty=1 ,cex=0.8)
