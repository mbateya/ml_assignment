#Load Packages and download/read data
library(caret)
library(tidyverse)
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile = "pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile = "pml-testing.csv")
pmlTraining <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

#Slice data into a training and validation data frames for cross-validation
inTrain <- createDataPartition(y = pmlTraining$classe, p=0.6, list = FALSE)
training <- pmlTraining[inTrain,]
validation <- pmlTraining[-inTrain,]

#Remove near zero variables
nsv <- nearZeroVar(training,saveMetrics = TRUE)
training2 <- training[,row.names(nsv[nsv$nzv == "FALSE",])]
#Remove variables with more than 90$ missing values
training3 <- training2[,colSums(!is.na(training2)) > (nrow(training2)*0.9)]

#Remove near zero variables and variables with missing data from validation set
validation2 <- validation[,colnames(training3)]

#Fitting a tree model
modelfit1 <- train(classe ~ .,method = "rpart", data = training3[,-(1:6)])
predictions1 <- predict(modelfit1,validation2[,-(1:6)])
confusionMatrix(predictions1,validation2$classe)

#Fitting a bagging model
modelfit2 <- train(classe ~ .,method = "treebag", data = training3[,-(1:6)])
predictions2 <- predict(modelfit2,validation2[,-(1:6)])
confusionMatrix(predictions2,validation2$classe)

# Fitting Random Forest model
modelfit3 <- randomForest::randomForest(classe ~ .,method = "treebag", 
                                        data = training3[,-(1:6)],prox=TRUE)
predictions3 <- predict(modelfit3,validation2[,-(1:6)])
confusionMatrix(predictions3,validation2$classe)

# Fitting gbm boosting model
modelfit4 <- train(classe ~ .,method = "gbm", data = training3[,-(1:6)],verbose=F)
predictions4 <- predict(modelfit4,validation2[,-(1:6)])
confusionMatrix(predictions4,validation2$classe)


# Fitting linear discrimnant analysis model
modelfit5 <- train(classe ~ .,method = "lda", data = training3[,-(1:6)])
predictions5 <- predict(modelfit5,validation2[,-(1:6)])
confusionMatrix(predictions5,validation2$classe)
