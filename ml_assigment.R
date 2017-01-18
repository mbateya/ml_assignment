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
training <- training[,row.names(nsv[nsv$nzv == "FALSE",])]
