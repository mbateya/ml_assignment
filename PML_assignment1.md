# Practical Machine Learning A1
Mohammad Ateya  
3/7/2017  
#Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

#Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

#Project Goals

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 



#Model Build:
- Necessary libraries were loaded first 

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(tidyverse)
```

```
## Loading tidyverse: tibble
## Loading tidyverse: tidyr
## Loading tidyverse: readr
## Loading tidyverse: purrr
## Loading tidyverse: dplyr
```

```
## Conflicts with tidy packages ----------------------------------------------
```

```
## filter(): dplyr, stats
## lag():    dplyr, stats
## lift():   purrr, caret
```

- Training and testing data were downloaded and read into data frames


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile = "pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile = "pml-testing.csv")
pmlTraining <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

- Training dataset was sliced into a training subset (60% of observations) and validation subset (40% of observations) data frames. The validation subset is to be used for cross validation 


```r
set.seed(3334)
inTrain <- createDataPartition(y = pmlTraining$classe, p=0.6, list = FALSE)
training <- pmlTraining[inTrain,]
validation <- pmlTraining[-inTrain,]
```

- Variables with limited variance ,i.e. near zero variance were removed from the predictors

```r
nsv <- nearZeroVar(training,saveMetrics = TRUE)
training2 <- training[,row.names(nsv[nsv$nzv == "FALSE",])]
```

- Removed variables with more than 90% missing values

```r
training3 <- training2[,colSums(!is.na(training2)) > (nrow(training2)*0.9)]
## print names of variables used as predictors and the outcome variable "classe"
names(training3)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"          
##  [7] "roll_belt"            "pitch_belt"           "yaw_belt"            
## [10] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [13] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [16] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [19] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [22] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [25] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [28] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [31] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [34] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [37] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [40] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [43] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [46] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [49] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [52] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [55] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [58] "magnet_forearm_z"     "classe"
```

- Remove un-used variables from the validation subset

```r
validation2 <- validation[,colnames(training3)]
```

- Used diffrent algorithms to build prediction models: trees, random forests, and tree bagging.
- Each model was used to predict the classe variable in the validation subset
- Confusion matrix function from the caret package was used to calculate accuracy of the model
- Random forest has the highest accuracy (lowest out of sample error rate).


```r
### All the code below, except that for the random forest model, was commented to faciliate generating the R Markdown file 

#Fitting a tree model
# modelfit1 <- rpart(classe ~ .,method = "class", data = training3[,-(1:6)])
# predictions1 <- predict(modelfit1,validation2[,-(1:6)],type = "class")
# confusionMatrix(predictions1,validation2$classe)

#Fitting a bagging model
# modelfit2 <- train(classe ~ .,method = "treebag", data = training3[,-(1:6)])
# predictions2 <- predict(modelfit2,validation2[,-(1:6)])
# confusionMatrix(predictions2,validation2$classe)

# Fitting Random Forest model
modelfit3 <- randomForest::randomForest(classe ~ .,method = "treebag", 
                                        data = training3[,-(1:6)],prox=TRUE)
predictions3 <- predict(modelfit3,validation2[,-(1:6)])
confusionMatrix(predictions3,validation2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    8    0    0    0
##          B    0 1507   13    0    0
##          C    0    3 1352   21    0
##          D    0    0    3 1264    5
##          E    0    0    0    1 1437
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9931         
##                  95% CI : (0.991, 0.9948)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9913         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9928   0.9883   0.9829   0.9965
## Specificity            0.9986   0.9979   0.9963   0.9988   0.9998
## Pos Pred Value         0.9964   0.9914   0.9826   0.9937   0.9993
## Neg Pred Value         1.0000   0.9983   0.9975   0.9967   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1921   0.1723   0.1611   0.1832
## Detection Prevalence   0.2855   0.1937   0.1754   0.1621   0.1833
## Balanced Accuracy      0.9993   0.9953   0.9923   0.9908   0.9982
```
Out of sample error rate for random forest model is expected to be 0.69%
