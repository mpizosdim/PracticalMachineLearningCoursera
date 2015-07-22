---
title: "Prediction Assignment:Predicting activity quality"
author: "Bizopoulos Dimitrios"
date: "Tuesday, July 21, 2015"
output: pdf_document
---

## Background
Using devices suck as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that epople regularly do is quantify how much of a particular activity they do, but rarely quantify how weill they do it. In this project, the goal is to used the data provided from accelerometers on the belt, forearm, arm , and dumbell of 6 participants. The participants were asked to perform barbell lifts correctly and incorrectly in five different ways.
##Reproducability
To reproduce the same results use the seed that is used later on the text. Furthermore the following packages has to be installed to run the code:

* caret package 
* rattle package
* randomForest package

##Loading the data 
In the beggining of the project we load the provided data,before that the data is downloaded from the links in the description of the project. The data  consist of the training data and the testing data. 

```r
setwd('C:/Users/Dimitrios/Documents/Dimitris_general/Coursera/PracticalMachineLearning')
trainData <- read.csv('pml-training.csv',na.strings=c('NA',''))
testData <- read.csv('pml-testing.csv',na.strings=c('NA',''))
```
variable `testData` is used later for the Submission of the project. To be mentioned also that the data for this project come from the following source [Activity Data](http://groupware.les.inf.puc-rio.br/har).


```r
DimTrain <- dim(trainData)
```
Dimension of the training data is 19622, 160. 

## Partioning the data
In this section the `trainData` is partioned to training and testing data. 60% of the data will be training and 40% will be testing data. Furthermore we set the seed and load the `caret` package.


```r
set.seed(86)
library(caret)
inTrain <- createDataPartition(y=trainData$classe,p=0.6,list=FALSE)
training <- trainData[inTrain,]
testing <- trainData[-inTrain,]
```

## Removing zero covariates

We are going to use the near zero variable function from carrot package to identify those variables that have very little variability and will likely not be good predictors for our model and remove them from the data set:


```r
NonUsefull <- nearZeroVar(training,saveMetrics=TRUE)
RemovedVariables <- rownames(NonUsefull)[NonUsefull$nzv]
training <- training[,!names(training) %in% RemovedVariables]
```


```r
DimTrain <- dim(training)
```
Dimension of the training data now is 11776, 121. It can be seen by removing zero covariates variables we excluded 38 variables.

## Removing variables with NA
Now we will check the variables that have a lot of NA values. It is decieded by the author to exclude variables with higher that 90% rate  with NA values.


```r
na_count <- apply(training,2,function(x) length(which(is.na(x)))/11776)
RemovedVariables <- names(na_count)[na_count>0.9]
training <- training[,!names(training) %in% RemovedVariables]
```


```r
DimTrain <- dim(training)
```
Dimension of the training data now is 11776, 59. It can be seen by removing high NA rate variables  we excluded 63 variables.

Finally, observing the variable we find some other variables that common sence says that can not be usefull for the prediction. It is deceided to remove the time variables `raw_timestamp_part_1`,`raw_timestamp_part_2`,`cvtd_timestamp`  as well as the `X` variable which seems to be the measurment ID. 

```r
training <- training[,!names(training) %in% c('X','raw_timestamp_part_1','raw_timestamp_part_2','cvtd_timestamp')]
```

The data set now is ready to get trained with total of 55 variables:

```r
dim(training)[2]
```

```
## [1] 55
```
## Machine Learning Model: Random Forest

A model to predict the classe variable using a random forest is created in this section. Random forests are an esemble learning method that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes. It is clear that having a matrix of 55 variables and more than 10000 observation for each variable the random forest will be computationlly slow but expected to have good results:

```r
#
library(randomForest)
RFmodel <- randomForest(classe~.,data=training,importance=TRUE,ntree=500)
saveRDS(RFmodel,'RFmodel.RDS')
readRDS('RFmodel.RDS')
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, importance = TRUE,      ntree = 500) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.34%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    0    0    0    1 0.0002986858
## B    4 2274    1    0    0 0.0021939447
## C    0   12 2041    1    0 0.0063291139
## D    0    0   15 1914    1 0.0082901554
## E    0    0    0    5 2160 0.0023094688
```

```r
print(RFmodel,digits=3)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, importance = TRUE,      ntree = 500) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.34%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    0    0    0    1 0.0002986858
## B    4 2274    1    0    0 0.0021939447
## C    0   12 2041    1    0 0.0063291139
## D    0    0   15 1914    1 0.0082901554
## E    0    0    0    5 2160 0.0023094688
```
It can be seen in the code that we use also ther `saveRDS` and `readRDS`. We use the save function to save the model as it is time consuming to generate every time and later we load it using the read function (in order to run and save the training u need to uncomment the first two lines of the above code block)
The accuracy of the model is tested in the following code:

```r
Predictions <- predict(RFmodel,testing)
confusionMatrix(Predictions,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    7    0    0    0
##          B    0 1509    6    0    0
##          C    0    2 1362   12    0
##          D    0    0    0 1274    2
##          E    0    0    0    0 1440
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9947, 0.9975)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9953          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9941   0.9956   0.9907   0.9986
## Specificity            0.9988   0.9991   0.9978   0.9997   1.0000
## Pos Pred Value         0.9969   0.9960   0.9898   0.9984   1.0000
## Neg Pred Value         1.0000   0.9986   0.9991   0.9982   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1923   0.1736   0.1624   0.1835
## Detection Prevalence   0.2854   0.1931   0.1754   0.1626   0.1835
## Balanced Accuracy      0.9994   0.9966   0.9967   0.9952   0.9993
```

The model is 99.8% accurate which is considered really good. It is expected that the accuracy will be a little bit less with new data set.



## Submission

The following code was used for the submission:

```r
testData <- testData[,names(training)]
```

```
## Error in `[.data.frame`(testData, , names(training)): undefined columns selected
```

```r
prediction <- predict(RFmodel,testData)
for (ii in seq(20)){
    fileName <- paste('problem',ii,'.txt',sep='_')
    write.table(prediction[ii],file=fileName,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
```

The model created succeded to predict all the 20 observations!

## Conclusion

We have built a predictor model using the random forest to predict exercise based on movement data. The model accuracy is satisfactory. Although the accuracy is really high, it is expected that it will be lower in real life situations.
