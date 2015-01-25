---
title: "Automatic Recognition of Weight Lifting Exercises"
author: "Jonathan J. Hull"
date: "Friday, January 23, 2015"
output:
  html_document:
    pandoc_args: [
      "+RTS", "-K64m",
      "-RTS"
    ]
---
# Automatic Recognition of Weight Lifting Exercises
## Jonathan J. Hull
## Friday, January 23, 2015

### Executive Summary

The objective of this project is to predict the manner in which subjects did barbell lifts.  They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.  This is the "classe" variable in the training set.  

Upon inspection of the training data, I observed that many of the variables had a large number of missing cases.  I eliminated them from consideration and trained a random forest on the rest of the data.  This proved quite effective with an Out Of Bag error rate of 0.6%.  A separate test set of 979 values that was held out of all previous development of the model, yielded a 0% error rate.  All samples were classified correctly.



```r
setwd("C:/Users/hull/Documents/data science/Practical ML/project/");
library(caret);
library(R.utils);

###########################################################################
#
#  pml_write_files -- supplied with project.  Write out answers on test set
#

pml_write_files = function(x) {
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
###############################################################################
#
#  accuracy(values,prediction) -- compare values to prediction and print
#     the percentage that are different from each other.  This is used to
#     calculate the out of sample error rate.
#

accuracy <- function(values,prediction) {
  num_errors <- sum(prediction != values);
  printf("out of sample error rate estimate=%5.2f%%\n", 100.0*num_errors/length(values));
  printf("   estimated on %d samples\n",length(values));
}
```

### Loading the Data

The training data is read from the csv file supplied for the project and the columns that are mostly empty are trimmed out. 5% of the training data are held back as an independent data set to test the validity of the model and verify the error rate estimate supplied by the randoem forest algorithm.


```r
training_orig    <- read.csv("pml-training.csv");
training_trimmed <- training_orig[-c(1,5,6,12:36,50:59,69:83,87:101,103:112,125:139,141:150)];

set.seed(1976);
inValid           <- createDataPartition(y=training_trimmed$classe, p=0.95, list=FALSE);
validity_test_set <- training_trimmed[-inValid,];
training_set      <- training_trimmed[inValid,];
```

### Training the model with Cross Validation

The random forest is trained on the 56 variables left after the empty columns were trimmed out.


```r
#  Train the random forest on the 56 variables left over after trimming

# rf_56vars <- train(classe~.,data=training_set, method="rf", trControl=trainControl(method="cv",number=10))
# save(rf_56vars, file="rf_56vars.RData");

load("rf_56vars.Rdata");

# Print the random forest

print(rf_56vars);
```

```
## Random Forest 
## 
## 18643 samples
##    56 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 16780, 16779, 16779, 16777, 16780, 16778, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9974255  0.9967435  0.0012587192  0.0015922836
##   31    0.9994637  0.9993216  0.0002527639  0.0003196871
##   60    0.9990344  0.9987786  0.0007920128  0.0010019435
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 31.
```

### Estimate Out of Sample Error Rate with Cross Validation

The out of sample error rate is estimated with the OOB or out of bag value in the finalModel.  In our case it is 0.06%.  The following explanation is on: [link](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr)

*In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the run, as follows:

Each tree is constructed using a different bootstrap sample from the original data. About one-third of the cases are left out of the bootstrap sample and not used in the construction of the kth tree.

Put each case left out in the construction of the kth tree down the kth tree to get a classification. In this way, a test set classification is obtained for each case in about one-third of the trees. At the end of the run, take j to be the class that got most of the votes every time case n was oob. The proportion of times that j is not equal to the true class of n averaged over all cases is the oob error estimate. This has proven to be unbiased in many tests.*

We verify that with our own validity_test_set of 979 cases.  When run against the random forest, every sample was classified correctly (a 0% error rate).


```r
# Print the final model

print(rf_56vars$finalModel);
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 31
## 
##         OOB estimate of  error rate: 0.06%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5301    0    0    0    0 0.0000000000
## B    1 3606    1    0    0 0.0005543237
## C    0    4 3247    0    0 0.0012303906
## D    0    0    3 3052    1 0.0013089005
## E    0    0    0    1 3426 0.0002918004
```

```r
res_validity <- predict(rf_56vars, newdata=validity_test_set);

accuracy(res_validity, validity_test_set$classe);
```

```
## out of sample error rate estimate= 0.00%
##    estimated on 979 samples
```

### Prediction on the Test Set

Run the model on the test set and save the results as specified in the project desc.


```r
test_orig    <- read.csv("pml-testing.csv");
test_trimmed <- test_orig[-c(1,5,6,12:36,50:59,69:83,87:101,103:112,125:139,141:150)];
res_test     <- predict(rf_56vars,newdata=test_trimmed);

setwd("results");
results_to_report <- as.character(res_test);
print(results_to_report);
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

```r
pml_write_files(results_to_report);
```

### Conclusions

We predicted the manner in which subjects did barbell lifts with a random forest trained on 18,643 samples.  Cross validation was used during training and an Out of Bag error of 0.06% was obtained.  Application to a separate validity test set held out from all model training provide a 0% error rate.
