# calling libraries
library(caret)
library(readr)
library(doParallel)

#initial settings
set.seed(888)
registerDoParallel(cores=5)

#loading data
CompleteResponses <- read.csv(file="CompleteResponses.csv")

#checking the structure
head(CompleteResponses)
str(CompleteResponses)
is.na(CompleteResponses)


#preprocessing
CompleteResponses$elevel <- as.factor(CompleteResponses$elevel)
CompleteResponses$car <- as.factor(CompleteResponses$car)
CompleteResponses$zipcode <- as.factor(CompleteResponses$zipcode)
CompleteResponses$brand <- as.factor(CompleteResponses$brand)

#data sampling
CompleteResponses <- CompleteResponses[sample(1:nrow(CompleteResponses),nrow(CompleteResponses),replace = FALSE),]

#partitioning
partition <- createDataPartition(CompleteResponses$brand, p = .75, list = FALSE)
training <- CompleteResponses[partition,]
testing <- CompleteResponses[-partition,]

#model tuning and training
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
C5Fit1 <- train(brand~., data = training, method = "C5.0", trControl=fitControl, tuneLength = 1)

#assessing
C5Fit1
C5.0 

7424 samples
   6 predictor
   2 classes: '0', '1' 

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 1 times) 
Summary of sample sizes: 6681, 6682, 6682, 6682, 6681, 6683, ... 
Resampling results across tuning parameters:

  model  winnow  Accuracy   Kappa    
  rules  FALSE   0.8437359  0.6860151
  rules   TRUE   0.8437354  0.6862942
  tree   FALSE   0.8437359  0.6860385
  tree    TRUE   0.8440045  0.6869098

Tuning parameter 'trials' was held constant at a value of 1
Accuracy was used to select the optimal model using the largest value.
The final values used for the model were trials = 1, model = tree and winnow = TRUE.

#increased tuneLenght to 15, repeats to 3
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
C5Fit1_trained <- train(brand~.,data = training, method = "C5.0", trControl = fitControl, tuneLenght = 15)
C5Fit1_trained
C5.0

7424 samples
   6 predictor
   2 classes: '0', '1'

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 3 times)
Summary of sample sizes: 6682, 6682, 6682, 6682, 6681, 6681, ...
Resampling results across tuning parameters:

  model  winnow  trials  Accuracy   Kappa
  rules  FALSE    1      0.8213361  0.6429588
  rules  FALSE   10      0.9213358  0.8319871
  rules  FALSE   20      0.9225033  0.8350970
  rules   TRUE    1      0.8214716  0.6432183
  rules   TRUE   10      0.9218308  0.8327695
  rules   TRUE   20      0.9250176  0.8402714
  tree   FALSE    1      0.8212012  0.6426885
  tree   FALSE   10      0.9217862  0.8334876
  tree   FALSE   20      0.9232228  0.8368925
  tree    TRUE    1      0.8214716  0.6432765
  tree    TRUE   10      0.9220107  0.8338716
  tree    TRUE   20      0.9235818  0.8375421

Accuracy was used to select the optimal model using the largest value.
The final values used for the model were trials = 20, model = rules and winnow = TRUE.

#assessing variables
C5F1Imp <- varImp(C5Fit1)
C5F1Imp
C5.0 variable importance

  only 20 most important variables shown (out of 34)

         Overall
salary    100.00
age        51.79
car15       0.00
car7        0.00
elevel2     0.00
car5        0.00
zipcode3    0.00
car8        0.00
car2        0.00
zipcode5    0.00
zipcode7    0.00
car16       0.00
car4        0.00
elevel4     0.00
zipcode2    0.00
car20       0.00
elevel3     0.00
elevel1     0.00
zipcode8    0.00
car9        0.00

#dropping all attributes except "salary", "age", "brand"
keeps <- c("salary", "age", "brand")
training_2 <- training[keeps]
testing_2 <- testing[keeps]

#training new tuned model without unimportant columns
C5Fit1_2 <- train(brand~., data = training_2, method = "C5.0", trControl=fitControl, tuneLength = 10)
C5Fit1_2
C5.0

7424 samples
   2 predictor
   2 classes: '0', '1'

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 3 times)
Summary of sample sizes: 6683, 6681, 6681, 6682, 6681, 6682, ...
Resampling results across tuning parameters:

  model  winnow  trials  Accuracy   Kappa
  rules  FALSE    1      0.8793221  0.7534489
  rules  FALSE   10      0.9232245  0.8359096
  rules  FALSE   20      0.9221904  0.8344051
  rules  FALSE   30      0.9221456  0.8342980
  rules  FALSE   40      0.9221456  0.8342980
  rules  FALSE   50      0.9221456  0.8342980
  rules  FALSE   60      0.9221456  0.8342980
  rules  FALSE   70      0.9221456  0.8342980
  rules  FALSE   80      0.9221456  0.8342980
  rules  FALSE   90      0.9221456  0.8342980
  rules   TRUE    1      0.8793221  0.7534489
  rules   TRUE   10      0.9232245  0.8359096
  rules   TRUE   20      0.9221904  0.8344051
  rules   TRUE   30      0.9221456  0.8342980
  rules   TRUE   40      0.9221456  0.8342980
  rules   TRUE   50      0.9221456  0.8342980
  rules   TRUE   60      0.9221456  0.8342980
  rules   TRUE   70      0.9221456  0.8342980
  rules   TRUE   80      0.9221456  0.8342980
  rules   TRUE   90      0.9221456  0.8342980
  tree   FALSE    1      0.8773919  0.7472963
  tree   FALSE   10      0.9221461  0.8347589
  tree   FALSE   20      0.9221003  0.8350058
  tree   FALSE   30      0.9225941  0.8361714
  tree   FALSE   40      0.9225941  0.8361714
  tree   FALSE   50      0.9225941  0.8361714
  tree   FALSE   60      0.9225941  0.8361714
  tree   FALSE   70      0.9225941  0.8361714
  tree   FALSE   80      0.9225941  0.8361714
  tree   FALSE   90      0.9225941  0.8361714
  tree    TRUE    1      0.8773919  0.7472963
  tree    TRUE   10      0.9221461  0.8347589
  tree    TRUE   20      0.9221003  0.8350058
  tree    TRUE   30      0.9225941  0.8361714
  tree    TRUE   40      0.9225941  0.8361714
  tree    TRUE   50      0.9225941  0.8361714
  tree    TRUE   60      0.9225941  0.8361714
  tree    TRUE   70      0.9225941  0.8361714
  tree    TRUE   80      0.9225941  0.8361714
  tree    TRUE   90      0.9225941  0.8361714

Accuracy was used to select the optimal model using the largest value.
The final values used for the model were trials = 10, model = rules and winnow = TRUE.

#now we have 3 trained models. let's make predictions for each
prediction1 <- predict(C5Fit1, testing) #10-fold cross validation and an Automatic Tuning Grid
prediction2 <- predict(C5Fit1_trained, testing) #tuned C5.0 model
prediction3 <- predict(C5Fit1_2, testing_2) #tuden C5.0 model with only 2 important columns

#confusion matrixs
confusionMatrix(prediction1, testing$brand)
Confusion Matrix and Statistics

          Reference
Prediction    0    1
         0  835   93
         1  101 1445
				
               Accuracy : 0.9216
                 95% CI : (0.9103, 0.9319)
    No Information Rate : 0.6217
    P-Value [Acc > NIR] : <2e-16

                  Kappa : 0.833

 Mcnemars Test P-Value : 0.6153

            Sensitivity : 0.8921
            Specificity : 0.9395
         Pos Pred Value : 0.8998
         Neg Pred Value : 0.9347
             Prevalence : 0.3783
         Detection Rate : 0.3375
   Detection Prevalence : 0.3751
      Balanced Accuracy : 0.9158

       'Positive' Class : 0

>confusionMatrix(prediction2, testing$brand)
Confusion Matrix and Statistics

          Reference
Prediction    0    1
         0  822   76
         1  114 1462
                                         
               Accuracy : 0.9232         
                 95% CI : (0.912, 0.9334)
    No Information Rate : 0.6217         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.8354         
                                         
 Mcnemars Test P-Value : 0.007269       
                                         
            Sensitivity : 0.8782         
            Specificity : 0.9506         
         Pos Pred Value : 0.9154         
         Neg Pred Value : 0.9277         
             Prevalence : 0.3783         
         Detection Rate : 0.3323         
   Detection Prevalence : 0.3630         
      Balanced Accuracy : 0.9144         
                                         
       'Positive' Class : 0
	 
>confusionMatrix(prediction3, testing_2$brand)
Confusion Matrix and Statistics

          Reference
Prediction    0    1
         0  826   81
         1  110 1457
                                         
               Accuracy : 0.9228         
                 95% CI : (0.9116, 0.933)
    No Information Rate : 0.6217         
    P-Value [Acc > NIR] : < 2e-16        
                                         
                  Kappa : 0.8349         
                                         
 Mcnemars Test P-Value : 0.04276        
                                         
            Sensitivity : 0.8825         
            Specificity : 0.9473         
         Pos Pred Value : 0.9107         
         Neg Pred Value : 0.9298         
             Prevalence : 0.3783         
         Detection Rate : 0.3339         
   Detection Prevalence : 0.3666         
      Balanced Accuracy : 0.9149         
                                         
       'Positive' Class : 0
	   
#resampling	   
>resamps <- resamples(list(m1 = C5Fit1_2, m2 = C5Fit1_trained))
>summary(resamps)
Call:
summary.resamples(object = resamps)

Models: m1, m2 
Number of resamples: 30 

Accuracy 
        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NAs
m1 0.9017497 0.9178174 0.9245791 0.9232245 0.9296770 0.9366577    0
m2 0.9044415 0.9191375 0.9258760 0.9250176 0.9283311 0.9555855    0

Kappa 
        Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NAs
m1 0.7906646 0.8259395 0.8386734 0.8359096 0.8486635 0.8650006    0
m2 0.7972497 0.8285197 0.8413938 0.8402714 0.8485471 0.9057639    0

> diffs <- diff(resamps)
> summary(diffs)
Call:
summary.diff.resamples(object = diffs)

p-value adjustment: bonferroni 
Upper diagonal: estimates of the difference
Lower diagonal: p-value for H0: difference = 0

Accuracy 
   m1     m2       
m1        -0.001793
m2 0.4597          

Kappa 
   m1     m2       
m1        -0.004362
m2 0.3997

> xyplot(resamps, what = "BlandAltman")

# Finish with C5.0 model training_2
# Starting Random Forest algorithm training

#setting control
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#training the basic RF model
rfFit1 <- train(brand~., data = training, method = "rf", trControl=fitControl, tuneLength = 1)
rfFit1
Random Forest 

7424 samples
   6 predictor
   2 classes: '0', '1' 

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 1 times) 
Summary of sample sizes: 6682, 6681, 6683, 6681, 6682, 6681, ... 
Resampling results:

  Accuracy   Kappa    
  0.9230861  0.8366767

Tuning parameter 'mtry' was held constant at a value of 11

#tuning the model
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
rfFit1_trained <- train(brand~., data = training, method = "rf", trControl=fitControl, tuneLength = 15)
rfFit1_trained
Random Forest 

7424 samples
   6 predictor
   2 classes: '0', '1' 

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 3 times) 
Summary of sample sizes: 6681, 6681, 6682, 6682, 6681, 6681, ... 
Resampling results across tuning parameters:

  mtry  Accuracy   Kappa    
   2    0.6217673  0.0000000
   4    0.8548382  0.6838300
   6    0.9081801  0.8053162
   8    0.9187765  0.8277583
  11    0.9229075  0.8362181
  13    0.9233569  0.8371632
  15    0.9237162  0.8378825
  18    0.9234472  0.8372855
  20    0.9226839  0.8356872
  22    0.9218307  0.8339218
  24    0.9217408  0.8337557
  27    0.9203047  0.8306875
  29    0.9197654  0.8294892
  31    0.9188224  0.8274958
  34    0.9187328  0.8273510

Accuracy was used to select the optimal model using the largest value.
The final value used for the model was mtry = 15.

#looking for the variables importance
rfFitImp <- varImp(rfFit1)
rfFitImp
rf variable importance

  only 20 most important variables shown (out of 34)

           Overall
salary   100.00000
age       55.59815
credit    13.40036
elevel1    0.79285
elevel3    0.75823
elevel4    0.74216
elevel2    0.61512
zipcode6   0.49252
zipcode1   0.47393
zipcode4   0.47141
zipcode3   0.45266
zipcode7   0.42882
zipcode5   0.36245
zipcode8   0.35798
zipcode2   0.34658
car15      0.22888
car8       0.15144
car7       0.14359
car10      0.12875
car9       0.09955

#predictions
>prediction1 <- predict(rfFit1, testing)
>prediction2 <- predict(rfFit1_trained, testing)
>confusionMatrix(prediction1, testing$brand)
Confusion Matrix and Statistics

          Reference
Prediction    0    1
         0  837   97
         1   99 1441
                                          
               Accuracy : 0.9208          
                 95% CI : (0.9094, 0.9311)
    No Information Rate : 0.6217          
    P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.8315          
                                          
 Mcnemars Test P-Value : 0.9431          
                                          
            Sensitivity : 0.8942          
            Specificity : 0.9369          
         Pos Pred Value : 0.8961          
         Neg Pred Value : 0.9357          
             Prevalence : 0.3783          
         Detection Rate : 0.3383          
   Detection Prevalence : 0.3775          
      Balanced Accuracy : 0.9156          
                                          
       'Positive' Class : 0               
                                          
> confusionMatrix(prediction2, testing$brand)
Confusion Matrix and Statistics

          Reference
Prediction    0    1
         0  843   94
         1   93 1444
                                          
               Accuracy : 0.9244          
                 95% CI : (0.9133, 0.9345)
    No Information Rate : 0.6217          
    P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.8393          
                                          
 Mcnemars Test P-Value : 1               
                                          
            Sensitivity : 0.9006          
            Specificity : 0.9389          
         Pos Pred Value : 0.8997          
         Neg Pred Value : 0.9395          
             Prevalence : 0.3783          
         Detection Rate : 0.3407          
   Detection Prevalence : 0.3787          
      Balanced Accuracy : 0.9198          
                                          
       'Positive' Class : 0               
                              
#loading Incomplete Survey
> IncompleteData <- read.csv(file="C:/Users/ms-msi/Desktop/Data Scientist/Course2/Task2/data source/SurveyIncomplete.csv")
> head(IncompleteData)

#preprocessing
> str(IncompleteData)
'data.frame':	5000 obs. of  7 variables:
 $ salary : num  150000 82524 115647 141443 149211 ...
 $ age    : int  76 51 34 22 56 26 64 50 26 46 ...
 $ elevel : int  1 1 0 3 0 4 3 3 2 3 ...
 $ car    : int  3 8 10 18 5 12 1 9 3 18 ...
 $ zipcode: int  3 3 2 2 3 1 2 0 4 6 ...
 $ credit : num  377980 141658 360980 282736 215667 ...
 $ brand  : int  1 0 1 1 1 1 1 1 1 0 ...
> IncompleteData$car <- as.factor(IncompleteData$car)
> IncompleteData$zipcode <- as.factor(IncompleteData$zipcode)
> IncompleteData$elevel <- as.factor(IncompleteData$elevel)
> IncompleteData$brand <- as.factor(IncompleteData$brand)

#final prediction
> p <- predict(rfFit1_trained, IncompleteData)

#checking the predicted values and factual
output2 <- cbind(IncompleteData, p)

#postResample checking
> postResample(p, IncompleteData$brand)
  Accuracy      Kappa 
0.38660000 0.01203978 

#calculating complete ansverws + predicted
> summary(p)
   0    1 
1880 3120

> summary(CompleteResponses$brand)
   0    1 
3744 6154 