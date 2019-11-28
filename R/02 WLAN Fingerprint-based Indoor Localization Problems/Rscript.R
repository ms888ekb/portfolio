library(doParallel)
library(dplyr)
library(tidyr)
library(stringr)
library(caret)
# install.packages("devtools")
# library(devtools)
# install_github("berndbischl/BBmisc")
# library(BBmisc)

registerDoParallel(cores=10)

setwd("C:\\Users\\ms-msi\\Desktop\\Data Scientist\\_C3\\T3")
getwd()

df <- read.csv(file="./data/trainingData.csv")

str(df, list.len=ncol(df))

# is.na(df)

df$SPACEID <- as.factor(df$SPACEID)
df$FLOOR <- as.factor(df$FLOOR)
df$BUILDINGID <- as.factor(df$BUILDINGID)
df$RELATIVEPOSITION <- as.factor(df$RELATIVEPOSITION)

df$SPACEID <- str_pad(df$SPACEID, 3, pad = "0")

# df_smpl <- mutate(df, combined_loc = paste(BUILDINGID,FLOOR,SPACEID, sep = ''))
  
df <- mutate(df, combined_loc = paste(BUILDINGID,FLOOR,SPACEID,RELATIVEPOSITION, sep = ''))

# df_smpl$SPACEID <- NULL
# df_smpl$FLOOR <- NULL
# df_smpl$BUILDINGID <- NULL
# df_smpl$RELATIVEPOSITION <- NULL

col_drop <- c('TIMESTAMP','PHONEID','USERID','LATITUDE','LONGITUDE')
#col_drop <- c('TIMESTAMP','USERID','LATITUDE','LONGITUDE')

df$combined_loc <- as.factor(df$combined_loc)


df$SPACEID <- NULL
df$FLOOR <- NULL
df$BUILDINGID <- NULL
df$RELATIVEPOSITION <- NULL


# df_smpl$combined_loc <- as.factor(df_smpl$combined_loc)
# df_smpl <- df_smpl[sample(1:nrow(df_smpl), nrow(df_smpl),replace=FALSE),]

df$combined_loc <- as.factor(df$combined_loc)
# df <- df[sample(1:nrow(df), nrow(df),replace=FALSE),]

# col_drop <- c('TIMESTAMP','PHONEID','USERID','LATITUDE','LONGITUDE')
col_drop <- c('TIMESTAMP','USERID','LATITUDE','LONGITUDE')


# df_loc_smpl <- df_smpl[ , !(names(df_smpl) %in% col_drop)]

# Delete columns with zero variance, i.e. constant
df_loc <- df[ , !(names(df) %in% col_drop)]

##
# df_loc$PHONEID <- as.integer(df_loc$PHONEID)
##

str(df_loc, list.len=ncol(df_loc))


# Delete columns with zero variance, i.e. constant
# which(apply(df_loc, 2, var)==0)

# df_loc_smpl <- df_loc_smpl[ , apply(df_loc_smpl, 2, var) != 0]

df_loc <- df_loc[ , apply(df_loc, 2, var) != 0]
# df_loc[,1:465]

#df_loc_cs <- scale(df_loc[, 1:465])
#colMeans(df_loc_cs)
#apply(df_loc_cs, 2, sd)

# df_loc_cs <- normalize(df_loc[, 1:465], method = "standardize", range = c(-1, 1), margin = 2, on.constant = "quiet")

#hist(df_loc_cs$WAP005, xlim=c(-100,100), ylim=c(0,50))

# Create train and test sets (simple):
# partition_s <- createDataPartition(df_loc_smpl$combined_loc, p = .75, list = FALSE)
# training_loc_s <- df_loc_smpl[partition_s,]
# testing_loc_s <- df_loc_smpl[-partition_s,]


# Create train and test sets:
partition <- createDataPartition(df_loc$combined_loc, p = .75, list = FALSE)
training_loc <- df_loc[partition,]
testing_loc <- df_loc[-partition,]


#C5 model training for location prediction
fitControl0 <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

C5Fit <- train(combined_loc~., data = training_loc, method = "C5.0", trControl=fitControl0, tuneLength = 3)

C5Fit0 <- train(combined_loc~., data = training_loc, method = "C5.0", trControl=fitControl0, tuneLength = 3)

C5Pred <- predict(C5Fit0, testing_loc)

postResample(C5Pred, testing_loc$combined_loc)

#Random Forest model training for location prediction
rfFit <- train(combined_loc~., data = training_loc, method = "rf", trControl=fitControl0, tuneLength = 1)

rfFit_test <- train(combined_loc~., data = training_loc, method = "rf", trControl=fitControl0, tuneLength = 1)

rfPred_test <- predict(rfFit_test, testing_loc)

postResample(rfPred_test, testing_loc$combined_loc)

#Random Forest model training for location prediction (simplified):
rfFit_s <- train(combined_loc~., data = training_loc_s, method = "rf", trControl=fitControl0, tuneLength = 1)

rfPred_s <- predict(rfFit_s, testing_loc_s)

postResample(rfPred_s, testing_loc_s$combined_loc)



str(testing_loc, list.len=ncol(testing_loc))

rfFit_ph <- train(combined_loc~., data = training_loc, method = "rf", trControl=fitControl0, tuneLength = 1)
rfPred_ph <- predict(rfFit_ph, testing_loc)
postResample(rfPred_ph, testing_loc$combined_loc)


#Validation
postResample(rfPred, testing_loc$combined_loc)

#KNN model training for location prediction
fitControl_knn <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#knnFit <- train(combined_loc~., data = training_loc, method = "knn", trControl=fitControl_knn, preProcess = c("center","scale"), tuneLength = 10)

knnFit <- train(combined_loc~., data = training_loc, method = "knn", trControl=fitControl_knn, preProcess = c("center","scale"), tuneGrid = expand.grid(k = 1:5))

knnPred <- predict(knnFit, testing_loc)

postResample(knnPred, testing_loc$combined_loc)

# PCA
df.pr <- prcomp(df_loc[c(1:465)], center = TRUE, scale = TRUE)
summary(df.pr)

df.pcst <- df.pr$x[,1:400]
df_pca <- as.data.frame(df.pcst)
df_pca <- cbind(df_pca, df_loc$combined_loc)
colnames(df_pca)[401] <- "combined_loc"

str(df_pca, list.len=ncol(df_pca))

# Create train and test sets for PCA:
partitionPCA <- createDataPartition(df_pca$combined_loc, p = .75, list = FALSE)
trainPCA <- df_pca[partitionPCA,]
testPCA <- df_pca[-partitionPCA,]

rfFit_PCA <- train(combined_loc~., data = trainPCA, method = "rf", trControl=fitControl0, tuneLength = 1)

knnFit_PCA <- train(combined_loc~., data = trainPCA, method = "knn", trControl=fitControl0, tuneLength = 1)


### Use only GT series phones
str(df_gt)

df_gt <- read.csv(file="./data/trainingData.csv")

df_gt <- filter(df_gt, between(PHONEID, 1, 7))

df_gt$SPACEID <- as.factor(df_gt$SPACEID)
df_gt$FLOOR <- as.factor(df_gt$FLOOR)
df_gt$BUILDINGID <- as.factor(df_gt$BUILDINGID)
df_gt$RELATIVEPOSITION <- as.factor(df_gt$RELATIVEPOSITION)

df_gt$SPACEID <- str_pad(df_gt$SPACEID, 3, pad = "0")

df_gt <- mutate(df_gt, combined_gt = paste(BUILDINGID,FLOOR,SPACEID,RELATIVEPOSITION, sep = ''))

df_gt$SPACEID <- NULL
df_gt$FLOOR <- NULL
df_gt$BUILDINGID <- NULL
df_gt$RELATIVEPOSITION <- NULL

df_gt$combined_gt <- as.factor(df_gt$combined_gt)

df_gt <- df_gt[sample(1:nrow(df_gt), nrow(df_gt),replace=FALSE),]

df_gt <- df_gt[ , !(names(df_gt) %in% col_drop)]

df_gt <- df_gt[ , apply(df_gt, 2, var) != 0]

summary(df_gt)

partition_gt <- createDataPartition(df_gt$combined_gt, p = .75, list = FALSE)

training_gt <- df_gt[partition_gt,]
testing_gt <- df_gt[-partition_gt,]

rfFit_gt <- train(combined_gt~., data = training_gt, method = "rf", trControl=fitControl0, tuneLength = 3)

rfPred_gt <- predict(rfFit_gt, testing_gt)
postResample(rfPred_gt, testing_gt$combined_gt)

# GBM
registerDoSEQ()
gbmFit <- train(combined_loc~., data = training_loc, method = "gbm", trControl=fitControl0)

# Filter on building id

df_b <- read.csv(file="./data/trainingData.csv")

df_b0 <- filter(df_b, BUILDINGID == "0" )
df_b1 <- filter(df_b, BUILDINGID == "1" )
df_b2 <- filter(df_b, BUILDINGID == "2" )


df_b0 <- mutate(df_b0, combined_b = paste(BUILDINGID,FLOOR,SPACEID,RELATIVEPOSITION, sep = ''))
df_b0$SPACEID <- NULL
df_b0$FLOOR <- NULL
df_b0$BUILDINGID <- NULL
df_b0$RELATIVEPOSITION <- NULL
df_b0$combined_b <- as.factor(df_b0$combined_b)
df_b0 <- df_b0[ , apply(df_b0, 2, var) != 0]
df_b0 <- df_b0[ , !(names(df_b0) %in% col_drop)]

df_b1 <- mutate(df_b1, combined_b = paste(BUILDINGID,FLOOR,SPACEID,RELATIVEPOSITION, sep = ''))
df_b1$SPACEID <- NULL
df_b1$FLOOR <- NULL
df_b1$BUILDINGID <- NULL
df_b1$RELATIVEPOSITION <- NULL
df_b1$combined_b <- as.factor(df_b1$combined_b)
df_b1 <- df_b1[ , apply(df_b1, 2, var) != 0]
df_b1 <- df_b1[ , !(names(df_b1) %in% col_drop)]

df_b2 <- mutate(df_b2, combined_b = paste(BUILDINGID,FLOOR,SPACEID,RELATIVEPOSITION, sep = ''))
df_b2$SPACEID <- NULL
df_b2$FLOOR <- NULL
df_b2$BUILDINGID <- NULL
df_b2$RELATIVEPOSITION <- NULL
df_b2$combined_b <- as.factor(df_b2$combined_b)
df_b2 <- df_b2[ , apply(df_b2, 2, var) != 0]
df_b2 <- df_b2[ , !(names(df_b2) %in% col_drop)]

partition0 <- createDataPartition(df_b0$combined_b, p = .75, list = FALSE)
train_b0 <- df_b0[partition0,]
test_b0 <- df_b0[-partition0,]

partition1 <- createDataPartition(df_b1$combined_b, p = .75, list = FALSE)
train_b1 <- df_b1[partition1,]
test_b1 <- df_b1[-partition1,]

partition2 <- createDataPartition(df_b2$combined_b, p = .75, list = FALSE)
train_b2 <- df_b2[partition2,]
test_b2 <- df_b2[-partition2,]

# gbmFit0 <- train(combined_b~., data = train_b0, method = "gbm", trControl=fitControl0, tuneLength = 1)
rfFit0 <- train(combined_b~., data = train_b0, method = "rf", trControl=fitControl0, tuneLength = 3)

rfPred_b0 <- predict(rfFit0, test_b0)
postResample(rfPred_b0, test_b0$combined_b)

rfFit1 <- train(combined_b~., data = train_b1, method = "rf", trControl=fitControl0, tuneLength = 3)

rfPred_b1 <- predict(rfFit1, test_b1)
postResample(rfPred_b1, test_b1$combined_b)

rfFit2 <- train(combined_b~., data = train_b2, method = "rf", trControl=fitControl0, tuneLength = 3)

rfPred_b2 <- predict(rfFit2, test_b2)
postResample(rfPred_b2, test_b2$combined_b)

###

modeldata <- resamples(list(C50 = C5Fit0, RF = rfFit, kNN = knnFit))

summary(modeldata)

###

kappaDiffs <- diff(modeldata, metric = "Kappa")
summary(kappaDiffs)

parallelplot(modeldata, metric = "Kappa")
dotplot(modeldata, metric = "Kappa")

dotplot(kappaDiffs, metric = "Kappa")

splom(modeldata, metric = "Kappa")

varImp(rfFit)
