setwd("C:/Users/ragha/Desktop/Raghav/Great Learning/Projects/Project 5")
library(readxl)
library(corrplot)
library(class)
library(gridExtra)
library(tidyverse)
library(car)
library(class)
library(caret)
library(ROCR)
library(ineq)
library(dplyr)
library(broom)
library(caTools)
library(DMwR)
library(xgboost)
library(ipred)
library(rpart)
library(gbm)
churn <- read.csv('Cars.csv')

head(churn)
tail(churn)
str(churn)
summary(churn)


churn$Gender <- factor(churn$Gender,levels = c('Male','Female'), labels = c(0,1))
churn$Engineer <- as.factor(churn$Engineer)
churn$MBA <- as.factor(churn$MBA)
churn$license <- as.factor(churn$license)

prop.table(table(churn$Transport))*100


#Histogram
par(mfrow = c(2,2))
for(i in names(churn[-c(2,3,4,8,9)])){
  hist(churn[,i], xlab = names(churn[i]), col = "red", border = "black", ylab = "Frequency",
       main =paste("Histogram of", names(churn[i])),col.main = "darkGreen")
}
par(mfrow = c(1,1))

#Barplot
table1 <- churn %>% group_by(Transport,Gender) %>% summarise("Values" = n())
table1[c(1,2)] <- lapply(table1[c(1,2)], as.factor)
A <- ggplot(table1,aes(Gender,Values,fill = Transport)) + geom_bar(stat = "identity",color = "Black", position = 'dodge')+
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)


table2 <- churn %>% group_by(Transport,MBA) %>% summarise("Values" = n())
table2[c(1,2)] <- lapply(table2[c(1,2)], as.factor)
B <- ggplot(data = table2, aes(MBA,Values, fill = Transport)) + geom_bar(stat = 'identity', position = 'dodge',color = "Black")+
  geom_text(aes(label = Values),
            size = 3,
            vjust = -0.2,
            position = position_dodge(width = 0.9))

table3 <- churn %>% group_by(Transport,'License' = license) %>% summarise("Values" = n())
table3[c(1,2)] <- lapply(table3[c(1,2)], as.factor)
C <- ggplot(table3,aes(License,Values,fill = Transport)) + geom_bar(stat = "identity",color = "Black", position = 'dodge')+
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)


table4 <- churn %>% group_by(Transport,Engineer) %>% summarise("Values" = n())
table4[c(1,2)] <- lapply(table4[c(1,2)], as.factor)
D <- ggplot(data = table4, aes(Engineer,Values, fill = Transport)) + geom_bar(stat = 'identity', position = 'dodge',color = "Black")+
  geom_text(aes(label = Values),
            size = 3,
            vjust = -0.2,
            position = position_dodge(width = 0.9))

grid.arrange(A,B,C,D)

#Boxplot
A <- ggplot(data = churn, aes(as.factor(Transport),Age)) + geom_boxplot(color = c("Red","Blue","Green")) + xlab("Transport") + ggtitle("Age vs Transport")
B <- ggplot(data = churn, aes(as.factor(Transport),Work.Exp)) + geom_boxplot(color = c("Red","Blue","Green")) + xlab("Transport") + ggtitle("Work Experience vs Transport")
C <- ggplot(data = churn, aes(as.factor(Transport),Salary)) + geom_boxplot(color = c("Red","Blue","Green")) + xlab("Transport") + ggtitle("Salary vs Transport")
D <- ggplot(data = churn, aes(as.factor(Transport),Distance)) + geom_boxplot(color = c("Red","Blue","Green")) + xlab("Transport") + ggtitle("Distance vs Transport")

grid.arrange(A,B,C,D)
par(mfrow =c(1,1))

#Missing values
any(is.na(churn))
colSums(is.na(churn))
churn <- na.omit(churn)

#Outlier Detection and Treatment 

#Cooks distance for outliers detection

cooksd <- cooks.distance(glm(as.numeric(Transport)~.,data = churn))
plot(cooksd,pch="*", cex =1, main ="Cook's Distance", col ="blue")
abline(h = 4*mean(cooksd, na.rm = TRUE), col = "DarkRED",lwd = 2)
#Any value above red line  indicates influential values or outliers

#Treatment using mean imputation method
mydata <- churn[,c(1,5:7)]

Outliers<- mydata[1:60,]


for(i in c(1:4)) {
  Outliers[,i] <- NA
  Box <-boxplot(mydata[,i],plot =F)$out
  if (length(Box)>0){
    Outliers[1:length(Box),i] <- Box
  }
}

for(i in names(mydata[c(1,2,4)])){
  q = quantile(mydata[,i])
  mydata[mydata[,i] > q[[4]] + 1.5*(q[[4]] - q[[2]]) ,i ] <- round(q[[4]] + 1.5*(q[[4]] - q[[2]]))
}
q = quantile(mydata[,'Salary'])
mydata[mydata$Salary > q[[4]] + 3*(q[[4]] - q[[2]]),'Salary'] <- round(q[[4]] + 3*(q[[4]] - q[[2]]))

churn <- cbind.data.frame(mydata,churn[,c(2:4,8,9)])
max(churn$Distance)

### Converting dependent variable into 2 levels
churn$Transport <- ifelse(churn$Transport == "Car",1,0)
churn$Transport <- as.factor(churn$Transport)
#Multicollinearity
cor <- cor(churn[c(1:4)])
corrplot.mixed(cor,lower = "number", upper = "pie")

#Variance Inflation Factor
vif(glm(as.numeric(Transport)~., data = churn))
vif(glm(as.numeric(Transport)~. -Work.Exp, data = churn))

#vif < 5. No significant evidence to treat multicollinearity



#-------------------------------------------------------------------
#1. Logistic Regression

#Testing assumptions of logistic regression 

#1. Linear relationship b/w log odds and independent variables
model <- train(Transport~., data = churn, method = 'glm')
prob <- predict(model,newdata = churn, type = 'prob')[,1]
mydata <- churn

mydata<- mydata %>% select_if(is.numeric) 
predictors <- colnames(mydata)

# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(prob/(1-prob))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

#2.Outliers
#3. Multicollinearity


#Model Formation

log.churn <- churn

set.seed(100)
split <- sample.split(log.churn$Transport,SplitRatio = 0.75)
smote.train <- log.churn[split == TRUE,]

#Handling imbalance dataset
log.train <- SMOTE(Transport ~., smote.train,perc.over = 3000,k = 5,perc.under = 150)
prop.table(table(log.train$Transport))

log.test <- log.churn[split == FALSE,]

#Visualing the smote analysis
SA <- ggplot(smote.train) + geom_point(aes(Age,Work.Exp, col = Transport)) + ggtitle('Before Smote')
SB <- ggplot(log.train) + geom_point(aes(Age,Work.Exp, col = Transport)) + ggtitle(('After Smote'))
grid.arrange(SA,SB)

#Stepwise logistic regression with 10 - fold cross validation
train.control <- trainControl(method = "cv", number = 10)
model <- train(Transport~. -Work.Exp, data = log.train, method = "glm",trControl = train.control,family =binomial())
vif(glm(as.numeric(Transport)~. -Work.Exp,data = log.train))
summary(model)

model <- train(Transport~. -Work.Exp - Engineer, data = log.train, method = "glm",trControl = train.control,family =binomial())
summary(model)

plot(varImp(model), main = "Important Variables")

log.train$pred.class <- predict(model,log.train, type = "raw")
confusionMatrix(as.factor(log.train$Transport),as.factor(log.train$pred.class))


#Predicting Testing data set

log.test$prob.pred <- predict(model,newdata = log.test, type = "prob")[,"1"]
log.test$pred.class <- predict(model,log.test, type = "raw")

log.cm.test <-confusionMatrix(as.factor(log.test$Transport),as.factor(log.test$pred.class))
log.cm.test


log.test.Accuracy <- log.cm.test$overall[[1]]
log.test.Accuracy

log.test.class.err <- 1 - log.test.Accuracy
log.test.class.err

log.test.Sensitivity <- log.cm.test$byClass[[1]] 
log.test.Sensitivity

log.test.Specificity <- log.cm.test$byClass[[2]]
log.test.Specificity

log.F1_score <- F_meas(data = log.test$pred.class,reference = log.test$Transport)
log.F1_score

#ROCR
log.test.pred <- prediction(predictions = as.numeric(log.test$pred.class),labels = as.numeric(log.test$Transport))
log.test.perf <- performance(log.test.pred,"tpr","fpr")
plot(log.test.perf, col = "Red",lwd =2, main = "ROC in Logistic regression")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.77,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


#AUC
log.test.auc <- performance(log.test.pred,"auc")
log.test.auc <- log.test.auc@y.values[[1]]
log.test.auc <- log.test.auc[[1]]
log.test.auc


#KS
log.test.ks <- log.test.perf@y.values[[1]] - log.test.perf@x.values[[1]]
log.test.ks <- log.test.ks[2]
log.test.ks

#Gini
log.test.gini <- ineq(log.test$prob.pred,type ="Gini")
log.test.gini

#Concordance
library(InformationValue)
log.test.cord <- Concordance(actuals = log.test$Transport,predictedScores = log.test$prob.pred)
log.test.cord<- log.test.cord$Concordance
log.test.cord
detach("package:InformationValue", unload = TRUE)


test.model.log <- t(data.frame(log.test.Accuracy,log.test.class.err,log.test.Sensitivity,log.test.Specificity,log.test.ks,log.test.auc,log.test.gini,log.test.cord,log.F1_score))
row.names(test.model.log) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance","F1_score")
test.model.log


#-------------------------------------------------------------------------------
# KNN

knn.churn <- churn

#Splitting of dataset
set.seed(100)
split <- sample.split(knn.churn$Transport,SplitRatio = 0.75)
knn.train <- knn.churn[split == TRUE,]
prop.table(table(knn.train$Transport))

knn.test <- knn.churn[split == FALSE,]

#Feature Scaling
knn.train[,5:8] <- as.numeric(unlist(knn.train[,5:8]))
knn.train[,c(1:8)] <- scale(knn.train[,c(1:8)])

knn.test[,5:8] <- as.numeric(unlist(knn.test[,5:8]))
knn.test[,c(1:8)] <- scale(knn.test[,c(1:8)])


#Model Formation
control <- trainControl(method = 'cv', number = 10)

kn <- train(Transport~. - Work.Exp,
            method     = "knn",
            tuneGrid   = expand.grid(k = 1:9),
            trControl  = control,
            metric     = "Accuracy",
            data       = knn.train)
kn
plot(kn,main= "KNN")
plot(varImp(kn),main = "Important Variables")

knn.train$pred.class <- predict(kn,newdata = knn.train,type = "raw")
confusionMatrix(as.factor(knn.train$Transport),knn.train$pred.class)

#Testing on 

knn.test$pred.class <- predict(kn,newdata = knn.test, type = "raw")
knn.test$pred.prob <-  predict(kn, newdata = knn.test, type = "prob")[,"1"]

knn.cm.test <-confusionMatrix(as.factor(knn.test$Transport),knn.test$pred.class)
knn.cm.test

knn.test.Accuracy <- knn.cm.test$overall[[1]]
knn.test.Accuracy

knn.test.class.err <- 1 - knn.test.Accuracy
knn.test.class.err

knn.test.Sensitivity <- knn.cm.test$byClass[[1]] 
knn.test.Sensitivity

knn.test.Specificity <- knn.cm.test$byClass[[2]]
knn.test.Specificity

knn.F1_score <- F_meas(data = as.factor(knn.test$pred.class),reference = as.factor(knn.test$Transport))
knn.F1_score


#ROCR
knn.test.pred <- prediction(labels = as.numeric(knn.test$Transport),predictions = as.numeric(knn.test$pred.class))
knn.test.perf <- performance(knn.test.pred,"tpr","fpr")
plot(knn.test.perf, col = "Red", lwd =2, main ="ROC in KNN")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)

#AUC
knn.test.auc <- performance(knn.test.pred,"auc")
knn.test.auc <- knn.test.auc@y.values[[1]]
knn.test.auc <- knn.test.auc[[1]]
knn.test.auc

#KS
knn.test.ks <- knn.test.perf@y.values[[1]] - knn.test.perf@x.values[[1]]
knn.test.ks <- knn.test.ks[2]
knn.test.ks

#Gini
knn.test.gini <- ineq(knn.test$pred.prob,type ="Gini")
knn.test.gini

#Concordance
library(InformationValue)
knn.test.cord <- Concordance(actuals = knn.test$Transport,predictedScores = knn.test$pred.prob)
knn.test.cord<- knn.test.cord$Concordance
knn.test.cord
detach("package:InformationValue", unload = TRUE)


test.model.knn <- t(data.frame(knn.test.Accuracy,knn.test.class.err,knn.test.Sensitivity,knn.test.Specificity,knn.test.ks,knn.test.auc,knn.test.gini,knn.test.cord,knn.F1_score))
row.names(test.model.knn) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance","F1_score")
test.model.knn


#-------------------------------------------------------------------------------
#Naive Bayes

nb.churn <- churn

#Splitting of dataset
set.seed(100)
split <- sample.split(nb.churn$Transport,SplitRatio = 0.75)
smote.train <- nb.churn[split == TRUE,]
set.seed(100)
nb.train <- SMOTE(Transport~., data = smote.train,perc.over = 3000, k=5,perc.under = 150)
nb.test <- nb.churn[split == FALSE,]

#Model Formation
control <- trainControl(method = 'cv', number = 10)

search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0:5,
  adjust = seq(0, 5, by = 1)
)

nb <- train(
  x = nb.train[-9],
  y = as.factor(nb.train$Transport),
  method = "nb",
  trControl = control,
  tuneGrid = search_grid
)

nb
nb$results %>% top_n(5, wt = Accuracy) %>% arrange(desc(Accuracy))

plot(nb)
plot(varImp(nb), main = "Important Variable")

nb.train$pred.class <- predict(nb,newdata = nb.train,type = "raw")
nb.train$pred.prob <- predict(nb, newdata = nb.train, type ="prob")[,"1"]
confusionMatrix(as.factor(nb.train$Transport),nb.train$pred.class)

#Testing of model

nb.test$pred.class <- predict(nb,newdata = nb.test, type = "raw")
nb.test$pred.prob <-  predict(nb, newdata = nb.test, type = "prob")[,"1"]

nb.cm.test <-confusionMatrix(as.factor(nb.test$Transport),nb.test$pred.class)
nb.cm.test

nb.test.Accuracy <- nb.cm.test$overall[[1]]
nb.test.Accuracy

nb.test.class.err <- 1 - nb.test.Accuracy
nb.test.class.err

nb.test.Sensitivity <- nb.cm.test$byClass[[1]] 
nb.test.Sensitivity

nb.test.Specificity <- nb.cm.test$byClass[[2]]
nb.test.Specificity

nb.F1_score <- F_meas(data = nb.test$pred.class,reference = nb.test$Transport)
nb.F1_score

#ROCR
nb.test.pred <- prediction(labels = as.numeric(nb.test$Trans),predictions = as.numeric(nb.test$pred.class))
nb.test.perf <- performance(nb.test.pred,"tpr","fpr")
plot(nb.test.perf, col = "Red", lwd = 2, main ="ROC in Naive Bayes")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


#AUC
nb.test.auc <- performance(nb.test.pred,"auc")
nb.test.auc <- nb.test.auc@y.values[[1]]
nb.test.auc <- nb.test.auc[[1]]
nb.test.auc

#KS
nb.test.ks <- nb.test.perf@y.values[[1]] - nb.test.perf@x.values[[1]]
nb.test.ks <- nb.test.ks[2]
nb.test.ks

#Gini
nb.test.gini <- ineq(nb.test$pred.prob,type ="Gini")
nb.test.gini

#Concordance
library(InformationValue)
nb.test.cord <- Concordance(actuals = nb.test$Transport,predictedScores = nb.test$pred.prob)
nb.test.cord<- nb.test.cord$Concordance
nb.test.cord
detach("package:InformationValue", unload = TRUE)


test.model.nb <- t(data.frame(nb.test.Accuracy,nb.test.class.err,nb.test.Sensitivity,nb.test.Specificity,nb.test.ks,nb.test.auc,nb.test.gini,nb.test.cord,nb.F1_score))
row.names(test.model.nb) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance","F1_score")
test.model.nb


#Bagging
bag.churn <- churn

set.seed(100)
split <- sample.split(bag.churn$Transport,SplitRatio = 0.75)
smote.train <- bag.churn[split == TRUE,]
bag.train <- SMOTE(Transport~., data = smote.train,perc.over = 3000, k=5,perc.under = 150)
bag.test <- bag.churn[split == FALSE,]


bag.model<- bagging(Transport ~.,
                          data=bag.train,
                          control=rpart.control(maxdepth=5, minsplit=4))

RocImp2 = varImp(bag.model)
write.csv(RocImp2, file = 'RocImp2.csv')
VIMP <- read.csv('RocImp2.csv')
VIMP
ggplot(VIMP) + geom_bar(aes(x = reorder(X,-Overall), y = Overall),fill= 'LightBlue',stat ='identity') +
  xlab('Variables') + ylab('Importance') +ggtitle('Variable Importance')

bag.train$pred.class <- predict(bag.model, bag.train,type = "class")
confusionMatrix(as.factor(bag.train$Transport),as.factor(bag.train$pred.class))

#Testing the model
bag.test$pred.class <- predict(bag.model,newdata = bag.test, type = "class")
bag.test$pred.prob <-  predict(bag.model,newdata = bag.test, type = "prob")[,"1"]

bag.cm.test <-confusionMatrix(bag.test$Transport,bag.test$pred.class)
bag.cm.test

bag.test.Accuracy <- bag.cm.test$overall[[1]]
bag.test.Accuracy

bag.test.class.err <- 1 - bag.test.Accuracy
bag.test.class.err

bag.test.Sensitivity <- bag.cm.test$byClass[[1]] 
bag.test.Sensitivity

bag.test.Specificity <- bag.cm.test$byClass[[2]]
bag.test.Specificity

bag.F1_score <- F_meas(data = bag.test$pred.class,reference = bag.test$Transport)
bag.F1_score

#ROCR
bag.test.pred <- prediction(labels = as.numeric(bag.test$Transport),predictions =  as.numeric(bag.test$pred.class))
bag.test.perf <- performance(bag.test.pred,"tpr","fpr")
plot(bag.test.perf, col = "Red", lwd = 2, main ="ROC in Bagging")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


#AUC
bag.test.auc <- performance(bag.test.pred,"auc")
bag.test.auc <- bag.test.auc@y.values[[1]]
bag.test.auc <- bag.test.auc[[1]]
bag.test.auc

#KS
bag.test.ks <- bag.test.perf@y.values[[1]] - bag.test.perf@x.values[[1]]
bag.test.ks <- bag.test.ks[2]
bag.test.ks

#Gini
bag.test.gini <- ineq(bag.test$pred.prob,type ="Gini")
bag.test.gini

#Concordance
library(InformationValue)
bag.test.cord <- Concordance(actuals = bag.test$Transport,predictedScores = bag.test$pred.prob)
bag.test.cord<- bag.test.cord$Concordance
bag.test.cord
detach("package:InformationValue", unload = TRUE)



test.model.bag <- t(data.frame(bag.test.Accuracy,bag.test.class.err,bag.test.Sensitivity,bag.test.Specificity,bag.test.ks,bag.test.auc,bag.test.gini,bag.test.cord,bag.F1_score))
row.names(test.model.bag) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance","F1_score")
test.model.bag


#Boosting
gbm.churn <- churn

#Splitting of dataset
set.seed(100)
split <- sample.split(gbm.churn$Transport,SplitRatio = 0.75)
smote.train <- gbm.churn[split == TRUE,]
set.seed(100)
gbm.train <- SMOTE(Transport~., data = smote.train,perc.over = 3000, k=5,perc.under = 150)
gbm.test <- gbm.churn[split == FALSE,]

fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10)

gbmGrid <-  expand.grid(interaction.depth = 5, 
                        n.trees = 950, 
                        shrinkage = 0.01,
                        n.minobsinnode = c(5,10,15,20))

gbm.model <- train(Transport ~ ., data = gbm.train, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid)
gbm.model
plot(varImp(gbm.model), main = 'Important Variables')
gbm.train$pred.class <- predict(gbm.model,gbm.train,type = "raw")
confusionMatrix(as.factor(gbm.train$Transport),as.factor(gbm.train$pred.class))

#Testing the model
gbm.test$pred.class <- predict(gbm.model,newdata = gbm.test, type = "raw")
gbm.test$pred.prob <-  predict(gbm.model,newdata = gbm.test, type = "prob")[,"1"]

gbm.cm.test <-confusionMatrix(gbm.test$Transport,gbm.test$pred.class)
gbm.cm.test

gbm.test.Accuracy <- gbm.cm.test$overall[[1]]
gbm.test.Accuracy

gbm.test.class.err <- 1 - gbm.test.Accuracy
gbm.test.class.err

gbm.test.Sensitivity <- gbm.cm.test$byClass[[1]] 
gbm.test.Sensitivity

gbm.test.Specificity <- gbm.cm.test$byClass[[2]]
gbm.test.Specificity

gbm.F1_score <- F_meas(data = gbm.test$pred.class,reference = gbm.test$Transport)
gbm.F1_score

#ROCR
gbm.test.pred <- prediction(labels = as.numeric(gbm.test$Transport),predictions =  as.numeric(gbm.test$pred.class))
gbm.test.perf <- performance(gbm.test.pred,"tpr","fpr")
plot(gbm.test.perf, col = "Red", lwd = 2, main ="ROC in GBM")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


#AUC
gbm.test.auc <- performance(gbm.test.pred,"auc")
gbm.test.auc <- gbm.test.auc@y.values[[1]]
gbm.test.auc <- gbm.test.auc[[1]]
gbm.test.auc

#KS
gbm.test.ks <- gbm.test.perf@y.values[[1]] - gbm.test.perf@x.values[[1]]
gbm.test.ks <- gbm.test.ks[2]
gbm.test.ks

#Gini
gbm.test.gini <- ineq(gbm.test$pred.prob,type ="Gini")
gbm.test.gini

#Concordance
library(InformationValue)
gbm.test.cord <- Concordance(actuals = gbm.test$Transport,predictedScores = gbm.test$pred.prob)
gbm.test.cord<- gbm.test.cord$Concordance
gbm.test.cord
detach("package:InformationValue", unload = TRUE)


test.model.gbm <- t(data.frame(gbm.test.Accuracy,gbm.test.class.err,gbm.test.Sensitivity,gbm.test.Specificity,gbm.test.ks,gbm.test.auc,gbm.test.gini,gbm.test.cord,gbm.F1_score))
row.names(test.model.gbm) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance","F1_score")
test.model.gbm

#Combining performance measures of all models
combined.model <- cbind(test.model.knn,test.model.log,test.model.nb,test.model.bag,test.model.gbm)
colnames(combined.model) <- c("KNN","Logistic","Naive Bayes","Bagging","Boosting")
combined.model

#ROC curve of all models
par(mfrow = c(3,2))

plot(log.test.perf, col = "Red",lwd =2, main = "ROC in Logistic regression")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.77,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,1.2))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


plot(knn.test.perf, col = "Red", lwd =2, main ="ROC in KNN")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,1.2))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


plot(nb.test.perf, col = "Red", lwd = 2, main ="ROC in Naive Bayes")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,1.2))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)

plot(bag.test.perf, col = "Red", lwd = 2, main ="ROC in Bagging")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)

plot(gbm.test.perf, col = "Red", lwd = 2, main ="ROC in GBM")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)

par(mfrow = c(1,1))

