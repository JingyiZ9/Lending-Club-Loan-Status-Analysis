if (!require("pacman")) 
  install.packages("pacman")

pacman::p_load(
  "xgboost",
  "glmnet",
  "randomForest",
  "kernlab",
  "caret",
  "tidyr",
  "tidyverse",
  "lubridate"
)

setwd("~/Desktop/STAT542_project4") ###delete before submit
set.seed(6404)

#############################################
# Part 1 preprocessing the data
# import the csv data
data = read.csv("loan_stat542.csv", stringsAsFactors = FALSE)
test.id = read.csv("Project4_test_id.csv", stringsAsFactors = FALSE)
id = data$id

# binary response
data$loan_status <- ifelse(data$loan_status == "Fully Paid", 0, 1)

# large income with log transform
data$annual_inc=log(data$annual_inc+1)
data$revol_bal=log(data$revol_bal+1)

# deal with the missing values
# fill NA with 'Others' category
data$emp_length[is.na(data$emp_length)] = "Others"

# fill NA with zero
data$pub_rec_bankruptcies[is.na(data$pub_rec_bankruptcies)] = 0

# fill NA with mean
data$revol_util[is.na(data$revol_util)] = mean(data$revol_util[!is.na(data$revol_util)])
data$dti[is.na(data$dti)] = mean(data$dti[!is.na(data$dti)])
data$mort_acc[is.na(data$mort_acc)] = mean(data$mort_acc[!is.na(data$mort_acc)])

# add or transfer new features
data$fico_avg=(data$fico_range_low+data$fico_range_high)/2

d1=as.character(data$earliest_cr_line)
d2=paste('1-',d1,sep = "")
d3=dmy(d2)
data$earliest_cr_line = as.integer(round((as.Date("2018-1-1")-d3)/30))

data$term[which(data$term == '36 months')] = 36
data$term[which(data$term == '60 months')] = 60
data$term = as.integer(data$term)

# remove features
data= data %>%
  select(-c(id,grade,emp_title,purpose,title,zip_code,addr_state,application_type,fico_range_low,fico_range_high))

# group the features
group_features <- function(data){
  categorical.vars <- colnames(data)[which(sapply(data, 
                                                  function(x) is.character(x)))]
  data.matrix <- data[, !colnames(data) %in% categorical.vars, drop=FALSE]
  n.data <- nrow(data)
  for(var in categorical.vars){
    mylevels <- sort(unique(data[, var]))
    m <- length(mylevels)
    tmp.data <- matrix(0, n.data, m)
    col.names <- NULL
    for(j in 1:m){
      tmp.data[data[, var]==mylevels[j], j] <- 1
      col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
    }
    colnames(tmp.data) <- col.names
    data.matrix <- cbind(data.matrix, tmp.data)
  }
  return(as.matrix(data.matrix))
}

data = group_features(data)

# split the data into three sets of training/test pairs
split <- function(data,s){
  loc = which(id %in% test.id[, s])
  train = data[-loc, ]
  test = data[loc, ]
  test.x = test[, colnames(test) != 'loan_status']
  test.y = test[, colnames(test) == 'loan_status']
  
  return(list(train = train, test.x = test.x, test.y = test.y))
}

set1 = split(data,1)
train_1 = as.data.frame(set1$train)
test_1.x = as.data.frame(set1$test.x)
test_1.y = set1$test.y

set2 = split(data,2)
train_2 = as.data.frame(set2$train)
test_2.x = as.data.frame(set2$test.x)
test_2.y = set2$test.y

set3 = split(data,3)
train_3 = as.data.frame(set3$train)
test_3.x = as.data.frame(set3$test.x)
test_3.y = set3$test.y

############################################
# Part 2 Build the classification model
# the average log-loss on the three test sets should be lower than 0.45

#logistic regression
logit_predict = function(train, test.x){
  logit.fit = glm(loan_status ~ ., data = train, family = "binomial")
  predict(logit.fit, test.x, type="response")
}

#regression with lasso
lasso_predict = function(train, test.x){
  train.x = train[, colnames(train) != 'loan_status']
  train.x = model.matrix(~., train.x)[, -1]
  test.x = model.matrix(~., test.x)[, -1]
  train.y = train$loan_status
  cv.out = cv.glmnet(train.x, train.y, family="binomial", alpha = 1)
  predictions = predict(cv.out, s = cv.out$lambda.min, newx = test.x, type="response")
}

# RandomForest
randomforest_predict = function(train, test.x) {
  train$loan_status = as.factor(train$loan_status)
  randomforest.fit = randomForest(loan_status ~ ., data = train,
                          do.trace = TRUE, ntree = 500);
  
  predictions = predict(randomforest.fit, test.x, type="prob")[,2]
}

# Xgboost
xgboost_predict = function(train, test.x) {
  train.x = train[, colnames(train) != 'loan_status']
  train.x = model.matrix(~., train.x)[, -1]
  test.x = model.matrix(~., test.x)[, -1]
  train.y = train$loan_status
  
  xgboost.fit = xgboost(data = train.x, label=train.y,
                      objective = "binary:logistic", eval_metric = "logloss",
                      eta = 0.09,
                      nrounds = 300,
                      # colsample_bytree = 0.6,
                      # subsample = 0.75,
                      verbose = TRUE)

  predictions = predict(xgboost.fit, test.x, type="response")
}

# SVM 
svm_predict = function(train, test.x) {
  svm.fit = ksvm(loan_status ~ ., data = train, prob.model=TRUE)
  predictions = predict(svm.fit, test.x, type="probabilities")
  return(predictions)
}

##################################################
# calculate the log-loss
# log-loss function
logLoss = function(y, p){
  if (length(p) != length(y)){
    stop('Lengths of prediction and labels do not match.')
  }
  
  if (any(p < 0)){
    stop('Negative probability provided.')
  }
  
  p = pmax(pmin(p, 1 - 10^(-15)), 10^(-15))
  mean(ifelse(y == 1, -log(p), -log(1 - p)))
}

# test on three sets
# set1 0.4547164
logit_predictions = logit_predict(train_1,test_1.x)
logLoss(test_1.y, logit_predictions)
# set2 0.4556264
logit_predictions = logit_predict(train_2,test_2.x)
logLoss(test_2.y, logit_predictions)
# set3 0.4549093
logit_predictions = logit_predict(train_3,test_3.x)
logLoss(test_3.y, logit_predictions)
xgboost_predict

# lasso
lasso_predictions = lasso_predict(train_1,test_1.x)
logLoss(test_1.y, lasso_predictions)

# Xgboosting
# set1  0.4478636
xgb_predictions = xgboost_predict(train_1,test_1.x)
logLoss(test_1.y, xgb_predictions)
# set2  0.4496196
xgb_predictions = xgboost_predict(train_2,test_2.x)
logLoss(test_2.y, xgb_predictions)
# set3  0.4485444
xgb_predictions = xgboost_predict(train_3,test_3.x)
logLoss(test_3.y, xgb_predictions)
############################################
# Part 3 Build the final classifier 
# using all data from LoanStats_2007_to_2018Q2.csv
train = read.csv("loan_stat542.csv", stringsAsFactors = FALSE)





############################################
# Part 4 Model evaluation

test = read.csv("LoanStats_2018Q3.csv", stringsAsFactors = FALSE)
write.table(submission1, 'mysubmission1.txt', row.names = FALSE, 
            col.names = c("id", "prob"), sep = ", ")



