
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

set.seed(6404)

#############################################
# Part 1 preprocessing the data
# import the csv data
#data = read.csv("loan_stat542.csv", stringsAsFactors = FALSE)
data = read.csv("LoanStats_2007_to_2008Q2.csv", stringsAsFactors = FALSE)
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
data$fico_avg=(data$fico_range_low+data$fico_range_high)/2#the 31st column

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
# Xgboost
xgboost_predict = function(train, test.x, n=300) {
  train.x = train[, colnames(train) != 'loan_status']
  train.x = model.matrix(~., train.x)[, -1]
  test.x = model.matrix(~., test.x)[, -1]
  train.y = train$loan_status
  
  xgboost.fit = xgboost(data = train.x, label=train.y,
                        objective = "binary:logistic", 
                        eval_metric = "logloss",
                        nrounds = n,
                        verbose = TRUE,
                        eta = 0.09)

  predictions = predict(xgboost.fit, test.x, type="response")
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
## Xgboosting model
# set1  0.4474065
id = test.id[,1]
xgb_predictions1 = xgboost_predict(train_1,test_1.x)

submission1=cbind(id,xgb_predictions1)
write.table(submission1, 'mysubmission_test1.txt', row.names = FALSE, 
            col.names = c("id", "prob"), sep = ", ")

# set2  0.4488618
id = test.id[,2]
xgb_predictions2 = xgboost_predict(train_2,test_2.x)

submission2=cbind(id,xgb_predictions2)
write.table(submission2, 'mysubmission_test2.txt', row.names = FALSE, 
            col.names = c("id", "prob"), sep = ", ")

# set3  0.4477756
id = test.id[,3]
xgb_predictions3 = xgboost_predict(train_3,test_3.x)

submission3=cbind(id,xgb_predictions3)
write.table(submission3, 'mysubmission_test3.txt', row.names = FALSE, 
            col.names = c("id", "prob"), sep = ", ")

############################################
# Part 3 Model evaluation
##evaluation fro 2018 Q3
train = read.csv("LoanStats_2007_to_2008Q2.csv", stringsAsFactors = FALSE)
test_Q3 = read.csv("LoanStats_2018Q3.csv", stringsAsFactors = FALSE)
id = test_Q3$id
test_Q3 = test_Q3[,c('addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 
  'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 
                     'home_ownership', 'initial_list_status', 'installment', 'int_rate', 'id',
                     'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 
                     'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade',
                     'term', 'title', 'total_acc', 'verification_status', 'zip_code')]

test_Q3$term[which(test_Q3$term == ' 36 months')] = 36
test_Q3$term[which(test_Q3$term == ' 60 months')] = 60

#data processing for train/test data
current.id = which(test_Q3$loan_status == "Current")
dim(train) #844006     30
dim(test_Q3) #128194     30 
combi = rbind(train, test_Q3)

# binary response
combi$loan_status <- ifelse(combi$loan_status == c("Fully Paid","Current"), 0, 1)

# large income with log transform
combi$annual_inc=log(combi$annual_inc+1)
combi$revol_bal=log(combi$revol_bal+1)

# deal with the missing values
# fill NA with 'Others' category
combi$emp_length[is.na(combi$emp_length)] = "Others"

# fill NA with zero
combi$pub_rec_bankruptcies[is.na(combi$pub_rec_bankruptcies)] = 0

# fill NA with mean
combi$revol_util = gsub("%","",combi$revol_util)
combi$revol_util = as.numeric(combi$revol_util)
combi$int_rate = gsub("%","",combi$int_rate)
combi$int_rate = as.numeric(combi$int_rate)
combi$revol_util[is.na(combi$revol_util)] = mean(combi$revol_util[!is.na(combi$revol_util)])
combi$dti[is.na(combi$dti)] = mean(combi$dti[!is.na(combi$dti)])
combi$mort_acc[is.na(combi$mort_acc)] = mean(combi$mort_acc[!is.na(combi$mort_acc)])

# add or transfer new features
combi$fico_avg=(combi$fico_range_low+combi$fico_range_high)/2

d1=as.character(combi$earliest_cr_line)
d2=paste('1-',d1,sep = "")
d3=dmy(d2)
combi$earliest_cr_line = as.integer(round((as.Date("2018-1-1")-d3)/30))

combi$term[which(combi$term == '36 months')] = 36
combi$term[which(combi$term == '60 months')] = 60
combi$term = as.integer(combi$term)

# remove features
combi= combi %>%
  select(-c(id,grade,emp_title,purpose,title,zip_code,addr_state,application_type,fico_range_low,fico_range_high))

combi = group_features(combi)

train = combi[1:844006, ]
test_Q3 = combi[844007:972200, ]
train = as.data.frame(train)
test_Q3_x = test_Q3[,colnames(test_Q3) != 'loan_status']
test_Q3_x = as.data.frame(test_Q3_x)
test_Q3_y = test_Q3[,colnames(test_Q3) == 'loan_status']

#get fitted value from Q3
xgb_pred_Q3 = xgboost_predict(train,test_Q3_x)
submission_Q3=cbind(id,xgb_pred_Q3)
write.table(submission_Q3, 'mysubmission_2018Q3.txt', row.names = FALSE, 
            col.names = c("id", "prob"), sep = ", ")
test_Q3_y = test_Q3_y[-current.id]
xgb_pred_Q3 = xgb_pred_Q3[-current.id]
logLoss(test_Q3_y, xgb_pred_Q3) #0.6487541



##evaluation fro 2018 Q4
train = read.csv("LoanStats_2007_to_2008Q2.csv", stringsAsFactors = FALSE)
test_Q4 = read.csv("LoanStats_2018Q4.csv", stringsAsFactors = FALSE)
id = test_Q4$id
test_Q4 = test_Q4[,c('addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 
                     'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 
                     'home_ownership', 'initial_list_status', 'installment', 'int_rate', 'id',
                     'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 
                     'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade',
                     'term', 'title', 'total_acc', 'verification_status', 'zip_code')]

test_Q4$term[which(test_Q4$term == ' 36 months')] = 36
test_Q4$term[which(test_Q4$term == ' 60 months')] = 60

#data processing for train/test data
current.id = which(test_Q4$loan_status == "Current")
dim(train) #844006     30
dim(test_Q4) #128412    30 
combi = rbind(train, test_Q4)

# binary response
combi$loan_status <- ifelse(combi$loan_status == c("Fully Paid","Current"), 0, 1)

# large income with log transform
combi$annual_inc=log(combi$annual_inc+1)
combi$revol_bal=log(combi$revol_bal+1)

# deal with the missing values
# fill NA with 'Others' category
combi$emp_length[is.na(combi$emp_length)] = "Others"

# fill NA with zero
combi$pub_rec_bankruptcies[is.na(combi$pub_rec_bankruptcies)] = 0

# fill NA with mean
combi$revol_util = gsub("%","",combi$revol_util)
combi$revol_util = as.numeric(combi$revol_util)
combi$int_rate = gsub("%","",combi$int_rate)
combi$int_rate = as.numeric(combi$int_rate)
combi$revol_util[is.na(combi$revol_util)] = mean(combi$revol_util[!is.na(combi$revol_util)])
combi$dti[is.na(combi$dti)] = mean(combi$dti[!is.na(combi$dti)])
combi$mort_acc[is.na(combi$mort_acc)] = mean(combi$mort_acc[!is.na(combi$mort_acc)])

# add or transfer new features
combi$fico_avg=(combi$fico_range_low+combi$fico_range_high)/2

d1=as.character(combi$earliest_cr_line)
d2=paste('1-',d1,sep = "")
d3=dmy(d2)
combi$earliest_cr_line = as.integer(round((as.Date("2018-1-1")-d3)/30))

combi$term[which(combi$term == '36 months')] = 36
combi$term[which(combi$term == '60 months')] = 60
combi$term = as.integer(combi$term)

# remove features
combi= combi %>%
  select(-c(id,grade,emp_title,purpose,title,zip_code,addr_state,application_type,fico_range_low,fico_range_high))

combi = group_features(combi)
dim(combi)
train = combi[1:844006, ]
test_Q4 = combi[844007:972418, ]
train = as.data.frame(train)
test_Q4_x = test_Q4[,colnames(test_Q4) != 'loan_status']
test_Q4_x = as.data.frame(test_Q4_x)
test_Q4_y = test_Q4[,colnames(test_Q4) == 'loan_status']

#get fitted value from Q3
xgb_pred_Q4 = xgboost_predict(train,test_Q4_x)
submission_Q4=cbind(id,xgb_pred_Q4)
write.table(submission_Q4, 'mysubmission_2018Q4.txt', row.names = FALSE, 
            col.names = c("id", "prob"), sep = ", ")
test_Q4_y = test_Q4_y[-current.id]
xgb_pred_Q4 = xgb_pred_Q4[-current.id]
logLoss(test_Q4_y, xgb_pred_Q4) # 0.6576236






