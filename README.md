# Lending-Club-Loan-Status-Analysis
## 01 Introduction
In this project, our data is from a historical loan dataset issued by Lending Club, but this dataset has over 100 features, and
some of them have too many NA values, and some are not supposed to be available at the beginning of the loan. Thus, based on 
our goals and needs, we use a cleaned dataset here to predict the chance of default/charged off for a loan, with 30 features in
total including the response 'loan_status' and a total of 844006 observations.

## 02 Methodology
### Logistic regression
Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Logistic regression
transforms its output using the logistic sigmoid function to return a probability value. We write a function using binomial to
apply the logistic regression, and predict the proportion for default/charged off status by using predict() function. All 
average log-loss for three datasets are larger than 0.45, though smaller than 0.46.

### regression with Lasso
Lasso regression uses the L1 penalty term and stands for Least Absolute Shrinkage and Selection Operator. In this model 
function, we first find the best penalty lambda, and build logistic regression model with lasso. Then we predict the fitted 
values for the test data. Later, we calculate the average log-loss. All average log-loss for three datasets are larger than 
0.45, but smaller than 0.46.

### Random Forest
Random forest is a very popular ensemble method that can be used to build predictive models for both classification and 
regression problems. Ensemble methods use multiple learning models to gain better predictive results. This model creates an 
entire forest of random uncorrelated decision trees to arrive at the best possible answer. When we build a function of a random
forest model of 500 trees, we predict the fitted values for the test data. Later, we calculate the average log-loss. All 
average log-loss for three datasets are larger than 0.45.

### Xgboosting
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It 
implements machine learning algorithms under the Gradient Boosting framework. We use max.depth = 6, iterations = 300, learning
rate = 0.09, subsampling =1 and loss function L2 regularization term. Also, objective = "binary:logistic" means we train a 
binary classification model. We predict the fitted values for the test data. Later, we calculate the average log-loss. All 
average log-loss for three datasets are less than 0.45. So this is the model we are going to use.

### SVM
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space that distinctly 
classifies the data points. We build a SVM model to train the data and predict the fitted value by probabilities type. Later,
we calculate the average log-loss. All average log-loss for three datasets are larger than 0.45.

