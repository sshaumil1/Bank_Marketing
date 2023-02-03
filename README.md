# Bank Marketing | Customer Churn
###### 03.02.2023

## Name : Shaumil Sahariya

## Overview
The case study is based upon bank marketing campaigns. In which,  a Portuguese banking institution held marketing campaigns  based on phone calls in order to assess if the product (bank term deposit) would be subscribed or not. They made more than one contact to the same customer to get more precise information. 
## Goal
The classification goal is to predict if the client will subscribe to a term deposit (If not, means the client will churn).
## Problem Statement
A Portuguese bank has been observing a lot of customers not subscribing to bank term deposits and subscribing to competitor banks' term deposits over the past couple of quarters. As such, this has caused a huge dent in the quarterly revenues and might drastically affect annual revenues for the ongoing financial year. So, to arrest this problem, they want to build a classification model to predict whether the customer will subscribe to a term deposit or not.
## Hypothesis Generation
**H0:** The customer having low average annual balance can be considered as not subscribing to the bank term deposit.

**H1:** A customer who has not availed Housing/Personal Loan and has a low average annual balance can be treated as not subscribing to a bank term deposit.
## Data Gathering : Feature Set
A Portuguese banking institution held marketing campaigns  based on phone calls in order to assess if the product (bank term deposit) would be subscribed or not. They made more than one contact to the same client to get more precise information. On the basis of campaigns, they gathered the data and gave it to us in a csv file.

The data contains 4521 instances with the following attributes:

**Attribute information:**

age:  Age of the customer (continuous)

job: Job type (categorical: admin", "unknown", "unemployed", "management",  "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services")

marital: Marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)

education: Educational Background (categorical:  unknown",  "secondary", "primary",  "tertiary")

default: Has credit in default?  (binary: "yes","no")

balance: Average yearly balance, in euros (numeric)

housing: Has a housing loan?  (binary: "yes","no")

loan: Has a personal loan?  (binary: "yes","no")

contact: Contact communication type (categorical: "unknown", "telephone", "cellular")

day: Last contact day of the month (numeric)

month: lLast contact month of year (categorical: "jan",  "feb",  ..., "nov",  "dec")

duration: Last contact duration, in seconds (numeric)

campaign: Number of contacts performed during this campaign and for this client (numeric, includes last contact)

pdays: Number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)

previous: Number of contacts performed before this campaign and for this client (numeric)

poutcome: Outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")

**Output Variable:**

y: Has the client subscribed to a term deposit? (binary: "yes","no")

Apart from training files, we also require a "schema" file from the client, which contains all the relevant information about the training files such as Name of the files, Number of Columns, Name of the Columns, their data types, etc.
## Exploratory Data Analysis:
First I used the ".info()” function to get information about the features, in this I found the number of records was 4521. There were 7 features with numeric values and 10 features with categorical (object data type) values. 
Then I used the “.describe()” function to get all the stats of the numerical features. 
Next I used the ".isnull().sum()” function to get the count of the null values. No null values found. 
Then I used the “.value_counts()” over target feature. The data was highly imbalanced (class yes - 521 records & class no - 4000 records).
Next I wrote a function to get value counts of the categorical features and one function to get box plots as well as distribution plots of continuous features. Similarly, I plotted several plots in order to get insights from the data, including pairplot, catplot, etc. Also, I used pivot tables, group by function, etc.
## Feature Engineering:
First I made a copy of the data frame then dropped the 2 features (day and month) because the information of these features already exists in the “pdays” feature. After that, I replaced the “unknown” with “np.nan” as I observed during EDA that this category had no information. So, now I had null values in 4 features. Then, I checked percentages of null values in features. I dropped those features (“poutcome” & “contact”), which had high percentages of null values. Then, I filled in the null values of the rest of the 2 features by using the “.fillna()” function and used “method = ffill” (forward fill). 

Next I used the replace function in place of ordinal encoder to convert categorical features (having ordinal data) into numerical. And I used OneHotEncoder for the job column as it has nominal data. Lastly I used LabelEncoder for the output variable. 
## Feature Selection:
VIF: I checked if the input variables are independent of each other or not. And, I found that all the features are independent of each other (except “age").

fisher_score: Then I checked fisher score. I found all the features had good participation scores except the “marital” feature. 

VarianceThreshold: Gave “True” for all the features, which means all the features were good predictors. 

Chi2 Test: I conducted the ”chi2” test (between categorical features). The test told me that “marital” and “default” features are not good for the model. 

Anova Test: Lastly, I conducted the “Anova Test” (between continuous and categorical features). The result said that the “balance” feature was not good for the model. 

After using all the feature selection techniques, I found that the majority of the tests stated that the “merital” feature was not good for model building. That’s why I dropped it. After all these things, I had a final dataset with relevant features. 
## Balancing & Scaling:
Balancing: As I saw during EDA that the data was highly imbalanced. There are oversampling and undersampling methods to balance the data. But if I used undersampling, it would lead to loss of real information. Also, random oversampling led to duplicacy as the data was highly imbalanced. That’s why I used “SMOTE” to create new instances and add them into minority classes (class yes). I kept sampling strategy = 0.75 because it was not real information so I kept it as little as I could. 

Scaling: Before scaling I splitted the data for training as well as testing by using train_test_split() function. Because if I would pass the whole data to scaler, data leakage would happen. I used the standard scaler and fitted it to the training data set. And then I only transformed the test data set by using the same object of the standard scaler. 
## Model Selection | Tuning | Training:
Model Selection: After the data have been scaled, I used 5 models to see the accuracy of them by making a list of tuples filled with model name and model class. Then I initiated a loop and got the accuracy of the models. Lastly, I selected the XGBClassifier for my model by getting a good accuracy score.

Model Tuning : I tuned the model by using hyperparameter tuning. For that, I used RandomizedSearchCV. After tuning I got the best parameters to train the model. 

Model Training: After getting parameters, I trained the model with the best parameters. 
Feature Importance: After model training, I used feature importance function to get importance of the features. I found all the features were participating to make predictions. It showed that our feature selection techniques worked well.
## Model Evaluation:
I used different types of performance metrics to evaluate the model. 

Confusion Matrix: I got only 16 false negative values at the time of training and 61 false negatives (539 true positive) & 42 false positive (758 True negative) values at test data set.

Accuracy Score: The accuracy Score of the model is 93% on test data, which is quite good. We can obtain more than that by getting more data for training. 

f1_score: I got 94% f1 score, which tells that model is good.

Precision:  As we can see in this particular problem, precision is important so we have to give more importance on precision. Here I got precision 0.95, mens it is a good model. 

Recall: The recall value of the model is 0.93, which is quite good. 

Roc_Auc Curve: I plotted the roc_auc curve and got 0.978 Auc value, which is quite good.
## Exportation of Model for Production:
I exported the trained XGB model and Scaling model for production. Then I made a dictionary to export as a json file with features information.
## Model Testing:
First I gave the values as user input then made a list of features names. Then I built a test array filled with 0 by using np.zeros() where the length of the array was equal to the length of the list of the feature names. Then I filled the array with the given values as user input. After getting the array, I passed it to the scaling object by using the .transform() function in 2D format. Lastly, I got the prediction by passing the output of scaling into the model object by using  predict() function.
## Model Limitation:
The limitation of the model is that it has been trained on a small data set. So, it will not be accurate all the time.
## Possible Future Work:
We can retrain this model on a big dataset to get precise predictions in the near future. 
## Web Frameworks:
I wrote code for Flask Web Frameworks and tested the API with Postman. The test was successful. 
## Environment & Git:
Finally I made a virtual environment to use this code in further time. Also, push all code files to GitHub, including jupyter file, web frameworks files, readme file, etc.  
## Thank You
Shaumil Sahariya

DataSet Link:

https://www.kaggle.com/competitions/bank-marketing-uci/overview
