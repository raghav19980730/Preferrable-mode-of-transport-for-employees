# Preferrable-mode-of-transport-for-employees

## Project Objectives
The main objective of this project is to build a model to identify the mode of transportation used by the employees of the company. This is further divided into small objectives which are as follows:
1)	To perform exploratory data analysis.
2)	To check the presence of multicollinearity and deal with it.
3)	To build logistic regression model, k – nearest neighbour model and Naïve Bayes model.
4)	To build a model using ensemble techniques like bagging and boosting.
5)	Selecting the best model by comparing different model performance measures like accuracy, KS stat, gini coefficient, concordance ratio, etc.

## Exploratory Data Analysis
### Environment set up and Data Import
Setting up of the working directory help in accessing the dataset easily. Different packages like “tidyverse”, “car”, “InformationValue”, “bloom”, “caTools”, “caret”, “dplyr”, “gini”, “ROCR”, “gridExtra”, “glm”, “DMwr”, “rpart” and “corrplot” are installed to make the analysis process slightly easier. <br/>
The data file is in “.csv” format. To import the dataset, read.csv function is used. The dataset consists of 444 rows and 9 columns.


### Variable Identification
“head” function is used to show top six rows of the dataset. <br/>
“tail” function is used to show last six rows of the dataset.<br/>
“str” function is used to identify type of the variables. In this dataset, all the variables are numeric. Initially all the variables are numerical. Some of the variables like Churn, DataPlan and ContractRenewal are converted into factors using as.factor() function.<br/>
“summary” function is use to show the descriptive statistics of all the variables.<br/>


#### Descriptive Summary
|Descriptive Stats|	Age (in years)|	Work Experience (in years)|	Salary  (in lakhs)|	Distance (in kms)|
|-----|-----|-----|----|------|
|Min|	18	|0|	6.5|	3.2|
|Q1|	25	|3|	9.8	|8.8|
|Median|	27|	5|	13.6|	11|
|Mean|	27	|6	|15.27	|11.30|
|Q3|	30	|8|	15.75|	13.45|
|Max|	38	|16	|34	|20|


The historical data reflects that around 67.56% employees use Public Transport, 18.69% employees use 2wheeler and only 13.74% employees use cars.

## Univariate Analysis
### A)	Histograms
The histogram is used to signify the normality and skewness of the data. It also signifies central location, shape and size of the data.
![image](https://user-images.githubusercontent.com/61781289/145580102-fb208e6a-02aa-476c-b1ba-9186cd5a9b7a.png)

From above figure we can infer that:
1.	Variables like Age and Distance are nearly normally distributed.
2.	Variables like Salary and Work Experience shows sign of right skewness.

### B)	Barplots
![image](https://user-images.githubusercontent.com/61781289/145580152-886bbc3b-62b3-46c6-86ab-de8dc4263e13.png)

1)	Out of total male employees, 223 uses Public Transport, 48 uses cars and 45 uses 2Wheeler. Out of total female employees, 77 uses Public Transport, 13 uses cars and 38 uses 2Wheeler.
2)	Out of total employees who holds MBA degree, 83 uses Public Transport, 12 uses cars and 17 uses 2Wheeler.
3)	Out of total employees who are Engineers’, 223 uses Public Transport, 52 uses cars and 60 uses 2Wheeler.
4)	Out of total employees who has license, 33 uses Public Transport, 48 uses cars and 23 uses 2Wheeler.

## Bivariate Analysis
### Boxplot
![image](https://user-images.githubusercontent.com/61781289/145580229-2e616bec-66e0-43ba-af3c-343ab8e535b3.png)


### Outlier detection and missing values
There is only 1 missing value in the given dataset. The value is removed from the dataset. <br/>
To check the presence of the outliers in the dataset, cook’s distance is used. Cook’s distance shows how removing a particular observation from the data can affect the predicted values.<br/>
The general rule of thumb suggests than any observation above the threshold i.e. 4* mean of cook’s coefficient D   is considered as an influential value or outlier.

![image](https://user-images.githubusercontent.com/61781289/145580284-c4f723d6-cc8d-41da-b76f-4a259d9109d8.png)


It is clearly seen that all the values above red line are outliers and need to be treated accordingly.



### Outliers Treatment
Mean imputation method is used to treat the outliers. The table below shows variables having outliers, how many outliers are present and capping value used to replace the outliers.

|Variables 	Number of outliers|	Capping Value|
|-----|-----|
|Age	25|	38|
|Work Experience|	38	|16|
|Salary|	59|	34|
|Distance	|9	|20|



### Multicollinearity
The problem of multicollinearity exists when the independent variables are highly correlated with each other.  Variance Inflation Factor (VIF) and Tolerance level are the key factors to analyse the level of the multicollinearity among independent variables.  In VIF, any value closer to 1 signifies low level of correlation and any value above 10 signifies high level of correlation.<br/>
As per the given dataset, Working Experience has VIF value greater than 5. Thus, they depict problem of multicollinearity and need to be treated.

![image](https://user-images.githubusercontent.com/61781289/145580476-ebb44cd3-efd7-4ab3-b8b8-efcefb509964.png)


It can be seen in the above figure that: 
1.	There exists high correlation (0.92) between Age and Work Experience.
2.	Work Experience and Salary also shows signs of high correlation (0.92).
3.	Age and Salary shows signs of high correlation (0.85).


### Handling of Imbalanced dataset 
A dataset is said to be imbalanced when number of observations per class is not equally distributed or data is skewed towards one class. Due to this unbalancing of the data, certain algorithm will only emphasize on the majority class and not on minority class. <br/>
In other words, many machine learning models are subject to a frequency bias in which they place more emphasis on learning from data observations which occur more commonly. Even though the model failed to predict minority class correctly, still the accuracy measure will be high for the model due to the presence of frequency bias.<br/>
So, accuracy should not be used to check the performance of model. Instead, we use precision and recall or F1-score to measure the performance of the model. <br/>
There are many ways to handle the imbalanced data but we will focus only on Smote analysis.<br/>

### SMOTE (Synthetic Minority Oversampling Technique)
This technique solves the problem of imbalanced dataset by generating synthetic samples for the minority class by interpolating the observations from original dataset. <br/>
The data reflects that around 86.26% employees do not use car and only 13.74% employees use cars. 

![image](https://user-images.githubusercontent.com/61781289/145580646-831dab23-366b-45f4-8009-a3cb7af4bf6c.png)

The first graph shows the skewed dataset where red dot reflects majority class and blue dots minority class. <br/>
In second figure, after applying the smote analysis, the minority class was populated with synthetic samples and is increased to 40% and majority class reduces to 60%. Formula used under R: - <br/>
SMOTE(Transport~., data = smote.train, perc.over = 3000, k=5,perc.under = 150) <br/>


## Logistic Regression 

Logistic Regression is a type of generalized linear model which is used to solve binary classification problem. Logistic regression uses maximum likelihood method to obtain a best fit line. It uses logit function to estimate the coefficients. The function is given by <br/>
log(odds) = beta0 + beta1*X1 +beta2*X2 + ……… + beta(n)* Xn <br/>
where log(odds) = log (probability of occurrence of an event/prob of non-occurrence of an event) = log(p/1-p)

### Assumptions:
1.	**Linear Relationship**: There should exist a linear relationship between log(odds) and regressors. It can be seen in the figure that most of the variables depicts this linear relationship.

![image](https://user-images.githubusercontent.com/61781289/145580791-ad01745e-eda6-404f-a99b-acee5c44dd53.png)

2.	**Outliers Treatment**: The dataset should be free from outliers. Using cook’s distance, it was found that there is presence of outliers in the model and capping method is used to handle these outliers.
3.	**No Multicollinearity**: The independent variables should not be highly correlated with each other. Using VIF, it was found that variables like Working Experience shows sign of high correlation with other variables. Thus, it was dropped from the model.


### Model Building 
The data has been split randomly into training set and test set with a split ratio of 2/3. K-fold validation technique was used to build the model on training data set and the model which best fits the training set (i.e. model with highest accuracy) was selected. Then the best fit model is used to predict the test dataset. 

#### Model 1

|Variables	|Estimate	|z-value	|p-value	|Status|
|-----|-----|-----|-----|-----|
|Intercept	|-41.00919|	-19.812	|< 2e-16	|Significant|
|Age	|1.14658	|18.392	|< 2e-16	|Significant|
|Salary	|-0.03866	|-2.668	|0.00763	|Significant|
|Distance|	0.40848	|11.446	|< 2e-16	|Significant|
|Gender1	|0.74455|	3.908	|9.31e-05	|Significant|
|MBA1	|-1.17137	|-5.883	|4.04e-09|	Significant|
|License1	|1.75341	|9.493	|< 2e-16	|Significant|

**Car = -41.01 + 1.15(Age) – 0.04(Salary) + 0.41(Distance) +0.74(Gender1) – 1.17(MBA1) + 01.65(license1)**

![image](https://user-images.githubusercontent.com/61781289/145581165-7ea1e393-69b2-483b-95d3-0a07e185d9bb.png)


From the analysis, we can infer that variables like Engineer doesn’t has any significant effect on whether an employee will use car or not. Age has the highest effect on the dependent variable followed by Distance, license, MBA1, Gender1 and Salary.


## K- Nearest Neighbour
KNN algorithm is a supervised machine learning model which is used for both regression and classification problems. K- Nearest Neighbour is estimated using the concept of the shortest distance between the observation. There are majorly 3 ways to calculate the distance: Euclidian Distance, Manhattan Distance and Minkowski Distance.

### Feature Scaling
Since we are calculating the Euclidian distance between different variables, thus standardization or scaling becomes a necessary part in data pre-processing. The formula used to standardize the variables is given by  <br/>
Standardized value = (Old value – Mean(n)) / Standard Deviation(n) <br/>
Where n is total number of observations 

### Hyperparameter Tuning
The data has been split randomly into training set and test set with a split ratio of 75/100. K-fold validation technique was used to build the model on training data set and the model which best fits the training set (i.e. model with highest accuracy) was selected. Then the best fit model is used to predict the test dataset. 

**Selection of K-Value**
The optimal number of K nearest neighbour is decided using a hyperparameter tuning method called grid search. Initially, a grid of 9 K-values is selected and “accuracy” metric is used to assess the performance of each model. It is found that accuracy is highest when the optimal value for K- nearest neighbour is 3.

![image](https://user-images.githubusercontent.com/61781289/145581295-dacf09d8-f0b6-4b53-bb8b-daf61c9ce7d4.png)

It can be seen in the above figure that the most important variable influencing whether a employee will use a car as means of transport or not is Age, followed by Salary, Work.Exp and so on. Engineer is the least important variable in the model.


## Naïve Bayes
It is based on the concept of Bayes Theorem but it follows a naïve assumption that all the variables are independent of each other. This method is used for both classification and regression problems. It is relatively faster as compared to other models.

### Model Building
The data has been split randomly into training set and test set with a split ratio of 0.75. K-fold validation technique was used to build the model on training data set and the model which best fits the training set (i.e. model with highest accuracy) was selected. Then the best fit model is used to predict the test dataset. 

### Hyperparameter Tuning
The grid search method is used to tune the hyperparameter which are given under:
1)	**Laplace Smoothing** – This method is generally used to smooth the categorical variable. It is used when conditional probability for a case is zero. Values taken for tuning are from 0 to 5.
2)	**Use of kernel**: - This parameter allows us to use a kernel density estimate for a gaussian density estimate vs a continuous variable. It takes only two values – “True” and “False”
3)	**Bandwidth of kernel Density**: - “adjust” parameter allows us to adjust the bandwidth of the kernel density. Values taken for tuning are 1 to 5
The final value used for each parameter is 0 for Laplace smoothing, 1 for adjust and TRUE for usekernel.


![image](https://user-images.githubusercontent.com/61781289/145581452-4073510b-17ee-4894-bcb6-7ce2d6a2d0f9.png)


It can be seen in the above figure that the most important variable influencing whether a employee will use a car as means of transport or not is Age, followed by Work.Exp, Salary and so on. MBA is the least important variable in the model.

### Ensemble Learning
Ensemble learning refers to combining different algorithms into one predictive model to reduce bias and variance in the model. There are main four types of ensembling techniques: - Bagging, Boosting, Stacking and Blending.

## Bagging (Bootstrap Aggregation)
In this technique, different samples are drawn from the dataset using sample with replacement technique. Models are prepared on each sample dataset and evaluated accordingly. Key information in bagging technique is given as under:
1.	Number of rows drawn in each sample should be less than the number of rows of main dataset.
2.	Rows can be repeated several times in a sample.
3.	In case of regression task, average of all models’ measure and in case of classification task, mode of all models’ measure is chosen as the final answer.

<br/>The bagging algorithm uses recursive binary splitting technique to choose a variable which leads best split at each step. The “best split variable” can be chosen using different metrics like Gini Impurity, Entropy, Variance, etc. We will emphasis on the Gini impurity metrics. <br/>

#### Gini Impurity
It is a measure which show that how many randomly chosen data points if labelled randomly are incorrectly labelled. The formula used to calculate the Gini impurity is given by: <br/>
Gini impurity = 1 – sum[(pi)^2]   where pi is the probability of item with label i to be chosen. <br/>

### Hyperparameter Tuning
The data has been split randomly into training set and test set with a split ratio of 0.75. K-fold validation technique was used to build the model on training data set and the model which best fits the training set (i.e. model with highest accuracy) was selected. Then the best fit model is used to predict the test dataset.  <br/>

The grid search method is used to tune the hyperparameter which are given under:
1)	maxdepth: - This parameter signifies max depth that a tree is allowed to grow. The final value selected is 5.
2)	minsplit: - It refers to the number of minimum observations required at terminal node to make the next split. The final value selected is 4.

![image](https://user-images.githubusercontent.com/61781289/145581672-36dd312f-ea03-4d97-b4d3-4965df34a2a7.png)


It can be seen in the above figure that the most important variable influencing whether a employee will use a car as means of transport or not is Age, followed by Salary, Work.Exp, and so on. Engine is the least important variable in the model.


## Boosting
Boosting is type of ensemble technique which tries to reduce both bias and variance by building large number of weak learners/models where each successive weak learner learns from the mistakes of predecessor learners. It can be used for both classification and regression problems. <br/>
Gradient boosting method is used to train the model. Gradient boosting uses the concept of Gini Impurity(to make a split) and Pseudo Residuals(= (original – predicted)^2) to come up with the final answer for a tree. In case of regression task, average of all models’ measure and in case of classification task, mode of all models’ measure is chosen as the final answer.

### Hyperparameter Tuning
The data has been split randomly into training set and test set with a split ratio of 0.75. K-fold validation technique was used to build the model on training data set and the model which best fits the training set (i.e. model with highest accuracy) was selected. Then the best fit model is used to predict the test dataset. <br/>
The grid search method is used to tune the hyperparameter which are given under:
1)	n.tree: –This parameter signifies the total number of trees to be drawn in the GBM algorithm. The final optimal value selected is 950.
2)	Interaction.depth: - This parameter signifies max depth that a tree is allowed to grow. The final value selected is 5.
3)	shrinkage: - It is known as learning rate which is applied to each tree in expansion. The final value selected is 0.01.
4)	n.minobsinnode: - It refers to the number of minimum observations required at terminal node to make the next split. The final value selected is 20.


![image](https://user-images.githubusercontent.com/61781289/145581764-b1d5d10e-805a-4799-abe9-0d1fbb21fa1f.png)


It can be seen in the above figure that the most important variable influencing whether an employee will use a car as means of transport or not is Age, followed by Salary, Distance and so on. Gender1 is the least important variable in the model.

## Model Performance
Once the model is prepared on the training dataset, next step to is to measure the performance of the model on test dataset. Since the models predicts the test values in the form of probability, a threshold is selected to convert it to either 0 or 1. In these models, 0.5 is selected as threshold. Any probability less than 0.5 will be shifted to 0 and any probability above 0.5 will be shifted to 1. <br/>

Different key performance measures are used to check the efficiency and effectiveness of the model.
1)	Accuracy: - It is the ratio of total number of correct predictions to total number of samples. <br/>
Accuracy = (True Positive + False Negative)/ (True Positive + False Negative + True Negative + False Positive)
2)	Classification Error: - It is the ratio of total number of incorrect predictions to total number of samples. <br/>
Classification Error = (False Positive + True Negative)/ (True Positive + False Negative + True Negative + False Positive)
3)	Sensitivity: - It is the proportion of customers who didn’t cancel the post-paid services got predicted correctly to total number of customers who didn’t cancel the services. <br/>
Sensitivity = True Positive/ (True Positive + False Negative)
4)	Specificity: - It is the proportion of customers who cancelled the post-paid services got correctly predicted to total number of customers who cancelled the post-paid services. <br/>
Sensitivity = True Negative/ (True Negative + False Positive)
5)	Concordance Ratio: - It is the ratio of concordant pairs to the total number of pairs. After making all the pairs of alternative classes, the pairs in which the predicted probability for class 1 is higher than the predicted probability for class 0 is considered as concordant pairs. Higher the concordance ratio, better the model. <br/>
Concordance Ratio = Number of concordant pairs / Total number of pairs
6)	KS stat: - The Kolmogorov–Smirnov statistic quantifies a distance between the empirical distribution function of the sample and the cumulative distribution function of the reference distribution, or between the empirical distribution functions of two samples.
7)	F1 -Score = 2*(Precision + Recall)/(Precision + Recall)
8)	Area Under Curve/ Receiver Operating Characteristics (AUC/ROC): - It signifies the degree of correct predictions made by the model. Higher the AUC, better the model. 
9)	Gini Coefficient: - It is the ratio of area between ROC curve and random line to the area between the perfect ROC model and random model line. Higher the Gini coefficient, better the model. <br/>
Gini = 2AUC - 1

![image](https://user-images.githubusercontent.com/61781289/145581928-c105b857-92d3-421c-a82a-59b5a876e868.png)


The above figure shows Receiver Operating Characteristics of all the models. We can use ROC curve to derive the Gini coefficient and Area under curve (AUC). <br/>
Gini coefficient = B/B+C = B/0.5 <br/>
AUC = A + B = 0.5 + B  <br/>
Thus, using both equation we get, Gini = 2AUC – 1 <br/>

The results drawn from performance measure are as follows:

|Performance Measures|	KNN	|Logistic	|Naïve| Bayes|	Bagging|	Boosting|
|----|-----|----|----|----|----|----|
|Accuracy   | 	0.97297297|	0.990990991|	0.990990991|	0.98198198|	0.97297297|
|Classification error|	0.02702703	|0.009009009	|0.009009009	|0.01801802	|0.02702703|
|Sensitivity	|0.96969697	|0.989690722|	0.989690722	|0.98958333|	0.96969697|
|Specificity|	1.00000000	|1.000000000|	1.000000000|	0.93333333|	1.00000000|
|KS stat|              	0.80000000	|0.933333333|	0.933333333	|0.92291667	|0.80000000|
|AUC  |	0.90000000	|0.966666667|	0.966666667	|0.96145833|	0.90000000|
|Gini  | 	0.89189189|	0.841710278	|0.861967934	|0.87100737|	0.88542757|
|Concordance|	0.80000000|	0.989583333|	0.988888889|	0.93263889|	0.99791667|
|F1-Score|	0.98461538	|0.994818653	|0.994818653|	0.98958333|	0.98461538|




We can draw following conclusion from the above table: 
1)	KNN is the worst model as it performed poorly on all the measure as compared to other models.
2)	In case of imbalanced dataset, main emphasis must be laid on measures like AUC and F1 – score. It can be clearly seen that Logistic and Naïve Bayes algorithm performed equally great on both measures.
3)	 Bagging is the third best model followed by gradient boosting and KNN.

## Conclusion
1.	There was presence of outliers and missing values in the model. The missing values are removed from the model and capping method is used to handle the outliers. 
2.	The data reflects that around 67.56% employees use Public Transport, 18.69% employees use 2wheeler and 13.74% only employees use cars. SMOTE technique is used to handle this imbalanced dataset.
3.	Emphasis is laid on measures like F1-score and AUC (Area Under Curve) to determine the most optimal model. The Logistic and Naïve Bayes were the most optimal models.
4.	From all the above models, we can infer that the decision whether an employee uses car for travelling or not is mostly influenced by variables like Age, Salary, Working Experience, license and Distance.
It is quite intuitive that an adult employee having higher working experience, higher salary and license is more likely to use a car than low income employee. Also, higher the distance between home and office more likely it is to use a car to travel. 


