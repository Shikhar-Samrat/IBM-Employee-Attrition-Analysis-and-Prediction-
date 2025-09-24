# IBM Employee Attrition Prediction Using Machine Learning

## Introduction

Employee attrition refers to an employee’s voluntary or involuntary departure from a company. Organizations invest significant resources in hiring and training talented employees, making each one vital to the company’s success. Our project aimed to predict employee attrition and uncover the factors that contribute to employees leaving the workforce. We trained various classification models on our dataset and evaluated their performance using metrics such as accuracy, precision, recall, and F1 Score. Additionally, we conducted an in-depth analysis to pinpoint the key factors influencing employee turnover. Our findings will help organizations gain valuable insights into the drivers of attrition, ultimately aiding in efforts to improve employee retention.


#### Machine Learning Models

We trained and evaluated 8 supervised machine learning classification models.

1. Logistic Regression
2. Naive Bayes
3. Decision Tree
4. Random Forest
5. AdaBoost
6. Support Vector Machine
7. Multilayer Perceptron
8. K-Nearest Neighbors

#### Datasets
We trained our models on 6 different datasets
1. Imabalanced
2. Undersampled
3. Oversampled
4. PCA
5. Undersampling With PCA
6. Oversampling With PCA

To achieve the best performance, we carried out hyperparameter tuning using RandomSearchCV and GridSearchCV. We also performed 5-fold cross-validation on the training set to ensure robust model validation. To enhance model interpretability, we utilized various graphs and figures. Recognizing that accuracy alone can be a biased metric for predicting attrition, we evaluated our models using a comprehensive set of classification metrics: accuracy, precision, recall, and F1 Score.

## Dataset
We used the IBM HR Analytics Attrition Dataset from [Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset). It contains 35 columns and 1470 rows and has a mix of numerical and categorical features.

## Results

<img src="Results/Results.png" height="800">


The figure below shows feature importance w.r.t random forest with oversampling. We observe that the most important features were `MonthlyIncome` followed by `OverTime` and `Age`, while the least important features were `PerformanceRating`, `Gender` and `BusinessTravel`.


<img src="Results/FeatureImportance.png" height="500">

#### Best Performance
The best performance was obtained in the Random Forest Model
with PCA and Oversampling with an accuracy of `99.2%`,
the precision of `98.6%`, recall of `99.8%` and F1 Score of
`99.2%`.

## Libraries Used
1. [Numpy](https://numpy.org/)
2. [Pandas](https://pandas.pydata.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [Seaborn](https://seaborn.pydata.org/)
5. [Scikit-learn](https://scikit-learn.org/stable/index.html)



