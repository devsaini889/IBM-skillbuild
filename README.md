# Employee-Salary-Prediction-with-Regression-Algorithm

We aim to solve the problem statement by creating a plan of action, Here are some of the necessary steps:

1. Data Exploration
2. Exploratory Data Analysis (EDA)
3. Data Pre-processing
4. Data Manipulation
5. Feature Selection/Extraction
6. Predictive Modelling
7. Project Outcomes & Conclusion


## What is regression analysis and what does it mean to perform a regression?
Regression analysis is a reliable method of identifying which variables have impact on a topic of interest. The process of performing a regression allows you to confidently determine which factors matter most, which factors can be ignored, and how these factors influence each other.

### In order to understand regression analysis fully, it’s essential to comprehend the following terms:

* Dependent Variable: This is the main factor that you’re trying to understand or predict. 
* Independent Variables: These are the factors that you hypothesize have an impact on your dependent variable.
In our application Independent Variable Age, PHD, Gender and dependent Variable is Salary.

In our Project we are going to use different regression analysis algorithms to get best possible model for predicting salries.
## 1. Multiple Linear Regression (MLR)

</br>The Coeffecient of the Regresion Model was found to be  [ 6.80241388 12.93840677 17.70862574]
</br>The Intercept of the Regresion Model was found to be  55.54625

</br>--------------------Training Set Metrics--------------------

</br>R2-Score on Training set ---> 0.4129947710728412
</br>Residual Sum of Squares (RSS) on Training set  ---> 83748.38755379422
</br>Mean Squared Error (MSE) on Training set       ---> 1046.8548444224277
</br>Root Mean Squared Error (RMSE) on Training set ---> 32.355136291204644

</br>--------------------Testing Set Metrics--------------------

</br>R2-Score on Testing set ---> 0.3423022373050564
</br>Residual Sum of Squares (RSS) on Training set  ---> 19832.95783522526
</br>Mean Squared Error (MSE) on Training set       ---> 991.6478917612632
</br>Root Mean Squared Error (RMSE) on Training set ---> 31.49044127606444

![alt text](https://github.com/dnyanshwalwadkar/Employee-Salary-Prediction-with-Regression-Algorithm/blob/main/Regression_Analasys_Images/LinearRegression.PNG)

## 2. Ridge Linear Regression (RLR)

</br>The Coeffecient of the Regresion Model was found to be  [ 6.80241388 12.93840677 17.70862574]
</br>The Intercept of the Regresion Model was found to be  55.54625

</br>--------------------Training Set Metrics--------------------

</br>R2-Score on Training set ---> 0.4129616938982482
</br>Residual Sum of Squares (RSS) on Training set  ---> 83753.10669410249
</br>Mean Squared Error (MSE) on Training set       ---> 1046.9138336762812
</br>Root Mean Squared Error (RMSE) on Training set ---> 32.35604786861772

</br>--------------------Testing Set Metrics--------------------

</br>R2-Score on Testing set ---> 0.34259043687681
</br>Residual Sum of Squares (RSS) on Training set  ---> 19824.267141294207
</br>Mean Squared Error (MSE) on Training set       ---> 991.2133570647104
</br>Root Mean Squared Error (RMSE) on Training set ---> 31.48354105028071

![alt text](https://github.com/dnyanshwalwadkar/Employee-Salary-Prediction-with-Regression-Algorithm/blob/main/Regression_Analasys_Images/RidgeRegression.PNG)

## 3. Lasso Linear Regression (LLR)

</br>The Coeffecient of the Regresion Model was found to be  [ 6.80241388 12.93840677 17.70862574]
</br>The Intercept of the Regresion Model was found to be  55.54625

</br>--------------------Training Set Metrics--------------------

</br>R2-Score on Training set ---> 0.41180399021628533
</br>Residual Sum of Squares (RSS) on Training set  ---> 83918.27697172793
</br>Mean Squared Error (MSE) on Training set       ---> 1048.9784621465992
</br>Root Mean Squared Error (RMSE) on Training set ---> 32.38793698503502

</br>--------------------Testing Set Metrics--------------------

</br>R2-Score on Testing set ---> 0.34718072866887206
</br>Residual Sum of Squares (RSS) on Training set  ---> 19685.846321387035
</br>Mean Squared Error (MSE) on Training set       ---> 984.2923160693517
</br>Root Mean Squared Error (RMSE) on Training set ---> 31.373433284697285

![alt text](https://github.com/dnyanshwalwadkar/Employee-Salary-Prediction-with-Regression-Algorithm/blob/main/Regression_Analasys_Images/LassoRegression.PNG)

## 4. Elastic-Net Regression (ENR)

</br>The Coeffecient of the Regresion Model was found to be  [ 6.80241388 12.93840677 17.70862574]
</br>The Intercept of the Regresion Model was found to be  55.54625

</br>--------------------Training Set Metrics--------------------

</br>R2-Score on Training set ---> 0.38009655896027433
</br>Residual Sum of Squares (RSS) on Training set  ---> 88441.99517781092
</br>Mean Squared Error (MSE) on Training set       ---> 1105.5249397226366
</br>Root Mean Squared Error (RMSE) on Training set ---> 33.249435179001715

</br>--------------------Testing Set Metrics--------------------

</br>R2-Score on Testing set ---> 0.31244411620388624
</br>Residual Sum of Squares (RSS) on Training set  ---> 20733.333190634858
</br>Mean Squared Error (MSE) on Training set       ---> 1036.6666595317429
</br>Root Mean Squared Error (RMSE) on Training set ---> 32.19730826531533

![alt text](https://github.com/dnyanshwalwadkar/Employee-Salary-Prediction-with-Regression-Algorithm/blob/main/Regression_Analasys_Images/ElasticNet_Regression.PNG)

## 5. Polynomial Regression (PNR)

</br>The Coeffecient of the Regresion Model was found to be  [ 6.80241388 12.93840677 17.70862574]
</br>The Intercept of the Regresion Model was found to be  55.54625

</br>--------------------Training Set Metrics--------------------

</br>R2-Score on Training set ---> 0.4658198996068942
</br>Residual Sum of Squares (RSS) on Training set  ---> 76211.79483019203
</br>Mean Squared Error (MSE) on Training set       ---> 952.6474353774005
</br>Root Mean Squared Error (RMSE) on Training set ---> 30.864987208443814

</br>--------------------Testing Set Metrics--------------------

</br>R2-Score on Testing set ---> 0.27350105503576017
</br>Residual Sum of Squares (RSS) on Training set  ---> 21907.66604370293
</br>Mean Squared Error (MSE) on Training set       ---> 1095.3833021851465
</br>Root Mean Squared Error (RMSE) on Training set ---> 33.096575384549176


![alt text](https://github.com/dnyanshwalwadkar/Employee-Salary-Prediction-with-Regression-Algorithm/blob/main/Regression_Analasys_Images/PolynomialFeatures.PNG)

# Regression Model Comparison

![alt text](https://github.com/dnyanshwalwadkar/Employee-Salary-Prediction-with-Regression-Algorithm/blob/main/Regression_Analasys_Images/Regression_model_comparison.PNG)

# R2 Score
![alt text](https://github.com/dnyanshwalwadkar/Employee-Salary-Prediction-with-Regression-Algorithm/blob/main/Regression_Analasys_Images/R2_Score.PNG)

# Correlation Matrix of Salary Data
![alt text](https://github.com/dnyanshwalwadkar/Employee-Salary-Prediction-with-Regression-Algorithm/blob/main/Regression_Analasys_Images/Correlation_Matrix.PNG)


