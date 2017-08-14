# What do the Heating and Cooling Loads of Residential Buildings depend on?

In this analysis, we study the influence of eight predictor variables, namely **Relative Compactness (RC), Surface Area (SA), Wall Area (WA), Roof Area (RA), Overall Height (OH), Orientation (O), Glazing Area (GA), and Glazing Area Distribution(GAD)** on Heating and Cooling Loads of 768 diverse residential buildings.

Models are trained on 80% of the available data and validated on the remaining 20% to avoid overfitting. 
The model building approaches considered include:

* [Basic Linear Regression](basic-linear-regression)
* [Feature Extraction](feature-extraction)
  * [Stepwise Linear Regression](stepwise-linear-regression)
  * [Principal Component Analysis](principal-component-analysis)
  * [Lasso Regression](lasso-regression)
 * [Classification and Regression Trees](classification-and-regression-trees)
 * [Random Forests](random-forests)
 
 Let's begin by previewing the data. 
 
In [1]: 
```
install.packages("xlsx")
library(xlsx)
table <- read.xlsx("EE Residential Buildings.xlsx",1)
colnames(table) <- c("Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area", 
"Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution", "Heating_Load", "Cooling_Load")
View(table)
```

Out[1]: 

Relative Compactness|Surface Area|Wall Area|Roof Area|Overall Height|Orientation|Glazing Area|Glazing Area Distribution
--------------------|------------|---------|---------|--------------|-----------|------------|-------------------------
0.98|514.5|294|110.25|7|2|0|0|15.55|21.33
0.98|514.5|294|110.25|7|3|0|0|15.55|21.33
0.98|514.5|294|110.25|7|4|0|0|15.55|21.33
0.98|514.5|294|110.25|7|5|0|0|15.55|21.33

# Basic Linear Regression
## Multivariate Linear Regression
First, all explantory variables are used to construct the linear regression model. After exploring the importance of each variable, only those whose *p-value is less than 0.05* are chosen to build a refined model.
### Heating Load

In [16]

```
Model1_HL <- lm(Heating_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Roof_Area+Overall_Height+Orientation+Glazing_Area+
Glazing_Area_Distribution, data = train)
summary(Model1_HL)

```

Out [16]

```
Call:
lm(formula = Heating_Load ~ Relative_Compactness + Surface_Area + 
    Wall_Area + Roof_Area + Overall_Height + Orientation + Glazing_Area + 
    Glazing_Area_Distribution, data = train)

Residuals:
    Min      1Q  Median      3Q     Max 
-9.5460 -1.2746  0.0174  1.3329  8.0884 

Coefficients: (1 not defined because of singularities)
                            Estimate Std. Error t value Pr(>|t|)    
(Intercept)                84.211417  21.033488   4.004 7.01e-05 ***
Relative_Compactness      -63.457403  11.361624  -5.585 3.52e-08 ***
Surface_Area               -0.088772   0.018884  -4.701 3.21e-06 ***
Wall_Area                   0.063204   0.007385   8.558  < 2e-16 ***
Roof_Area                         NA         NA      NA       NA    
Overall_Height              3.953196   0.377263  10.479  < 2e-16 ***
Orientation                -0.016103   0.106863  -0.151  0.88027    
Glazing_Area               19.510954   0.908273  21.481  < 2e-16 ***
Glazing_Area_Distribution   0.244939   0.077523   3.160  0.00166 ** 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 2.919 on 607 degrees of freedom
Multiple R-squared:  0.9146,	Adjusted R-squared:  0.9136 
F-statistic: 928.5 on 7 and 607 DF,  p-value: < 2.2e-16
```

Based on the output, Relative Compactness, Surface Area, Wall Area, Overall Height, and Glazing Area are used to refine the model. The new model is then validated using the Validated dataset (R-squared = 0.92). The figure below shows how the predicted heating loads (on the validation dataset) compare to the actual heating loads.

Heating and Cooling Load Analysis of Residential Buildings using Multivariate Linear Regression
![HL](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/Basic%20Linear%20Regression_HL.PNG)

### Cooling Load

The same process of model building and refinement is used for the Cooling Load. The variables chosen are: Relative Compactness, Surface Area, Wall Area, Overall Height, and Glazing Area (R-squared = 0.89). The figure below compares the predicted cooling loads with the actual cooling loads of the validation dataset.

![CL](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/Basic%20Linear%20Regression_CL.PNG)

#### Summary

Models | Multivariate Regression-Variables, R2 
-------|------------------|
Heating Load | RC, SA, WA, OH, GA, 0.91
Cooling Load | RC, SA, WA, OH, GA, 0.89

Pretty Good!

## Crossvalidated Linear Regression

Next, we build the models for heating and cooling loads using cross validation. In the above technique, we manually split the data into training and validation sets. In k-fold cross validation, we divide the entire dataset into k-folds, using k-1 folds for training and the remaining data for validation. Using this technique iteratively over all folds allows us to build a more effective model. Let's see if this leads to a better model quality

![CV Image](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/K-fold%20CV.PNG)

In this case, six folds are being used on the entire dataset to build the models. The code below illustrates the input for the heating load model. The variables of significance obtained from multivariate regression are used to build the crossvalidated model.

In[66]

```
Model_HL_CV <- lm( Heating_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Overall_Height+Glazing_Area, data = table)
CV_HL <- cv.lm(table, Model_HL_CV, m=6,plotit = FALSE)

```
The table below summarized the results obtained from cross validation and compares them against those from the multivariate linear regression. 

#### Summary

Models | Multivariate Regression-Variables, R2 | CV Regression-Variables, R2 
-------|------------------|-----------------|
Heating Load | RC, SA, WA, OH, GA, 0.91 | RC, SA, WA, OH, GA, 0.91 
Cooling Load | RC, SA, WA, OH, GA, 0.89 | RC, SA, WA, OH, GA, 0.89 

The R-squared values obtained are the same as that obtained from multivariate regression. 0.91 and 0.89 are pretty good values and suggest that basic linear regression might be enough to understand the dependencies of Heating and Cooling Loads on the explanatory variables. However, to avoid running the risk of being too optimistic, we will use some state-of-the-art machine learning tools to build a robust model. 

# Feature Extraction

In cases where the number of variables is too large, it might not be possible to refine the models based on the p-values manually. In such scenarios, we might want to employ techniques that will allow us to *extract pertinent features* without manual intervention. The goal of feature extraction techniques is to identify variables that are most important in building a model. Using fewer variables leads to simpler models that not only are easier to understand, but also require less data collection. Secondly, when the number of variables gets too close to the number of data points, there is a risk of overfitting. 
  
## Stepwise Linear Regression

Stepwise Linear Regression is a combination of forward selection and backward elimination approaches. At each step in the procedure, the model is evaluated and any variable that no longer seems necessary is removed. The code below illustrates the model built for heating loads.

```
In [96]

HL_step <- step(Model1_HL, direction = "both")
summary(HL_step)
residuals_HL_step <- validate$Heating_Load - predict(HL_step, validate)
HL_step_Rsqaured <- 1-sum(residuals_HL_step^2)/sum((validate$Heating_Load-mean(validate$Heating_Load))^2)
HL_step_Rsqaured

```

#### Summary

Models | Multivariate Regression-Variables, R2 | CV Regression-Variables, R2 | Stepwise Regression-Variables, R2
-------|------------------|-----------------|------------------------
Heating Load | RC, SA, WA, OH, GA, 0.91 | RC, SA, WA, OH, GA, 0.91 | RC, SA, WA, OH, GA, 0.92
Cooling Load | RC, SA, WA, OH, GA, 0.89 | RC, SA, WA, OH, GA, 0.89 | RC, SA, WA, OH, GA, 0.89


Stepwise Regression is a type of **Greedy Algotrithm**. At each step, it does the one thing that looks best without taking future options into consideration. Next, we will look at a more refined method that is based on optimization models and considers all options before making a decision.

## Lasso Regression

![Lasso](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/Lasso%20Regression.PNG)

Optimization leads the model to choose coefficiencts of the regression equation such that the residual errors are minimal. Lasso Regression adds a constraint to the model such that the sum of the coefficients isn't too large. The restriction ensures that only the most important variables are chosen to build the model. 

Since we are constraining the sum of coefficients, we first need to scale the data.

In [111]

```
scaledTable <- as.data.frame(scale(table[,c(1,2,3,4,5,6,7,8)]))
scaledTable <- cbind(scaledtable, table[,c(9,10)]) # Add response variables back in
install.packages("glmnet")
library(glmnet)
lasso_HL <- cv.glmnet(x = as.matrix(scaledTable[, c(1,2,3,4,5,6,7,8)]), y = as.vector(scaledTable[ ,9]), alpha=1, nfolds = 10, type.measure = "mse", family = "gaussian")
coef(lasso_HL)
mod_lasso_HL <- lm(Heating_Load~Wall_Area+Overall_Height+Glazing_Area+Glazing_Area_Distribution, data = scaledTable)
summary(mod_lasso_HL)
```

Out [111]

```
Coefficients:
                          Estimate Std. Error t value Pr(>|t|)    
(Intercept)                 22.307      0.109  204.61   <2e-16 ***
Wall_Area                    2.254      0.114   19.83   <2e-16 ***
Overall_Height               8.341      0.114   73.38   <2e-16 ***
Glazing_Area                 2.655      0.112   23.78   <2e-16 ***
Glazing_Area_Distribution    0.316      0.112    2.83   0.0048 ** 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 3.02 on 763 degrees of freedom
Multiple R-squared:  0.911,	Adjusted R-squared:  0.91 
F-statistic: 1.95e+03 on 4 and 763 DF,  p-value: <2e-16
```

Similar approach can be applied for the Cooling Load. The results are summarized below:

#### Summary

Models | Multivariate Regression-Variables, R2 | CV Regression-Variables, R2 | Stepwise Regression-Variables, R2| Lasso Regression-Variables, R2
-------|------------------|-----------------|-----------------
Heating Load | RC, SA, WA, OH, GA, 0.91 | RC, SA, WA, OH, GA, 0.91 | RC, SA, WA, OH, GA, 0.92 | WA, OH, GA, GAD, 0.91
Cooling Load | RC, SA, WA, OH, GA, 0.89 | RC, SA, WA, OH, GA, 0.89 | RC, SA, WA, OH, GA, 0.89 | RC, WA, OH, GA, 0.88

We see here that Lasso Regression has chosen different explantory variables to meet the restriction. 

For Heating Load, the variables considered to be the most important are: Wall Area, Overall Height, Glazing Area, and Glazing Area Distribution. Note that the p-value for Glazing Area Distribution is 0.01

For Cooling Load, the most important variables are: Relative Compactness, Wall Area, Overall Height, and Glazing Area. All variables had a p-value of less than 0.001.

## Principal Component Analysis

Principal Component Analysis is a technique used for dealing with high dimensional and correlated data. We start with the complete set of factors and generate variables that, together, explain most of the variation in the data.

In [129]

```
pca <- prcomp(table[, 1:8], scale. = TRUE)
summary(pca)
screeplot(pca, type = "lines", col = "blue")
var <- pca$sdev^2
propvar <- var/sum(var)
plot(propvar, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")
plot(cumsum(propvar), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = "b")
```

Out [129]

![PCA](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/PCA.PNG)

We can see here that most of the variation is explained by the first five PCs. Hence, we will build a model using these five variables.

In [137]

```
# Getting the first 5 principal components
PCs <- pca$x[,1:5]

# Build linear regression model with the first 5 principal components
PC_HL <- cbind(PCs, table[,9])
PC_HL_model <- lm(V6~., data = as.data.frame(PC_HL))
summary(PC_HL_model)
Intercept_HL <- PC_HL_model$coefficiencts[1]
betas_HL <- PC_HL_model$coefficients[2:6]
betas_HL

alphas_HL <- pca$rotation[,2:6] %*% betas_HL
t(alphas_HL)
```

Out [137]

```
Call:
lm(formula = V6 ~ ., data = as.data.frame(PC_HL))

Residuals:
    Min      1Q  Median      3Q     Max 
-11.154  -2.146  -0.123   1.368  10.345 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  22.3072     0.1253  178.09   <2e-16 ***
PC1          -4.1696     0.0651  -64.02   <2e-16 ***
PC2          -3.8052     0.1126  -33.80   <2e-16 ***
PC3           2.1012     0.1138   18.46   <2e-16 ***
PC4           0.0261     0.1253    0.21     0.84    
PC5          -1.6542     0.1413  -11.71   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 3.47 on 762 degrees of freedom
Multiple R-squared:  0.882,	Adjusted R-squared:  0.882 
F-statistic: 1.14e+03 on 5 and 762 DF,  p-value: <2e-16

     Relative_Compactness Surface_Area Wall_Area Roof_Area Overall_Height Orientation Glazing_Area
[1,]               -0.201        0.856      4.21      -1.2         -0.429        -2.1        -2.71
     Glazing_Area_Distribution
[1,]                     -2.67

```
The output above shows the model in terms of the first five principle components. We can choose the varibles based on the p-values. Lastly, the model based on PCA is transformed back to the original factor space, providing an intuitive understanding.

#### Summary
Models | Multivariate Regression-Variables, R2 | CV Regression-Variables, R2 | Stepwise Regression-Variables, R2| Lasso Regression-Variables, R2|PCA-Variables, R2
-------|------------------|-----------------|-----------------|-----------------------|------
Heating Load | RC, SA, WA, OH, GA, 0.91 | RC, SA, WA, OH, GA, 0.91 | RC, SA, WA, OH, GA, 0.92 | WA, OH, GA, GAD, 0.91| RC, SA, WA, OH, 0.88
Cooling Load | RC, SA, WA, OH, GA, 0.89 | RC, SA, WA, OH, GA, 0.89 | RC, SA, WA, OH, GA, 0.89 | RC, WA, OH, GA, 0.88 | RC, SA, WA, OH, 0.84


# Classification and Regression Trees
# Random Forests


