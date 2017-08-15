# What do the Heating and Cooling Loads of Residential Buildings depend on?

In this analysis, we study the influence of eight predictor variables, namely **Relative Compactness (RC), Surface Area (SA), Wall Area (WA), Roof Area (RA), Overall Height (OH), Orientation (O), Glazing Area (GA), and Glazing Area Distribution(GAD)** on Heating and Cooling Loads of 768 diverse residential buildings.

Models are trained on 80% of the available data and validated on the remaining 20% to avoid overfitting. 
The data preparation and model building approaches considered include:

* [Basic Linear Regression](#basic-linear-regression)
* [Data Preparation](#data-preparation)
  * [Principal Component Analysis](#principal-component-analysis)
* [Feature Extraction](#feature-extraction)
  * [Stepwise Linear Regression](#stepwise-linear-regression)
  * [Lasso Regression](#lasso-regression)
 * [Classification and Regression Trees](#classification-and-regression-trees)
 * [Random Forests](#random-forests)
 
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

Relative Compactness|Surface Area|Wall Area|Roof Area|Overall Height|Orientation|Glazing Area|Glazing Area Distribution| Heating Load|Cooling Load
--------------------|------------|---------|---------|--------------|-----------|------------|-----------|---------|-------
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

Based on the output, Relative Compactness, Surface Area, Wall Area, Overall Height, and Glazing Area are used to refine the model. The new model is then validated using the remaining 20% of the data. The figure below shows how the predicted heating loads (on the validation dataset) compare to the actual heating loads.

![HL](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/Basic%20Linear%20Regression_HL.PNG)

### Cooling Load

The same process of model building and refinement is used for the Cooling Load. The variables chosen are: Relative Compactness, Surface Area, Wall Area, Overall Height, and Glazing Area. The figure below compares the predicted cooling loads with the actual cooling loads of the validation dataset.

![CL](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/Basic%20Linear%20Regression_CL.PNG)

#### Model Statistics

Models | Multivariate Regression
-------|------------------|
Heating Load | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.01
Cooling Load | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20

Pretty Good!

## Crossvalidated Linear Regression

Next, we build the models for heating and cooling loads using cross validation. In the above given technique, we manually split the data into training and validation sets. In k-fold cross validation, we divide the entire dataset into k-folds, using k-1 folds for training and the remaining data for validation. Using this technique iteratively over all folds allows us to build a more effective model. Let's see if this leads to a better model quality

![CV Image](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/K-fold%20CV.PNG)

In this case, six folds are being used on the entire dataset to build the models. The code below illustrates the input for the heating load model. The variables of significance obtained from multivariate regression are used to build the crossvalidated model.

In [66]

```
Model_HL_CV <- lm( Heating_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Overall_Height+Glazing_Area, data = table)
CV_HL <- cv.lm(table, Model_HL_CV, m=6,plotit = FALSE)

```
The table below summarized the results obtained from cross validation and compares them against those from the multivariate linear regression. 

#### Model Statistics

Models | Multivariate Regression| CV Regression
-------|------------------|-----------------|
Heating Load | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.01 | RC, SA, WA, OH, GA, R2: 0.91, RMSE: 2.96 
Cooling Load | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.21 

The R-squared values obtained from the two techniques are pretty close to each other for both Heating as well as Cooling Loads. The Root Mean Squared Error for Heating Loads is found to be lower for CV Regression and is only slighttly higher for Cooling Loads, suggesting that Crossvalidated Regression is better than Basic Multivariate Regression. 

# Data Preparation

In situations where a large number of variables are present in a dataset and high collinearity is evident, we might benefit from a technique called Principal Component Analysis. 

## Principal Component Analysis

Principal Component Analysis is a technique used for dealing with high dimensional and correlated data. We start with the complete set of factors and generate Principal Components that explain most of the variation in the data, thereby reducing the dimensions of the variable space and retaining features that account for most of the variance in the dataset.

In [97]

```
pca <- prcomp(table[, 1:8], scale. = TRUE)
summary(pca)
screeplot(pca, type = "lines", col = "blue")
var <- pca$sdev^2
propvar <- var/sum(var)
plot(propvar, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")
plot(cumsum(propvar), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = "b")
```

Out [97

![PCA](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/PCA.PNG)

We can see here that most of the variation is explained by the first five PCs. Hence, we will build a model using these five variables.

In [105]

```
# Getting the first 5 principal components
PCs <- pca$x[,1:5]

install.packages("pls")
library(pls)

# Run principal component regression function with only the first 5 principal components

numcomp <- 5
pcr.fit_HL <- pcr(Heating_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Roof_Area+Overall_Height+Orientation+Glazing_Area+Glazing_Area_Distribution, data = train, scale = TRUE, ncomp = numcomp)
summary(pcr.fit_HL)

pcr.fit_HL$scores
coef(pcr.fit_HL)

```
Out [105]

```
Call:
Data: 	X dimension: 615 8 
	Y dimension: 615 1
Fit method: svdpc
Number of components considered: 5
TRAINING: % variance explained
              1 comps  2 comps  3 comps  4 comps  5 comps
X               46.28    62.08    77.39    89.85    99.30
Heating_Load    62.55    71.71    85.66    87.14    88.34

, , 5 comps

                          Heating_Load
Relative_Compactness        1.12012240
Surface_Area               -1.18970905
Wall_Area                   3.51099826
Roof_Area                  -2.86431696
Overall_Height              2.83596968
Orientation                -0.01130921
Glazing_Area                2.58315034
Glazing_Area_Distribution   0.38439689

```
The output above shows the percent variation explained by the first five principle components. We can see that together these explain 99.3% of variance in the training dataset. Lastly, the model based on PCA is transformed back to the original factor space to obtain the coefficients of the explanatory variables. 

#### Model Statistics

Models | Multivariate Regression | CV Regression | PCA
-------|------------------|-----------------|-----------------|-----------------------
Heating Load | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.01 | RC, SA, WA, OH, GA, R2: 0.91, RMSE: 2.96  | All, R2: 0.88, RMSE: 3.75  
Cooling Load | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.21  | All, R2: 0.83, RMSE: 4.01 

The model obtained as a result of Principal Component Analysis leads lower R-sqaured and higher RMSE values for the validation dataset. Therefore, while PCA is a useful technique for high dimensional data, we are probably better off without it.

# Feature Extraction

Now that we have settled on using our dataset as is, let us consider a few methods that can help build a focused model. We saw previously that we can select important variables in a model based on their associate p-values. While this technique worked for us (given the small number of variables), how can we tackle large number of variables? Feature Extraction!

The goal of feature extraction techniques is to identify and *extract pertinent variables* that are most important in building a model, without manual intervention. Moreover, using fewer variables leads to simpler models that not only are easier to understand, but also require less data collection. 
  
## Stepwise Linear Regression

Stepwise Linear Regression is a combination of forward selection and backward elimination approaches. At each step in the procedure, the model is evaluated and any variable that no longer seems necessary is removed. The code below illustrates the model built for heating loads.

```
In [152]

HL_step <- step(Model1_HL, direction = "both")
summary(HL_step)
residuals_HL_step <- validate$Heating_Load - predict(HL_step, validate)
HL_step_Rsqaured <- 1-sum(residuals_HL_step^2)/sum((validate$Heating_Load-mean(validate$Heating_Load))^2)
HL_step_Rsqaured
MSE_HL_step <- (sum((residuals_HL_step)^2))/nrow(validate)
RMSE_HL_step <- sqrt(MSE_HL_step)
RMSE_HL_step

```

#### Model Statistics

Models | Multivariate Regression | CV Regression | PCA | Stepwise Regression
-------|------------------|-----------------|------------------------
Heating Load | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.01 | RC, SA, WA, OH, GA, R2: 0.91, RMSE: 2.96 | All, R2: 0.88, RMSE: 3.75 | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.03
Cooling Load | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.21 | All, R2: 0.83, RMSE: 4.01 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20

For Heating Loads, the R-sqaured value obtained from Stepwise is slightly higher than that obtained from CV Regression,indicating a better fit. The RMSE value associated with Stepwise Regression, however, is seen to be higher than that of CV Regression. For Cooling Loads, the two sets of values are almost te same. 

Both, Stepwise and Crossvalidate Regression, seem to be good modeling techniques for this dataset.

Stepwise Regression is a type of **Greedy Algotrithm**. At each step, it does the one thing that looks best without taking future options into consideration. Next, we will look at a more refined method that is based on optimization models and considers all options before making a decision.

## Lasso Regression

Lasso Regression is one of the more advanced techniques of feature extraction. First, the model is layered on an optimization algorithm so that residual errors are ensured to be minimal overall. Secondly, constraints added within the Lasso Regression lead to only the most important variables being used in model construction. 

![Lasso](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/Lasso%20Regression.PNG)

Since we are constraining the sum of coefficients, we first need to scale the data.

In [173]

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
Out [173]

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

#### Model Statistics

Models | Multivariate Regression | CV Regression| Stepwise Regression| Lasso Regression
-------|------------------|-----------------|-----------------|----------
Heating Load | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.01 | RC, SA, WA, OH, GA, R2: 0.91, RMSE: 2.96 | All, R2: 0.88, RMSE: 3.75 | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.03 | WA, OH, GA, GAD, R2: 0.91, RMSE: 3.47
Cooling Load | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.21 | All, R2: 0.83, RMSE: 4.01 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | WA, OH, GA, R2: 0.88, RMSE: 3.54

Note here that number of variables chosen by Lasso is lower than those by other methods for both the Heating as well as the Cooling Loads. 

Since a fewer number of variables are chosen in the model, the RMSE values are seen to increase for Lasso Regression. This, however, is not necessarily a bad thing. If the risk associated with a slightly increased RMSE value is acceptable, it might be better to use the simpler model such as that chosen by Lasso Regression. 

Next, we explore a few advanced machine learning methods for model building and prediction. 

# Classification and Regression Trees

In all the methods discussed uptil now we used a single regression model on the entire training dataset. However, there might be scenarios when we can divide the dataset based on the explantory variables and build models specific to each subset. The CART method does just this - classifying the dataset into 'branches' and fitting regression model to each 'leaf' of the branches.It is based on recursive partitioning. We begin by growing the tree:

In [204]

```
install.packages("rpart")
library(rpart)
tree.train_HL.2 <- rpart(Heating_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Roof_Area+Overall_Height+Orientation+Glazing_Area+Glazing_Area_Distribution, data = train)  
summary(tree.train_HL.2)
plotcp(tree.train_HL.2)
plot(tree.train_HL.2)
text(tree.train_HL.2)
printcp(tree.train_HL.2)
```

Out [204]

```
Regression tree:
rpart(formula = Heating_Load ~ Relative_Compactness + Surface_Area + 
    Wall_Area + Roof_Area + Overall_Height + Orientation + Glazing_Area + 
    Glazing_Area_Distribution, data = train)

Variables actually used in tree construction:
[1] Glazing_Area         Overall_Height       Relative_Compactness

Root node error: 60546/615 = 98

n= 615 

    CP nsplit rel error xerror  xstd
1 0.79      0      1.00   1.00 0.036
2 0.08      1      0.21   0.21 0.015
3 0.03      2      0.13   0.13 0.010
4 0.01      3      0.10   0.10 0.007
5 0.01      4      0.09   0.09 0.006
6 0.01      5      0.07   0.08 0.006
7 0.01      6      0.06   0.07 0.004

```

The top branching variable is Overall Height.
![Tree](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/Tree.png)

Next, we prune the tree based on the variable with the lowest xerror and predict the validate dataset's loads based on the pruned tree. The idea is to prune the tree as the predictive ability of the model starts to decline. 

In [215]

```
prune.tree.train_HL.2 <- prune(tree.train_HL.2, cp = tree.train_HL.2$cptable[which.min(tree.train_HL.2$cptable[,"xerror"]),"CP"])
yhat.prune.tree.validate_HL <- predict(prune.tree.train_HL.2, newdata = validate)

plot(validate$Heating_Load, yhat.prune.tree.validate_HL, xlab = "Actual Heating Load", ylab = "Predicted Heating Load")
abline(0,1)

plot(validate$Heating_Load, scale(yhat.prune.tree.validate_HL - validate$Heating_Load), xlab = "Actual Heating Load", ylab = "Heating Load Error")
abline(0,0)

```

![CART_HL](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/CART_HL.png)

![CART_HL_Error](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/CART_HL_Error.png)

The same process can be applied to the Cooling Loads.

#### Model Statistics

Models | Multivariate Regression | CV Regression| Stepwise Regression| Lasso Regression| CART
-------|------------------|-----------------|-----------------|-----------------------|------|---------
Heating Load | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.01 | RC, SA, WA, OH, GA, R2: 0.91, RMSE: 2.96 | All, R2: 0.88, RMSE: 3.75 | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.03 | WA, OH, GA, GAD, R2: 0.91, RMSE: 3.47 | R2: 0.95, RMSE: 2.33
Cooling Load | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.21 | All, R2: 0.83, RMSE: 4.01 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | WA, OH, GA, R2: 0.88, RMSE: 3.54 | R2: 0.90, RMSE: 2.96

We can see how the **Classification and Regression Trees** has led to a sharp increase in the R-squared values and a decline in the RMSE values. This is because each regression model built within the tree is specific to a certain dataset. 

# Random Forests

We end this discussion with an extension of CART: The Random Forest method. In this method, we introduce randomness to help generate different trees with different strenghts and weaknesses. The idea is that overall, the average of all these trees is better than a single tree with a specific set of strengths and weaknesses. 

Random Forest gives better estimates overall, because while each tree might be over-fitting in one place or another, they don't necessarily over-fit the same way. So the average overall trees tends to flatten out those overreactions to random effects. 

Since the Random Forest method provides an average over the predicted response, we do not end up with a regression model. The importance of each variable is calculated and available for evaluation; however the method does not explain how the variables interact, or how a certain sequence of branches is significant. 

In [259]

```
install.packages("randomForest")
library(randomForest)
set.seed(1)

# Heating Load
numpred <- 3
rf.train_HL <- randomForest(Heating_Load~Relative_Compactness+Surface_Area+Wall_Area+Roof_Area+Overall_Height+Orientation+Glazing_Area+Glazing_Area_Distribution, data = train, mtry = numpred, importance = TRUE)
rf.train_HL
```

Out [259]

```
Call:
 randomForest(formula = Heating_Load ~ Relative_Compactness +      Surface_Area + Wall_Area + Roof_Area + Overall_Height + Orientation +      Glazing_Area + Glazing_Area_Distribution, data = train, mtry = numpred,      importance = TRUE) 
               Type of random forest: regression
                     Number of trees: 500
No. of variables tried at each split: 3

          Mean of squared residuals: 0.562
                    % Var explained: 99.4
```

500 Trees generated
99.4% Variance explained

In [268]

```
yhat.rf_HL <- predict(rf.train_HL, newdata = validate)
SSres_rf_HL <- sum((yhat.rf_HL-validate$Heating_Load)^2)
plot(validate$Heating_Load, yhat.rf_HL, xlab = "Actual Heating Load", ylab = "Predicted Heating Load")
abline(0,1)
```

![RF_HL](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/RF_HL.png)

The predicted values found via the Random Forest technique are found to be much closer to the actual values, as opposed to those found via other methods. 

![RF_HL_Error](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/RF_HL_Error.png)

The same process can be followed for Cooling Loads.

#### Model Statistics

Models | Multivariate Regression | CV Regression| Stepwise Regression| Lasso Regression| CART | RF
-------|------------------|-----------------|-----------------|-----------------------|------|---------|------------
Heating Load | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.01 | RC, SA, WA, OH, GA, R2: 0.91, RMSE: 2.96 | All, R2: 0.88, RMSE: 3.75 | RC, SA, WA, OH, GA, R2: 0.92, RMSE: 3.03 | WA, OH, GA, GAD, R2: 0.91, RMSE: 3.47 | R2: 0.95, RMSE: 2.33 | R2: 0.995, RMSE: 3.75
Cooling Load | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.21 | All, R2: 0.83, RMSE: 4.01 | RC, SA, WA, OH, GA, R2: 0.89, RMSE: 3.20 | WA, OH, GA, R2: 0.88, RMSE: 3.54 | R2: 0.90, RMSE: 2.96 | R2: 0.96, RMSE: 4.01

The Random Forest method leads to an even higher R-sqaured value! The RMSE values, however, are seen to increase as well and this is due to the randomness generated. 

Random Forest is a good quick solution and offers enhanced predictive capabilities. The downside, however, is that no single regression equation is obtained to communicate the model's workings with an audience. 

# Conclusion

Basic Multivariate Regression, Crossvalidated Regression, Stepwise Regression, and CART are well suited for this dataset given the simplicity of the models and prediction accuracy. 

Lasso Regression provides reasonable model quality and predictions, however it is not compare well against other simpler modeling techniques due to the restricted variable space.

Random Forest provides the best predictive capabilties. It is, however, not suited for this dataset due to its scale and simplicity.

###### References
1. Data: http://archive.ics.uci.edu/ml/datasets/Energy+efficiency
2. GTx: ISYE6501 Introduction to Analytics Modeling
