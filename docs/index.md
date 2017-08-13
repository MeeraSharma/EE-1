# What do the Heating and Cooling Loads of Residential Buildings depend on?

In this analysis, we study the influence of eight predictor variables, namely **Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, and Glazing Area Distribution** on Heating and Cooling Loads of 768 diverse residential buildings.

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

The same process of model building and refinement is used for the Cooling Load. The variables chosen are: Relative Compactness, Surface Area, Wall Area, Overall Height, and Glazing Area (R-squared = 0.88). The figure below compares the predicted cooling loads with the actual cooling loads of the validation dataset.

![CL](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency.github.io/master/docs/Basic%20Linear%20Regression_CL.PNG)
## Crossvalidated Linear Regression

# Feature Extraction
## Stepwise Linear Regressio
## Principal Component Analysis
## Lasso Regression
# Classification and Regression Trees
# Random Forests


