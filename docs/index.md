# What do the Heating and Cooling Loads of Residential Buildings depend on?

In this analysis, we study the influence of eight predictor variables, namely **Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, and Glazing Area Distribution** on Heating and Cooling Loads of 768 diverse residential buildings.

Models are trained on 80% of the available data and validated on the remaining 20% to avoid overfitting. 
The model building approaches considered include:

* [Linear Regression](linear-regression)
  * [Multivariation Linear Regression](multivariate-linear-regression)
  * [Crossvalidated Linear Regression](crossvalidated-linear-regression)
* [Feature Extraction](feature-extraction)
  * [Stepwise Linear Regression](stepwise-linear-regression)
  * [Principal Component Analysis](principal-component-analysis)
  * [Lasso Regression](lasso-regression)
 * [Classification and Regression Trees](classification-and-regression-trees)
 * [Random Forests](random-forests)
 
 Let's begin by previewing the data. 
 
In: 
```
install.packages("xlsx")
library(xlsx)
table <- read.xlsx("EE Residential Buildings.xlsx",1)
colnames(table) <- c("Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area", 
"Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution", "Heating_Load", "Cooling_Load")
View(table)
```

Out: 

Relative Compactness|Surface Area|Wall Area|Roof Area|Overall Height|Orientation|Glazing Area|Glazing Area Distribution
--------------------|------------|---------|---------|--------------|-----------|------------|-------------------------
0.98|514.5|294|110.25|7|2|0|0|15.55|21.33
0.98|514.5|294|110.25|7|3|0|0|15.55|21.33
0.98|514.5|294|110.25|7|4|0|0|15.55|21.33
0.98|514.5|294|110.25|7|5|0|0|15.55|21.33

# Linear Regression
## Multivariate Linear Regression
## Crossvalidated Linear Regression
Heating and Cooling Load Analysis of Residential Buildings using Multivariate Linear Regression
![Image of Model2](https://raw.githubusercontent.com/MeeraSharma/Residential-Energy-Efficiency-SLR.github.io/master/docs/Model2_HL.PNG)
# Feature Extraction
## Stepwise Linear Regressio
## Principal Component Analysis
## Lasso Regression
# Classification and Regression Trees
# Random Forests


