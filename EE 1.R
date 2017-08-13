install.packages("xlsx")
library(xlsx)
table <- read.xlsx("EE Residential Buildings.xlsx",1)
colnames(table) <- c("Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area", "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution", "Heating_Load", "Cooling_Load")
View(table)
indexes <- sample(1:nrow(table), size = 0.2*nrow(table))
#Split Data
validate <- table[indexes, ]
dim(validate)
train <- table[-indexes, ]
dim(train)
#Heating Load Multiple Linear Regression
Model1_HL <- lm(Heating_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Roof_Area+Overall_Height+Orientation+Glazing_Area+Glazing_Area_Distribution, data = train)
summary(Model1_HL)
#Remove variables with p >0.05
Model2_HL <- lm(Heating_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Overall_Height+Glazing_Area, data = train)
summary(Model2_HL)
pred_Model2_HL <- predict(Model2_HL, validate)
pred_Model2_HL
#To check the model quality
residuals_Model2_HL <- validate$Heating_Load - predict(Model2_HL, validate)
residuals_Model2_HL
MSE_Model2_HL <- (sum((residuals_Model2_HL)^2))/nrow(validate)
RMSE_Model2_HL <- sqrt(MSE_Model2_HL)
RMSE_Model2_HL 
#R-sqaured on the validate dataset
Model2_HL_Rsqaured <- 1-sum(residuals_Model2_HL^2)/sum((validate$Heating_Load-mean(validate$Heating_Load))^2)
Model2_HL_Rsqaured
# validate rsquared less than train rsquared. Yay!
x_HL <- validate$Cooling_Load
y_HL <- predict(Model2_HL, validate)
reg_HL <- lm(y_HL ~ x_HL, data = validate)
plot(validate$Heating_Load,predict(Model2_HL, validate), xlab = "Actual Heating Load Values", ylab = "Predicted Heating Load Values",type = "p")
abline(reg_HL, col = "blue")

#Cooling Load Multiple Linear Regression
Model1_CL <- lm(Cooling_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Roof_Area+Overall_Height+Orientation+Glazing_Area+Glazing_Area_Distribution, data = train)
summary(Model1_CL)
#Remove variables with p >0.05
Model2_CL <- lm(Cooling_Load ~ Relative_Compactness+Surface_Area+Wall_Area+Overall_Height+Glazing_Area, data = train)
summary(Model2_CL)
pred_Model2_CL <- predict(Model2_CL, validate)
pred_Model2_CL
#To check the model quality
residuals_Model2_CL <- validate$Cooling_Load - predict(Model2_CL, validate)
residuals_Model2_CL
MSE_Model2_CL <- (sum((residuals_Model2_CL)^2))/nrow(validate)
RMSE_Model2_CL <- sqrt(MSE_Model2_CL)
RMSE_Model2_CL
#R-sqaured on the validate dataset
Model2_CL_Rsqaured <- 1-sum(residuals_Model2_CL^2)/sum((validate$Cooling_Load-mean(validate$Cooling_Load))^2)
Model2_CL_Rsqaured
x_CL <- validate$Cooling_Load
y_CL <- predict(Model2_CL, validate)
reg_CL <- lm(y_CL ~ x_CL, data = validate)
plot(validate$Cooling_Load,predict(Model2_CL, validate), xlab = "Actual Cooling Load Values", ylab = "Predicted Cooling Load Values",type = "p")
abline(reg_HL, col = "red")