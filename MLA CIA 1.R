# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)

# Load the dataset
data <- read_csv("C:/Users/GAUTAM CHANDRA/Downloads/insurance.csv")

# Display the first few rows of the dataset
head(data)

# Summary statistics
summary(data)

# Structure of the dataset
str(data)

# Visualizing distributions of numerical variables
ggplot(data, aes(x = age)) + geom_histogram(binwidth = 1) + ggtitle("Age Distribution")
ggplot(data, aes(x = bmi)) + geom_histogram(binwidth = 1) + ggtitle("BMI Distribution")
ggplot(data, aes(x = children)) + geom_bar() + ggtitle("Children Distribution")

# Visualizing categorical variables
ggplot(data, aes(x = sex)) + geom_bar() + ggtitle("Sex Distribution")
ggplot(data, aes(x = smoker)) + geom_bar() + ggtitle("Smoker Distribution")
ggplot(data, aes(x = region)) + geom_bar() + ggtitle("Region Distribution")

# Check for missing values
missing_values <- sapply(data, function(x) sum(is.na(x)))
print(missing_values)

# Standardize numerical variables
data <- data %>%
  mutate(bmi = scale(bmi), age = scale(age), children = scale(children))

# One-hot encoding for categorical variables
data$sex <- as.factor(data$sex)
data$smoker <- as.factor(data$smoker)
data$region <- as.factor(data$region)
data_encoded <- model.matrix(~ . - 1, data = data) %>% as.data.frame()

# Outlier detection and treatment for 'charges'
Q1 <- quantile(data$charges, 0.25)
Q3 <- quantile(data$charges, 0.75)
IQR <- Q3 - Q1
outliers <- which(data$charges < (Q1 - 1.5 * IQR) | data$charges > (Q3 + 1.5 * IQR))
data <- data[-outliers, ]



# Modeling

# Split the data into training and testing sets
set.seed(42)
target <- data_encoded$charges
features <- data_encoded %>% select(-charges)
trainIndex <- createDataPartition(target, p = .8, list = FALSE)
X_train <- features[trainIndex, ]
X_test <- features[-trainIndex, ]
y_train <- target[trainIndex]
y_test <- target[-trainIndex]

# LINEAR REGRESSION 
linear_model <- train(X_train, y_train, method = "lm")
linear_predictions <- predict(linear_model, X_test)
linear_rmse <- RMSE(linear_predictions, y_test)
print(paste("Linear Regression RMSE:", linear_rmse))


# Plot actual vs. predicted values for Linear Regression
plot_actual_vs_predicted_linear <- ggplot(data = data.frame(actual = y_test, predicted = linear_predictions), aes(x = actual, y = predicted)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Actual vs. Predicted Charges (Linear Regression)", x = "Actual Charges", y = "Predicted Charges") +
  theme_minimal()

# Plot residuals for Linear Regression
plot_residuals_linear <- ggplot(data = data.frame(residuals = y_test - linear_predictions, predicted = linear_predictions), aes(x = predicted, y = residuals)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs. Predicted Charges (Linear Regression)", x = "Predicted Charges", y = "Residuals") +
  theme_minimal()

# Display plots for Linear Regression
grid.arrange(plot_actual_vs_predicted_linear, plot_residuals_linear, ncol = 2)




# LASSO REGRESSION 
lasso_model <- train(X_train, y_train, method = "glmnet", tuneGrid = expand.grid(alpha = 1, lambda = 0.01))
lasso_predictions <- predict(lasso_model, X_test)
lasso_rmse <- RMSE(lasso_predictions, y_test)
print(paste("Lasso Regression RMSE:", lasso_rmse))


## Plot actual vs. predicted values for Lasso Regression
plot_actual_vs_predicted_lasso <- ggplot(data = data.frame(actual = y_test, predicted = lasso_predictions), aes(x = actual, y = predicted)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Actual vs. Predicted Charges (Lasso Regression)", x = "Actual Charges", y = "Predicted Charges") +
  theme_minimal()

# Plot residuals for Lasso Regression
plot_residuals_lasso <- ggplot(data = data.frame(residuals = y_test - lasso_predictions, predicted = lasso_predictions), aes(x = predicted, y = residuals)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs. Predicted Charges (Lasso Regression)", x = "Predicted Charges", y = "Residuals") +
  theme_minimal()

# Display plots for Lasso Regression
grid.arrange(plot_actual_vs_predicted_lasso, plot_residuals_lasso, ncol = 2)



# Ridge Regression
ridge_model <- train(X_train, y_train, method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = 0.01))
ridge_predictions <- predict(ridge_model, X_test)
ridge_rmse <- RMSE(ridge_predictions, y_test)
print(paste("Ridge Regression RMSE:", ridge_rmse))


# Plot actual vs. predicted values for Ridge Regression
plot_actual_vs_predicted_ridge <- ggplot(data = data.frame(actual = y_test, predicted = ridge_predictions), aes(x = actual, y = predicted)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Actual vs. Predicted Charges (Ridge Regression)", x = "Actual Charges", y = "Predicted Charges") +
  theme_minimal()

# Plot residuals for Ridge Regression
plot_residuals_ridge <- ggplot(data = data.frame(residuals = y_test - ridge_predictions, predicted = ridge_predictions), aes(x = predicted, y = residuals)) +
  geom_point(color = 'blue', alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs. Predicted Charges (Ridge Regression)", x = "Predicted Charges", y = "Residuals") +
  theme_minimal()

# Display plots for Ridge Regression
grid.arrange(plot_actual_vs_predicted_ridge, plot_residuals_ridge, ncol = 2)


# Model Output
model_comparison <- data.frame(
  Model = c("Linear Regression", "Lasso Regression", "Ridge Regression"),
  RMSE = c(linear_rmse, lasso_rmse, ridge_rmse)
)


#COMPARISON
print(model_comparison)

best_model <- model_comparison[which.min(model_comparison$RMSE), ]
print(paste("Best Model:", best_model$Model))



