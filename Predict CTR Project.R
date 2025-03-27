
## PREDICTION CLICKS PROJECT ##

# Load required libraries
library(dplyr)
library(caret)
library(xgboost)

# Load data
train <- read.csv("~/Downloads/predicting-clicks/analysis_data.csv")
test <- read.csv("~/Downloads/predicting-clicks/scoring_data.csv")

# Compute proportions for categorical variables in training data
location_proportions <- prop.table(table(train$location, useNA = "no"))
gender_proportions <- prop.table(table(train$gender, useNA = "no"))
age_proportions <- prop.table(table(train$age_group, useNA = "no"))

# Impute missing categorical values in training data
train$location[is.na(train$location)] <- sample(
  names(location_proportions),
  sum(is.na(train$location)),
  replace = TRUE,
  prob = location_proportions
)

train$gender[is.na(train$gender)] <- sample(
  names(gender_proportions),
  sum(is.na(train$gender)),
  replace = TRUE,
  prob = gender_proportions
)

train$age_group[is.na(train$age_group)] <- sample(
  names(age_proportions),
  sum(is.na(train$age_group)),
  replace = TRUE,
  prob = age_proportions
)

# Impute missing categorical values in test data using training data proportions
test$location[is.na(test$location)] <- sample(
  names(location_proportions),
  sum(is.na(test$location)),
  replace = TRUE,
  prob = location_proportions
)

test$gender[is.na(test$gender)] <- sample(
  names(gender_proportions),
  sum(is.na(test$gender)),
  replace = TRUE,
  prob = gender_proportions
)

test$age_group[is.na(test$age_group)] <- sample(
  names(age_proportions),
  sum(is.na(test$age_group)),
  replace = TRUE,
  prob = age_proportions
)

# Identify numeric columns in training data
num_cols <- names(train)[sapply(train, is.numeric)]

# Perform mean imputation for numeric columns in both train and test datasets
for (col in num_cols) {
  if (col %in% names(test)) {
    train[[col]][is.na(train[[col]])] <- mean(train[[col]], na.rm = TRUE)
    test[[col]][is.na(test[[col]])] <- mean(train[[col]], na.rm = TRUE)
  }
}

# Ensure all categorical columns have matching levels
categorical_cols <- c("location", "gender", "age_group")
for (col in categorical_cols) {
  train[[col]] <- factor(train[[col]])
  test[[col]] <- factor(test[[col]], levels = levels(train[[col]]))
}

# Convert categorical variables to dummy variables
dummies <- dummyVars(" ~ .", data = train %>% select(-CTR))
train_dummies <- as.data.frame(predict(dummies, newdata = train %>% select(-CTR)))
test_dummies <- as.data.frame(predict(dummies, newdata = test))

# Add target variable back to training data
train_dummies$CTR <- train$CTR

# Hyperparameter tuning using caret
control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)  # 5-fold cross-validation

tune_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(4, 6, 8),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 1, 5),
  colsample_bytree = c(0.7, 1.0),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.7, 1.0)
)

# Use the formula interface for train
set.seed(123)
xgb_tuned <- caret::train(
  CTR ~ ., 
  data = train_dummies,
  method = "xgbTree",
  trControl = control,
  tuneGrid = tune_grid,
  metric = "RMSE"
)

# Best hyperparameters
cat("Best Hyperparameters:\n")
print(xgb_tuned$bestTune)

# Train the final model with best parameters
final_model <- xgboost(
  data = xgb.DMatrix(as.matrix(train_dummies %>% select(-CTR)), label = train_dummies$CTR),
  max_depth = xgb_tuned$bestTune$max_depth,
  eta = xgb_tuned$bestTune$eta,
  nrounds = xgb_tuned$bestTune$nrounds,
  gamma = xgb_tuned$bestTune$gamma,
  colsample_bytree = xgb_tuned$bestTune$colsample_bytree,
  min_child_weight = xgb_tuned$bestTune$min_child_weight,
  subsample = xgb_tuned$bestTune$subsample,
  objective = "reg:squarederror"
)

# Predict on test data
test_matrix <- xgb.DMatrix(as.matrix(test_dummies))
pred_test <- predict(final_model, test_matrix)

# Evaluate RMSE on training data
train_matrix <- xgb.DMatrix(as.matrix(train_dummies %>% select(-CTR)), label = train_dummies$CTR)
pred_train <- predict(final_model, train_matrix)
rmse_train <- sqrt(mean((train_dummies$CTR - pred_train)^2))
cat("Training RMSE:", rmse_train, "\n")

# Create a submission file
submission_file <- data.frame(id = test$id, CTR = pred_test)
write.csv(submission_file, "~/Downloads/predicting-clicks/xgboost_submission.csv", row.names = FALSE)

cat("Submission file successfully created: xgboost_submission.csv\n")
