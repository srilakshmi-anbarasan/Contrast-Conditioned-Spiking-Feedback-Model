library(ggplot2)
library(data.table)
library(dplyr)

session_data <- list ()
for (i in 1:18) {
  session_data[[i]] <- readRDS(paste0("/Users/srilakshmianbarasan/Downloads/sessions.rds/session", i, ".rds"))
}

# Data Summary and Visualization
summary <- rbindlist(lapply(session_data, function(session) {
  data.table(
    mouse33 = unique(session$mouse_name),
    date = unique(session$date_exp),
    number_of_trials = length(session$feedback_type),
    number_of_neurons = nrow(session$spks[[1]]),
    feedback_success = sum(session$feedback_type == 1),
    feedback_failure = sum(session$feedback_type == -1)
  )
}))

print(summary)

ggplot(summary, aes(x = as.factor(mouse33), y = number_of_trials, fill = as.factor(mouse33))) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Trials per Mouse", x = "Mouse Name", y = "Number of Trials")

# Stimuli Conditions
contrast_df <- rbindlist(lapply(session_data, function(s) {
  data.table(contrast_left = s$contrast_left, contrast_right = s$contrast_right)
}), fill = TRUE)

 
ggplot(contrast_df, aes(x = contrast_left, y = contrast_right)) +
  geom_point(alpha = 0.5, color = "green") +
  theme_minimal() +
  labs(title = "Stimuli Contrast: Left vs. Right", x = "Left Contrast", y = "Right Contrast")

neuron_firing_rate <- sapply(session_data, function(s) {
  colMeans(s$spks[[1]])
})
summary(neuron_firing_rate) # Average spike rate/neuron

# Spike count distribution across trials
spike_count <- sapply(session_data, function(s) rowSums(s$spks[[1]]))
hist(unlist(spike_count), breaks = 50, main = "Spike Count Distribution", xlab = "Total Spikes")

# Trends in firing patterns over time
spike_activity <- colMeans(session_data[[1]]$spks[[1]])
spike_df <-data.frame(Time = seq_along(spike_activity), FiringRate = spike_activity)

ggplot(spike_df, aes(x = Time, y= FiringRate)) +
  geom_line(color = "red") +
  theme_minimal() +
  labs(title = "Mean Firing Rate Over Time", x = "Time Intervals", y ="Average Spikes")

# Part 2: Data Integration

# Ensuring uniformity in data structure by handling missing values and creating a unified data set
library(data.table)
full_set <- rbindlist(lapply(session_data, function(s) {
  data.table(
    mouse = s$mouse_name,
    date = s$date_exp,
    feedback = s$feedback_type,
    contrast_left = s$contrast_left,
    contrast_right = s$contrast_right,
    spikes = rowSums(s$spks[[1]])
  )
}), fill = TRUE)

'
Identifying common trends in behavioral response and neural activity
How is neuron firing impacted across different stimuli conditions? Figure out by computing mean spike count
of contrast right vs left and feedback
'
trial_avg <- full_set[, .(mean_spikes = mean(spikes)), by = .(contrast_left, contrast_right, feedback)]
print(trial_avg)

# Find similarities in firing rate patterns to group sessions with similar neuron firing behavior
# and find patterns across sessions 
library(cluster)
clusters <- kmeans(scale(full_set[, .(spikes)]), centers = 3) 
full_set[, cluster := clusters$cluster]

'
To account for the variance in neural activity across different days and mice
 by standardizing spike counts across sessions to normalize neural activity
'
full_set[, norm_spikes := (spikes - mean(spikes)) / sd(spikes), by = .(mouse, date)]



'
Next step is to reduce bias that is session-specific by fitting a simple linear model
Then we can help induce generalization in the predictive model by extracting residual spikes
This way, variantions in days and different mice are removed from accountability, and neural activity
is more comparable across different sessions
'

# Fitting model and extracting residuals
session_model <- lm(norm_spikes ~ contrast_left + contrast_right + feedback + mouse + date, data = full_set, na.action = na.exclude)
full_set[, residual_spikes := NA_real_] # To prevent introducing size mismatches
full_set[!is.na(norm_spikes), residual_spikes := residuals(session_model)]

# Retaining relevant columns for predictive modeling part
final_data <- full_set[, .(contrast_left, contrast_right, feedback, norm_spikes, residual_spikes)]
print(head(final_data))

# Saving integrated data for predictive modeling
saveRDS(final_data, "integrated_data.rds")

# Preventing overfitting and good model training metrics
library(caret)
final_data <- na.omit(final_data)

set.seed(42)
index_train <- createDataPartition(final_data$feedback, p = 0.8, list = FALSE)
trainset <- final_data[index_train, ]
test1set <- final_data[index_train, ]

# Compare performance across different models

# Model 1: Logistic Regression
trainset[, feedback_bin := ifelse(feedback == -1, 0, 1)]
logistic <- glm(feedback_bin ~ contrast_left + contrast_right + residual_spikes,
                data = trainset, family = binomial())
summary(logistic)

# Model 2: Non-linear Random Forest Model
chooseCRANmirror(graphics = FALSE, ind = 1)
install.packages("randomForest")
library(randomForest)

trainset$feedback <- as.factor(trainset$feedback)

set.seed(123)

random_forest <- randomForest(feedback ~ contrast_left + contrast_right + residual_spikes,
                               data = trainset, ntree = 700, mtry = 3)
print(random_forest)

# Model 3: Support Vector Machine
library(e1071)
svm <- svm(feedback ~ contrast_left + contrast_right + residual_spikes,
            data = trainset, kernel = "radial")
summary(svm)

# Model accuracy on test1set on Logistic Regression Evaluation
log_pred <- predict(logistic, newdata = test1set, type = "response")
log_pred <- ifelse(log_pred > 0.5, 1, -1)
log_accuracy <- mean(log_pred == test1set$feedback)
cat("Accuracy of Logistic Regress Model:", log_accuracy, "\n")

# Model accuracy on test1set on Random Forest Evaluation
rfpred <- predict(random_forest, newdata = test1set)
rf_accuracy <- mean(rfpred == test1set$feedback)
cat("Random Forest Model Accuracy:", rf_accuracy, "\n")

# Evaluate SVM Model
svm_pred <- predict(svm, newdata = test1set)
svm_accuracy <- mean(svm_pred == test1set$feedback)
cat("SVM Accuracy:", svm_accuracy, "\n")

saveRDS(rf_model, "best_model.rds")  

test_data <- readRDS("test.rds")  
best_model <- readRDS("best_model.rds")

test_preds <- predict(best_model, newdata = test_data)
write.csv(data.frame(Predictions = test_preds), "test_predictions.csv", row.names = FALSE)
test1 <- readRDS("test1.rds")
test2 <- readRDS("test2.rds")

write.csv(test1, "test1.csv", row.names = FALSE)
write.csv(test2, "test2.csv", row.names = FALSE)



