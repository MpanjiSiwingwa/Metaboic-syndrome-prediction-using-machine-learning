# =============================================
# METABOLIC SYNDROME PREDICTION PIPELINE - UPDATED
# =============================================
# Version: 9.1 - Optimized and Standardized
# Last Updated: 15th November 2025
# Author: Mpanji Siwingwa
# 
# COMPLETE UPDATED PIPELINE WITH ALL FIXES APPLIED
# =============================================

# 1. INITIAL SETUP AND CONFIGURATION
# ==================================
# Load libraries
required_packages <- c(
  "tidyverse", "caret", "recipes", "themis", "pROC", 
  "randomForest", "xgboost", "glmnet", "e1071", "rpart", 
  "kknn", "lightgbm", "PRROC", "rmda", "ResourceSelection"
)

invisible(lapply(required_packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}))

# Set global parameters
RANDOM_SEED <- 123
set.seed(RANDOM_SEED)

# Set working directory to a specific folder
setwd('/Users/mpanjisiwingwa/Library/Mobile Documents/com~apple~CloudDocs/School/PhD/Articles Publication/Mathine Learning/Finals/Week_144')

# 2. DATA PREPARATION AND CLEANING
# =================================

# 2.1 Load and Inspect Data
# -------------------------
cat("Loading data...\n")
df <- read.csv("MetSyn_dataset_Week_144.csv")

# 2.2 Data Cleaning
# -----------------
df <- df %>%
  select(-Blood_Sugar..mmol.L.,
         -Triglycerides..mmol.L.,
         -Cholesterol_HDL_.mmol.L.,
         -BMI.kg.m2.,
         -Bp_Systolic..mmHg.,
         -Bp_Diastolic..mmHg.,
         -Waist_Circumference.cm.)

# 2.3 Feature Engineering
# -----------------------
df <- df %>%
  rename(
    Alcohol_Consumption = Alcohol_Consuption,
    Regimen_Type = Drug_Code
  )

# 2.4 Additional Cleaning
# -----------------------
cols_to_clean <- c("Regimen_Type", "Sex", "Diabetes_Mellitus_status", 
                   "Tobbacco_Use", "Alcohol_Consumption", "Educational_Level", 
                   "Event_Name")

df[cols_to_clean] <- lapply(
  df[cols_to_clean], 
  function(x) if (is.character(x)) trimws(x) else x
)

# Clean Event_Name values
df$Event_Name[df$Event_Name == "Week_ 144"] <- "week_144"

# Remove Baseline and drop Event_Name
df <- df %>%
  filter(Event_Name != "Baseline") %>%
  select(-Event_Name)

# 2.5 Imputation with MICE
# ------------------------
library(mice)

imp <- mice(df, m = 5, method = "pmm", maxit = 10, seed = 123)

# Check missing data pattern
md.pattern(df)

# Extract one completed dataset (first imputation)
df_imputed <- complete(imp, 1)

# Verify imputation
stopifnot(sum(is.na(df_imputed)) == 0)

# After imputation
metabolic_data <- df_imputed   # use the completed dataset

# 2.6 Convert to Factors
# ----------------------
metabolic_data <- df_imputed %>%
  mutate(across(c("Regimen_Type", "Sex", "Diabetes_Mellitus_status",
                  "Alcohol_Consumption", "Educational_Level",  
                  "Tobbacco_Use", "Metabolic_Syndrome"), as.factor),
         Educational_Level = factor(Educational_Level,
                                    levels = c("none", "primary", "secondary", "Terciary"),
                                    ordered = TRUE))

# Final dataset ready for modeling
str(metabolic_data)


# 3. FEATURE SELECTION
# =====================
# use data set after imputation

metabolic_data <- df_imputed   # use the completed dataset


# 3.1 Feature Importance with Random Forest
# -----------------------------------------
cat("Performing feature selection...\n")

# 1. Define predictors and outcome, excluding ID and target
predictor_vars <- setdiff(names(metabolic_data), c("Metabolic_Syndrome", "Trial_number"))
x <- metabolic_data[, predictor_vars]
y <- as.factor(metabolic_data$Metabolic_Syndrome)

# 2. Fit Random Forest model
rf_model <- randomForest::randomForest(
  x = x,
  y = y,
  importance = TRUE,
  ntree = 500,
  strata = y,
  sampsize = rep(min(table(y)), 2)
)

# 3. Extract and rank variable importance
importance_scores <- randomForest::importance(rf_model, type = 2, scale = TRUE)
varimp_data <- data.frame(
  Variable = rownames(importance_scores),
  Importance = importance_scores[, "MeanDecreaseGini"]
) %>% arrange(desc(Importance))

# 4. Select top features based on threshold
importance_threshold <- max(varimp_data$Importance) * 0.07
varimp_data <- varimp_data %>%
  mutate(Selected = ifelse(Importance >= importance_threshold, "Selected", "Not Selected"))

# NEW: Extract selected variable names 
rf_selected <- varimp_data %>% 
  filter(Selected == "Selected") %>% 
  pull(Variable)

# 5. Plot all features, color-coded by selection status
ggplot(varimp_data, aes(x = reorder(Variable, Importance), y = Importance, fill = Selected)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("Selected" = "#0072B2", "Not Selected" = "#D55E00")) +
  labs(title = "Figure 2. Random Forest Feature Importance",
       x = "Features",
       y = "MeanDecreaseGini",
       fill = "Selection Status") +
  theme_minimal(base_size = 14)


# Save the plot as high-resolution PNG
ggsave("Figure_2_RF_Feature_Importance.png",
       width = 10,
       height = 6,
       dpi = 600)  # High resolution

# ----------------------------------------------------
# 3.0 Collinearity Check (Treat categorical as single variables)
# ----------------------------------------------------

cat("Checking for multicollinearity...\n")

# Step 1: Prepare dataset
vif_data <- metabolic_data %>%
  mutate(across(where(is.factor), droplevels)) %>%        
  mutate(Metabolic_Syndrome_numeric = ifelse(Metabolic_Syndrome == "MetSyn", 1, 0))

# Step 2: Ensure categorical variables are factors and set reference levels
vif_data$Diabetes_Mellitus_status <- relevel(factor(vif_data$Diabetes_Mellitus_status), ref = "no")
vif_data$Tobbacco_Use             <- relevel(factor(vif_data$Tobbacco_Use), ref = "no")
vif_data$Alcohol_Consumption      <- relevel(factor(vif_data$Alcohol_Consumption), ref = "no")
vif_data$Educational_Level        <- relevel(factor(vif_data$Educational_Level), ref = "none")
vif_data$Sex                      <- relevel(factor(vif_data$Sex), ref = "female")
vif_data$Regimen_Type             <- relevel(factor(vif_data$Regimen_Type), ref = "AZT+3TC+LPVr")

# Step 3: Select predictors ONLY
predictors <- vif_data %>%
  select(-Trial_number, -Metabolic_Syndrome, -Metabolic_Syndrome_numeric)

# Step 4: Combine outcome + predictors (no dummy expansion)
vif_model_data <- data.frame(
  Metabolic_Syndrome = vif_data$Metabolic_Syndrome_numeric,
  predictors
)

# Step 5: Fit full linear model with factors intact
full_lm <- lm(Metabolic_Syndrome ~ ., data = vif_model_data)

# Step 6: Detect aliased (perfectly collinear) variables
aliased <- alias(full_lm)$Complete

if (!is.null(aliased)) {
  cat("\n🚨 Aliased (perfectly collinear) variables detected:\n")
  print(rownames(aliased))
  
  # Remove aliased variables
  vif_model_data <- vif_model_data %>%
    select(-all_of(rownames(aliased)))
  
  # Refit model without collinear variables
  vif_model <- lm(Metabolic_Syndrome ~ ., data = vif_model_data)
  
} else {
  vif_model <- full_lm
}

# Step 7: Compute VIF
vif_results <- car::vif(vif_model)

cat("\n📊 VIF Summary (categorical variables treated as single predictors):\n")
print(round(vif_results, 2))

# Step 9: Create tidy VIF table safely
if (is.matrix(vif_results)) {
  vif_table <- data.frame(
    Variable = rownames(vif_results),
    VIF = round(vif_results[,1], 2)
  )
} else if (is.numeric(vif_results)) {
  vif_table <- data.frame(
    Variable = names(vif_results),
    VIF = round(as.numeric(vif_results), 2)
  )
} else {
  vif_table <- data.frame(
    Variable = character(),
    VIF = numeric()
  )
  cat("\n⚠️ No VIF results available.\n")
}

vif_table

library(flextable)
library(officer)


# vif_table <- data.frame(Variable = names(vif_results), VIF = round(vif_results, 2))

# Define border style
gray_border <- fp_border(color = "gray", width = 0.5)
black_border <- fp_border(color = "black", width = 1)

# Create flextable
ft <- flextable(vif_table) %>%
  autofit() %>%
  theme_vanilla() %>%
  set_header_labels(
    Variable = "Predictor Variable",
    VIF = "Variance Inflation Factor"
  ) %>%
  bold(part = "header") %>%
  fontsize(size = 11, part = "all") %>%
  border_remove() %>%
  border_outer(part = "all", border = black_border) %>%
  border_inner_h(border = gray_border) %>%
  border_inner_v(border = gray_border) %>%
  add_header_row(values = c("Table 2. Variance Inflation Factors (VIF)"), colwidths = 2) %>%
  bold(i = 1, part = "header") %>%
  align(align = "center", part = "all")

# Save to Word
doc <- read_docx() %>%
  body_add_flextable(ft) %>%
  body_add_par("Note: VIF > 8 indicates potential multicollinearity.", style = "Normal")

print(doc, target = "Table_2_VIF.docx")

# 3.2 Create Final Data set
# ------------------------
selected_data <- metabolic_data %>%
  select(all_of(rf_selected), Metabolic_Syndrome)

cat("Selected features:", paste(rf_selected, collapse = ", "), "\n")
cat("Final dataset dimensions:", dim(selected_data), "\n")

# 4. DATA PREPROCESSING PIPELINE
# ===============================

# 4.1 Enhanced Feature Engineering Recipe
# ---------------------------------------
create_feature_engineered_recipe <- function(data) {
  recipe(Metabolic_Syndrome ~ ., data = data) %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_mutate(
      CD4_VL_Ratio = CD4_count.cells.µl. / (Viral_Load.cp.ml. + 1),
      Log_Viral_Load = log10(Viral_Load.cp.ml. + 1)
    ) %>%
    step_rm(CD4_count.cells.µl., Viral_Load.cp.ml.) %>%
    step_nzv(all_numeric_predictors()) %>%
    step_corr(all_numeric_predictors(), threshold = 0.95) %>%
    step_zv(all_predictors()) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_normalize(all_numeric_predictors())
}

# 4.2 Data Validation Function
# ----------------------------
validate_data_for_ml <- function(data, data_name = "dataset") {
  cat("=== VALIDATING", toupper(data_name), "===\n")
  
  # Basic checks
  cat("Dimensions:", dim(data), "\n")
  cat("NA values:", sum(is.na(data)), "\n")
  cat("Infinite values:", sum(apply(data, 2, function(x) any(is.infinite(x)))), "\n")
  
  # Class distribution
  if ("Metabolic_Syndrome" %in% names(data)) {
    cat("Class distribution:\n")
    print(table(data$Metabolic_Syndrome))
  }
  
  # Near-zero variance
  nzv_check <- nearZeroVar(data, saveMetrics = TRUE)
  nzv_problems <- sum(nzv_check$nzv)
  cat("Near-zero variance features:", nzv_problems, "\n")
  
  cat("=============================\n")
}

# 4.3 Unified Data Splitting
# --------------------------
cat("Splitting data temporally...\n")

# ------------------------------------------------------------
# Data Split: 70% Training | 30% Testing
# Dataset: Week 144 only
# ------------------------------------------------------------
# Load required package
library(caret)

# Set seed for reproducibility
set.seed(123)

# Ensure the outcome variable is a factor with correct levels
selected_data$Metabolic_Syndrome <- factor(
  selected_data$Metabolic_Syndrome,
  levels = c("No MetSyn", "MetSyn")
)

# Create stratified partition (70% training)
train_indices <- createDataPartition(
  selected_data$Metabolic_Syndrome,
  p = 0.7,
  list = FALSE
)

# Split the dataset
train_set <- selected_data[train_indices, ]
test_set  <- selected_data[-train_indices, ]

# Display summary
cat("Training set size:", nrow(train_set), "\n")
cat("Testing set size:", nrow(test_set), "\n")

# Check distribution of the outcome in each split
prop.table(table(train_set$Metabolic_Syndrome))
prop.table(table(test_set$Metabolic_Syndrome))

# Ensure proper factor levels for the outcome variable
train_set$Metabolic_Syndrome <- factor(train_set$Metabolic_Syndrome, levels = c("No MetSyn", "MetSyn"))
test_set$Metabolic_Syndrome  <- factor(test_set$Metabolic_Syndrome, levels = c("No MetSyn", "MetSyn"))

# 4.4 Create All Data set Variants
# --------------------------------

create_dataset_variants <- function(train_data, test_data) {
  datasets <- list()
  
  # Original data
  datasets$original <- list(train = train_data, test = test_data)
  
  # Feature engineered data
  feature_recipe <- create_feature_engineered_recipe(train_data) %>%
    update_role(Metabolic_Syndrome, new_role = "outcome")   # Preserve outcome
  
  prepped_features <- prep(feature_recipe, training = train_data)
  
  # Bake predictors + outcome preserved
  train_fe <- bake(prepped_features, new_data = train_data, composition = "data.frame")
  test_fe  <- bake(prepped_features, new_data = test_data, composition = "data.frame")
  
  # Validate baked datasets
  cat("Checking baked feature-engineered datasets...\n")
  cat("Train outcome NAs:", sum(is.na(train_fe$Metabolic_Syndrome)), "\n")
  cat("Test outcome NAs:", sum(is.na(test_fe$Metabolic_Syndrome)), "\n")
  
  datasets$feature_engineered <- list(train = train_fe, test = test_fe)
  
  # SMOTE data (requires clean outcome factor)
  if (sum(is.na(train_fe$Metabolic_Syndrome)) > 0) {
    stop("Outcome variable has missing values in train_fe — cannot run SMOTE.")
  }
  
  smote_recipe <- recipe(Metabolic_Syndrome ~ ., data = train_fe) %>%
    step_smote(Metabolic_Syndrome, over_ratio = 1)
  
  prepped_smote <- prep(smote_recipe, training = train_fe)
  train_smote <- bake(prepped_smote, new_data = NULL, composition = "data.frame")
  
  datasets$smote <- list(train = train_smote, test = test_fe)
  
  # Validate all datasets
  for (name in names(datasets)) {
    validate_data_for_ml(datasets[[name]]$train, paste(name, "training"))
    validate_data_for_ml(datasets[[name]]$test, paste(name, "testing"))
  }
  
  return(datasets)
}


# Create all data set variants
datasets <- create_dataset_variants(train_set, test_set)


# 5. MODEL TRAINING CONFIGURATION
# ================================

# --------------------------------------
# 5.1 FIXED: Enhanced Clinical Summary Function with valid level names
# -------------------------------------------------------------------
robust_clinical_summary <- function(data, lev = NULL, model = NULL) {
  # Use valid R variable names (without spaces)
  if (is.null(lev)) lev <- c("No_MetSyn", "MetSyn")
  if (!all(lev %in% levels(data$obs))) data$obs <- factor(data$obs, levels = lev)
  if (!all(lev %in% levels(data$pred))) data$pred <- factor(data$pred, levels = lev)
  
  tryCatch({
    # Base metrics
    base_metrics <- defaultSummary(data, lev, model)
    
    # Clinical metrics
    cm <- confusionMatrix(data$pred, data$obs, positive = "MetSyn")
    
    sensitivity <- cm$byClass["Sensitivity"]
    specificity <- cm$byClass["Specificity"]
    ppv <- cm$byClass["Pos Pred Value"]
    npv <- cm$byClass["Neg Pred Value"]
    
    # Clinical utility (simplified)
    tp <- sum(data$pred == "MetSyn" & data$obs == "MetSyn")
    fn <- sum(data$pred == "No_MetSyn" & data$obs == "MetSyn")
    fp <- sum(data$pred == "MetSyn" & data$obs == "No_MetSyn")
    clinical_utility <- (tp * 1) - (fn * 5) - (fp * 1)
    
    c(base_metrics,
      Sensitivity = sensitivity,
      Specificity = specificity,
      PPV = ppv,
      NPV = npv,
      ClinicalUtility = clinical_utility)
  }, error = function(e) {
    warning(paste("Metric calculation error:", e$message))
    return(c(Accuracy = NA, Kappa = NA, Sensitivity = NA, Specificity = NA, 
             PPV = NA, NPV = NA, ClinicalUtility = NA))
  })
}

# FIX: Convert factor levels to valid R variable names in your datasets
# ---------------------------------------------------------------------
cat("=== FIXING FACTOR LEVELS IN ALL DATASETS ===\n")

fix_factor_levels <- function(dataset_list) {
  for (data_name in names(dataset_list)) {
    # Fix training data
    dataset_list[[data_name]]$train$Metabolic_Syndrome <- factor(
      dataset_list[[data_name]]$train$Metabolic_Syndrome,
      levels = c("No MetSyn", "MetSyn"),
      labels = c("No_MetSyn", "MetSyn")  # Convert to valid names
    )
    
    # Fix testing data
    dataset_list[[data_name]]$test$Metabolic_Syndrome <- factor(
      dataset_list[[data_name]]$test$Metabolic_Syndrome,
      levels = c("No MetSyn", "MetSyn"),
      labels = c("No_MetSyn", "MetSyn")  # Convert to valid names
    )
    
    cat("Fixed levels in", data_name, ":\n")
    cat("  Train:", levels(dataset_list[[data_name]]$train$Metabolic_Syndrome), "\n")
    cat("  Test:", levels(dataset_list[[data_name]]$test$Metabolic_Syndrome), "\n")
  }
  return(dataset_list)
}

# Apply the fix to all datasets
datasets <- fix_factor_levels(datasets)

# Verify the fix
cat("\n=== VERIFYING FIXED LEVELS ===\n")
for (data_name in names(datasets)) {
  cat(data_name, "train levels:", levels(datasets[[data_name]]$train$Metabolic_Syndrome), "\n")
  cat(data_name, "test levels:", levels(datasets[[data_name]]$test$Metabolic_Syndrome), "\n")
}

# Also fix your original train_set and test_set for any future use
train_set$Metabolic_Syndrome <- factor(train_set$Metabolic_Syndrome, 
                                       levels = c("No MetSyn", "MetSyn"),
                                       labels = c("No_MetSyn", "MetSyn"))

test_set$Metabolic_Syndrome <- factor(test_set$Metabolic_Syndrome, 
                                      levels = c("No MetSyn", "MetSyn"), 
                                      labels = c("No_MetSyn", "MetSyn"))

cat("\nOriginal sets fixed:\n")
cat("train_set levels:", levels(train_set$Metabolic_Syndrome), "\n")
cat("test_set levels:", levels(test_set$Metabolic_Syndrome), "\n")

# 5.2 Training Control Configuration
# ----------------------------------
ctrl_robust <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = robust_clinical_summary,
  selectionFunction = "best",
  verboseIter = TRUE,
  allowParallel = TRUE,
  sampling = "up"
)

# 5.3 Model Configuration
# -----------------------

# ---- Load required package ----
library(lightgbm)

# ---- Define caret-compatible LightGBM model ----
caret_lightgbm <- list(
  label = "LightGBM",
  library = "lightgbm",
  type = c("Classification"),
  parameters = data.frame(
    parameter = c("num_leaves", "learning_rate", "nrounds"),
    class     = rep("numeric", 3),
    label     = c("Number of Leaves", "Learning Rate", "Boosting Rounds")
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    expand.grid(
      num_leaves    = c(31, 63),
      learning_rate = c(0.05, 0.1),
      nrounds       = c(100, 200)
    )
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    x_matrix <- as.matrix(x)
    dtrain <- lgb.Dataset(x_matrix, label = as.numeric(y) - 1)
    
    model <- lightgbm::lgb.train(
      params = list(
        objective     = "binary",
        num_leaves    = param$num_leaves,
        learning_rate = param$learning_rate,
        verbose       = -1
      ),
      data    = dtrain,
      nrounds = param$nrounds
    )
    
    # Create a wrapper object
    result <- list(
      lgb_model = model,
      xNames = colnames(x),
      obsLevels = lev,
      problemType = "Classification"
    )
    class(result) <- "caretLightGBM"
    return(result)
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    newdata_matrix <- as.matrix(newdata[, modelFit$xNames, drop = FALSE])
    preds <- predict(modelFit$lgb_model, newdata_matrix)
    factor(
      ifelse(preds > 0.5, modelFit$obsLevels[2], modelFit$obsLevels[1]),
      levels = modelFit$obsLevels
    )
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    newdata_matrix <- as.matrix(newdata[, modelFit$xNames, drop = FALSE])
    preds <- predict(modelFit$lgb_model, newdata_matrix)
    out <- cbind(1 - preds, preds)
    colnames(out) <- modelFit$obsLevels
    as.data.frame(out)
  }
)
# ---- Define model configurations ----
model_configs <- list(
  logistic = list(
    method = "glmnet",
    grid = expand.grid(alpha = 0.5, lambda = 0.01)
  ),
  svm_linear = list(
    method = "svmLinear",
    grid = expand.grid(C = c(0.01, 0.1, 1))
  ),
  svm_radial = list(
    method = "svmRadial",
    grid = expand.grid(sigma = c(0.01, 0.1), C = c(0.5, 1))
  ),
  decision_tree = list(
    method = "rpart",
    grid = NULL
  ),
  random_forest = list(
    method = "rf",
    grid = expand.grid(mtry = c(2, 4, 6))
  ),
  knn = list(
    method = "kknn",
    grid = NULL
  ),
  xgboost = list(
    method = "xgbTree",
    grid = expand.grid(
      nrounds = 100,
      max_depth = 6,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      subsample = 0.8
    )
  ),
  naive_bayes = list(
    method = "naive_bayes",
    grid = data.frame(laplace = 0, usekernel = FALSE, adjust = 1)
  ),
  lightgbm = list(
    method = caret_lightgbm,
    grid = NULL
  )
)

# 6. UNIFIED MODEL TRAINING PIPELINE
# ===================================

# 6.1 Robust Model Training Function (uses baked data, scoped feature checks)
# ---------------------------------------------------------------------------
train_robust_model <- function(train_data, method, tune_grid = NULL, model_name = "model") {
  cat("Training", model_name, "...\n")
  
  # Data validation
  if (nrow(train_data) == 0) {
    warning("Empty training data")
    return(NULL)
  }
  
  class_counts <- table(train_data$Metabolic_Syndrome)
  if (length(class_counts) < 2 || any(class_counts < 2)) {
    warning(paste("Insufficient classes:", paste(class_counts, collapse = ", ")))
    return(NULL)
  }
  
  # Feature check only for engineered datasets
  if (grepl("feature_engineered|smote", model_name)) {
    required_features <- c("CD4_VL_Ratio", "Log_Viral_Load")
    missing_features <- setdiff(required_features, names(train_data))
    if (length(missing_features) > 0) {
      warning(paste("Missing features in", model_name, ":", paste(missing_features, collapse = ", ")))
      return(NULL)
    }
  }
  
  tryCatch({
    if (!is.null(tune_grid)) {
      model <- train(
        Metabolic_Syndrome ~ .,
        data = train_data,
        method = method,
        trControl = ctrl_robust,
        tuneGrid = tune_grid,
        metric = "ClinicalUtility"
      )
    } else {
      model <- train(
        Metabolic_Syndrome ~ .,
        data = train_data,
        method = method,
        trControl = ctrl_robust,
        metric = "ClinicalUtility",
        tuneLength = 3
      )
    }
    
    cat("✓", model_name, "trained successfully\n")
    return(model)
    
  }, error = function(e) {
    warning(paste("Failed to train", model_name, ":", e$message))
    return(NULL)
  })
}

# 6.2 Batch Model Training (uses baked datasets)
# ----------------------------------------------
train_all_models <- function(datasets, model_configs) {
  all_models <- list()
  
  with_progress({
    p <- progressor(steps = length(datasets) * length(model_configs))
    
    for (data_name in names(datasets)) {
      all_models[[data_name]] <- list()
      train_data <- datasets[[data_name]]$train  # baked data
      
      for (model_name in names(model_configs)) {
        config <- model_configs[[model_name]]
        full_model_name <- paste(data_name, model_name, sep = "_")
        
        model <- train_robust_model(
          train_data = train_data,
          method = config$method,
          tune_grid = config$grid,
          model_name = full_model_name
        )
        
        all_models[[data_name]][[model_name]] <- model
        p()
      }
    }
  })
  
  return(all_models)
}

# 6.3 Training Control Configuration (parallel disabled)
# ------------------------------------------------------
ctrl_robust <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = robust_clinical_summary,
  selectionFunction = "best",
  verboseIter = TRUE,
  allowParallel = FALSE,  # 🔧 disabled to prevent connection errors
  sampling = "up"
)

# 6.4 Execute Model Training
# --------------------------
# load library
library(progressr)

cat("\n=== STARTING MODEL TRAINING ===\n")
all_models <- train_all_models(datasets, model_configs)


# -----------------------------------------------------------------
# 6.5 Manual Retraining of svm-Radial (for debugging or validation)
# -----------------------------------------------------------------

# Clean and prepare training data
training_data <- datasets$original$train
training_data <- training_data[complete.cases(training_data), ]
training_data <- distinct(training_data)
training_data <- training_data[, sapply(training_data, function(x) length(unique(x)) > 1)]  # remove zero-variance features
rownames(training_data) <- NULL
training_data$Metabolic_Syndrome <- factor(training_data$Metabolic_Syndrome, levels = c("No_MetSyn", "MetSyn"))

# Custom control for ROC scoring and probability support
ctrl_svm <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter = TRUE,
  allowParallel = FALSE,
  sampling = "up"
)

# Train svmRadial model with ROC metric
svm_radial_model <- train(
  Metabolic_Syndrome ~ ., 
  data = training_data,
  method = "svmRadial",
  trControl = ctrl_svm,
  tuneGrid = model_configs$svm_radial$grid,
  preProcess = c("center", "scale"),
  metric = "ROC"
)

# Prepare test data for prediction
test_data <- datasets$original$test
test_data <- test_data[complete.cases(test_data), ]
test_data <- distinct(test_data)
test_data <- test_data[, sapply(test_data, function(x) length(unique(x)) > 1)]
rownames(test_data) <- NULL
test_data$Metabolic_Syndrome <- factor(test_data$Metabolic_Syndrome, levels = c("No_MetSyn", "MetSyn"))

# Predict probabilities
probs <- predict(svm_radial_model, newdata = test_data, type = "prob")
head(probs)


# 7 Model Evaluation
# 7.1 Enhanced MCC Calculation
# ----------------------------
calculate_mcc <- function(cm) {
  if (inherits(cm, "confusionMatrix")) {
    cm <- cm$table
  }
  
  if (all(dim(cm) == c(2, 2))) {
    TP <- as.numeric(cm[2, 2])
    TN <- as.numeric(cm[1, 1])
    FP <- as.numeric(cm[1, 2])
    FN <- as.numeric(cm[2, 1])
    
    numerator <- (TP * TN) - (FP * FN)
    denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    if (denominator == 0) return(0)
    return(numerator / denominator)
  }
  return(NA)
}

# 7.2 FIXED Evaluation Using Youden Index Threshold
# -------------------------------------------
evaluate_model_youden <- function(model, test_data, model_name = "model", outcome_col = "Metabolic_Syndrome") {
  if (is.null(model)) {
    warning(paste("Model is NULL:", model_name))
    return(list(
      Accuracy = NA, Sensitivity = NA, Specificity = NA,
      PPV = NA, NPV = NA, MCC = NA, AUC = NA, Threshold = NA
    ))
  }
  
  tryCatch({
    # Clean test data
    test_data <- test_data[complete.cases(test_data), ]
    rownames(test_data) <- NULL
    
    # Get true labels and ensure factor levels
    truth <- factor(test_data[[outcome_col]], levels = c("No_MetSyn", "MetSyn"))
    
    # Get predicted probabilities as numeric vector
    probs_raw <- predict(model, newdata = test_data, type = "prob")
    if (!("MetSyn" %in% colnames(probs_raw))) stop("Missing 'MetSyn' column in predicted probabilities")
    
    probs <- as.vector(probs_raw[, "MetSyn", drop = TRUE])
    if (!is.numeric(probs)) probs <- as.numeric(probs)
    cat("✅ probs length:", length(probs), "\n")
    
    # Final length check
    if (length(probs) != length(truth)) {
      stop(paste("Length mismatch: probs =", length(probs), "truth =", length(truth)))
    }
    
    if (any(is.na(probs))) {
      stop("Predicted probabilities contain NA")
    }
    
    # Compute ROC and Youden threshold
    roc_obj <- pROC::roc(truth, probs, levels = c("No_MetSyn", "MetSyn"), direction = "<")
    
    # 🔧 FIX: Extract single threshold value
    youden_coords <- pROC::coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
    youden_threshold <- as.numeric(youden_coords)  # Ensure it's a single numeric value
    
    cat("✅ youden_threshold:", youden_threshold, "\n")
    
    # 🔧 FIX: Use vectorized comparison directly
    pred_labels <- ifelse(probs > youden_threshold, "MetSyn", "No_MetSyn")
    
    cat("✅ pred_labels length:", length(pred_labels), "\n")
    
    preds <- factor(pred_labels, levels = c("No_MetSyn", "MetSyn"))
    cat("✅ preds length:", length(preds), "\n")
    
    # Final length check
    if (length(preds) != length(truth)) {
      stop(paste("Length mismatch: preds =", length(preds), "truth =", length(truth)))
    }
    
    # Confusion matrix
    cm <- confusionMatrix(preds, truth, positive = "MetSyn")
    
    # Metrics
    list(
      Accuracy = cm$overall["Accuracy"],
      Sensitivity = cm$byClass["Sensitivity"],
      Specificity = cm$byClass["Specificity"],
      PPV = cm$byClass["Pos Pred Value"],
      NPV = cm$byClass["Neg Pred Value"],
      MCC = calculate_mcc(cm),
      AUC = as.numeric(pROC::auc(roc_obj)),
      Threshold = round(youden_threshold, 4)
    )
  }, error = function(e) {
    warning(paste("Evaluation failed for", model_name, ":", e$message))
    return(list(
      Accuracy = NA, Sensitivity = NA, Specificity = NA,
      PPV = NA, NPV = NA, MCC = NA, AUC = NA, Threshold = NA
    ))
  })
}

# 7.3 Batch Model Evaluation (Youden Index)
# -----------------------------------------
evaluate_all_models_youden <- function(models, datasets) {
  results <- list()
  
  for (data_name in names(models)) {
    results[[data_name]] <- list()
    
    test_data <- datasets[[data_name]]$test  # 🔧 use matching test set
    
    for (model_name in names(models[[data_name]])) {
      full_model_name <- paste(data_name, model_name, sep = "_")
      model <- models[[data_name]][[model_name]]
      
      results[[data_name]][[model_name]] <- evaluate_model_youden(
        model, test_data, full_model_name
      )
    }
  }
  
  return(results)
}

# 7.4 Execute Model Evaluation
# ----------------------------
cat("\n=== EVALUATING MODELS WITH YOUDEN INDEX THRESHOLD ===\n")
all_results_youden <- evaluate_all_models_youden(all_models, datasets)

# 7.6 SAVE TRAINING PARAMETERS FOR EXTERNAL VALIDATION ===
cat("\n=== SAVING TRAINING PARAMETERS ===\n")

# Save ACTUAL training statistics from the ORIGINAL dataset
training_stats <- list(
  # Numeric feature statistics from training data
  numeric_means = sapply(datasets$original$train %>% select(where(is.numeric)), mean, na.rm = TRUE),
  numeric_sds = sapply(datasets$original$train %>% select(where(is.numeric)), sd, na.rm = TRUE),
  
  # Factor levels from training data
  factor_levels = lapply(datasets$original$train %>% select(where(is.factor)), levels),
  
  # Imputation values from training
  imputation_medians = sapply(datasets$original$train %>% select(where(is.numeric)), median, na.rm = TRUE)
)

saveRDS(training_stats, "training_parameters.rds")
cat("✓ Training parameters saved to 'training_parameters.rds'\n")

# Also save the selected features from RF
if (exists("rf_selected")) {
  saveRDS(rf_selected, "training_selected_features.rds")
  cat("✓ Selected features saved\n")
}

# Save the pre-processing recipe
if (exists("prepped_features")) {
  saveRDS(prepped_features, "training_preprocessing_recipe.rds")
  cat("✓ Preprocessing recipe saved\n")
}


# 8. RESULTS ANALYSIS AND VISUALIZATION
# ======================================

# 8.1 Results Compilation
# -----------------------
compile_results <- function(results) {
  results_df <- data.frame()
  
  for (data_name in names(results)) {
    for (model_name in names(results[[data_name]])) {
      metrics <- results[[data_name]][[model_name]]
      
      row <- data.frame(
        Dataset = data_name,
        Model = model_name,
        Accuracy = round(metrics$Accuracy, 3),
        Sensitivity = round(metrics$Sensitivity, 3),
        Specificity = round(metrics$Specificity, 3),
        PPV = round(metrics$PPV, 3),
        NPV = round(metrics$NPV, 3),
        MCC = round(metrics$MCC, 3),
        AUC = round(metrics$AUC, 3),
        stringsAsFactors = FALSE
      )
      
      results_df <- rbind(results_df, row)
    }
  }
  
  return(results_df)
}

# 8.2 Generate Results Summary
# ----------------------------
results_summary <- compile_results(all_results_youden)

# Display top models by AUC
cat("\n=== TOP MODELS BY AUC ===\n")
top_models <- results_summary %>%
  arrange(desc(AUC)) %>%
  head(10)

print(top_models)

# 8.3 Visualization
# -----------------
# ROC Curves
generate_roc_curves <- function(models, test_data) {
  roc_data <- data.frame()
  
  for (data_name in names(models)) {
    for (model_name in names(models[[data_name]])) {
      model <- models[[data_name]][[model_name]]
      if (!is.null(model)) {
        tryCatch({
          probs <- predict(model, test_data, type = "prob")[, "MetSyn"]
          roc_obj <- pROC::roc(test_data$Metabolic_Syndrome, probs)
          
          temp_data <- data.frame(
            FPR = 1 - roc_obj$specificities,
            TPR = roc_obj$sensitivities,
            Model = paste(data_name, model_name, sep = "_"),
            AUC = round(pROC::auc(roc_obj), 3)
          )
          
          roc_data <- rbind(roc_data, temp_data)
        }, error = function(e) NULL)
      }
    }
  }
  
  return(roc_data)
}

roc_data <- generate_roc_curves(all_models, test_set)

# Plot ROC curves
if (nrow(roc_data) > 0) {
  library(ggplot2)
  
  roc_plot <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
    geom_line() +
    geom_abline(linetype = "dashed", color = "gray") +
    labs(title = "ROC Curves - All Models",
         x = "False Positive Rate",
         y = "True Positive Rate") +
    theme_minimal()
  
  print(roc_plot)
  ggsave("roc_curves.png", plot = roc_plot, width = 10, height = 8)
}

# ===============================================================
# 8.4 Precision–Recall Curves (Discrimination in Imbalanced Data)
# ===============================================================

library(PRROC)
library(ggplot2)
library(dplyr)
library(purrr)
library(caret)
library(tibble)

# ---- Precision–Recall Function ----
plot_pr_from_probs <- function(pred_probs, true_labels, label = "Model") {
  pr <- pr.curve(
    scores.class0 = pred_probs[true_labels == 1],
    scores.class1 = pred_probs[true_labels == 0],
    curve = TRUE
  )
  tibble(
    Recall = pr$curve[,1],
    Precision = pr$curve[,2],
    Threshold = pr$curve[,3],
    AUC_PR = pr$auc.integral,
    Model = label
  )
}

# ---- PR Data for All Models ----
train_all_models <- all_models[["original"]]
labels_bin <- ifelse(datasets$original$test$Metabolic_Syndrome == "MetSyn", 1, 0)

pr_data <- map_dfr(names(train_all_models), function(m) {
  probs <- predict(train_all_models[[m]], datasets$original$test, type = "prob")[, "MetSyn"]
  plot_pr_from_probs(probs, labels_bin, m)
})

fig2_pr <- ggplot(pr_data, aes(x = Recall, y = Precision, color = Model)) +
  geom_line(linewidth = 1) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Figure 2. Precision–Recall Curves by Model",
    x = "Recall (Sensitivity)", y = "Precision (PPV)"
  ) +
  theme(legend.position = "bottom", legend.title = element_text(face = "bold")) +
  geom_abline(slope = 0, intercept = mean(pr_data$Precision),
              linetype = "dotted", color = "gray") +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))


# ===============================================================
# 8.5 Calibration Helper
# ===============================================================

plot_calibration_from_probs <- function(pred_probs, true_labels, label = "Model", bins = 10) {
  tibble(Pred = pred_probs, Actual = as.numeric(true_labels)) %>%
    mutate(bin = ntile(Pred, bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(Pred, na.rm = TRUE),
      mean_obs = mean(Actual, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Model = label)
}


# ===============================================================
# 8.6 Brier Scores (Original)
# ===============================================================

brier_data <- map_dfr(names(train_all_models), function(m) {
  probs <- predict(train_all_models[[m]], datasets$original$test, type = "prob")[, "MetSyn"]
  tibble(Model = m, Brier_Score = mean((probs - labels_bin)^2, na.rm = TRUE))
})


# ===============================================================
# 8.7 Recalibration: Platt Scaling + Isotonic Regression
# ===============================================================

recalibrate_and_plot <- function(pred_probs, true_labels, model_name = "Model", bins = 10) {
  labels_bin <- as.numeric(true_labels)
  
  # ---- Platt scaling ----
  platt_model <- glm(labels_bin ~ pred_probs, family = binomial)
  platt_probs <- predict(platt_model, type = "response")
  
  # ---- Isotonic regression ----
  iso_model <- isoreg(pred_probs, labels_bin)
  iso_probs <- fitted(iso_model)
  
  # ---- Brier scores ----
  brier_scores <- tibble(
    Model = c(paste(model_name, "Original"),
              paste(model_name, "Platt"),
              paste(model_name, "Isotonic")),
    Brier_Score = c(
      mean((pred_probs - labels_bin)^2, na.rm = TRUE),
      mean((platt_probs - labels_bin)^2, na.rm = TRUE),
      mean((iso_probs - labels_bin)^2, na.rm = TRUE)
    )
  )
  
  # ---- Calibration data ----
  cal_df <- bind_rows(
    plot_calibration_from_probs(pred_probs, labels_bin, paste(model_name, "Original"), bins),
    plot_calibration_from_probs(platt_probs, labels_bin, paste(model_name, "Platt"), bins),
    plot_calibration_from_probs(iso_probs, labels_bin, paste(model_name, "Isotonic"), bins)
  ) %>%
    mutate(Base_Model = model_name)
  
  list(brier_scores = brier_scores, calibration_data = cal_df)
}

# ---- Batch Recalibration for All Models ----
recalibration_results <- map(names(train_all_models), function(m) {
  probs <- predict(train_all_models[[m]], datasets$original$test, type = "prob")[, "MetSyn"]
  recalibrate_and_plot(probs, labels_bin, model_name = m)
})

# ---- Collect Brier Scores ----
brier_all_models <- bind_rows(map(recalibration_results, "brier_scores"))

# ---- Collect Calibration Data ----
calibration_all <- bind_rows(map(recalibration_results, "calibration_data"))

# ---- Combined Faceted Calibration Plot ----
fig_calibration_all <- ggplot(calibration_all,
                              aes(x = mean_pred, y = mean_obs, color = Model)) +
  geom_point(size = 2) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  facet_wrap(~ Base_Model, scales = "free") +
  labs(title = "Calibration Curves Across Models (Original vs Recalibrated)",
       x = "Mean Predicted Probability", y = "Mean Observed Outcome") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", legend.title = element_text(face = "bold"))


# ===============================================================
# 8.8 Confusion Matrix Metrics (Optimal Threshold by Youden Index)
# ===============================================================
get_cm_metrics_youden_from_probs <- function(pred_probs, true_labels, label = "Model") {
  require(pROC)
  
  thresholds <- seq(0, 1, by = 0.01)
  metrics <- sapply(thresholds, function(t) {
    predicted <- ifelse(pred_probs >= t, 1, 0)
    cm <- table(factor(predicted, levels = c(0,1)),
                factor(true_labels, levels = c(0,1)))
    TP <- cm[2,2]; TN <- cm[1,1]; FP <- cm[2,1]; FN <- cm[1,2]
    sens <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
    spec <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
    youden <- sens + spec - 1
    return(c(sens, spec, youden))
  })
  
  # Find best threshold by Youden Index
  best_idx <- which.max(metrics[3,])
  best_threshold <- thresholds[best_idx]
  
  # Apply best threshold
  predicted <- ifelse(pred_probs >= best_threshold, "MetSyn", "No_MetSyn")
  actual_label <- ifelse(true_labels == 1, "MetSyn", "No_MetSyn")
  
  cm <- caret::confusionMatrix(factor(predicted, levels = c("No_MetSyn","MetSyn")),
                               factor(actual_label, levels = c("No_MetSyn","MetSyn")))
  
  # ---- Compute AUC ----
  roc_obj <- pROC::roc(true_labels, pred_probs, quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  
  tibble(
    Model = label,
    AUC = round(auc_val, 3),
    Optimal_Threshold = round(best_threshold, 3),
    Sensitivity = metrics[1, best_idx],
    Specificity = metrics[2, best_idx],
    PPV = cm$byClass["Pos Pred Value"],
    NPV = cm$byClass["Neg Pred Value"],
    Accuracy = cm$overall["Accuracy"],
    Youden_Index = metrics[3, best_idx]
  )
}

# ---- Batch across all models ----
cm_table <- map_dfr(names(train_all_models), function(m) {
  probs <- predict(train_all_models[[m]], datasets$original$test, type = "prob")[, "MetSyn"]
  get_cm_metrics_youden_from_probs(probs, labels_bin, m)
})

cm_table

# ===============================================================
# 1️⃣ LOAD REQUIRED LIBRARIES
# ===============================================================
library(dplyr)
library(purrr)
library(pROC)
library(caret)
library(flextable)
library(officer)

# ===============================================================
# 2️⃣ FUNCTION: Compute Metrics (Optimal Threshold via Youden Index)
# ===============================================================
get_cm_metrics_youden_from_probs <- function(pred_probs, true_labels, label = "Model") {
  thresholds <- seq(0, 1, by = 0.01)
  metrics <- sapply(thresholds, function(t) {
    predicted <- ifelse(pred_probs >= t, 1, 0)
    cm <- table(factor(predicted, levels = c(0,1)),
                factor(true_labels, levels = c(0,1)))
    TP <- cm[2,2]; TN <- cm[1,1]; FP <- cm[2,1]; FN <- cm[1,2]
    sens <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
    spec <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
    youden <- sens + spec - 1
    return(c(sens, spec, youden))
  })
  
  # Identify optimal threshold by Youden Index
  best_idx <- which.max(metrics[3,])
  best_threshold <- thresholds[best_idx]
  
  # Apply optimal threshold
  predicted <- ifelse(pred_probs >= best_threshold, "MetSyn", "No_MetSyn")
  actual_label <- ifelse(true_labels == 1, "MetSyn", "No_MetSyn")
  
  cm <- caret::confusionMatrix(factor(predicted, levels = c("No_MetSyn","MetSyn")),
                               factor(actual_label, levels = c("No_MetSyn","MetSyn")))
  
  # Compute AUC
  roc_obj <- pROC::roc(true_labels, pred_probs, quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  
  tibble(
    Model = label,
    AUC = round(auc_val, 3),
    Optimal_Threshold = round(best_threshold, 3),
    Sensitivity = round(metrics[1, best_idx], 3),
    Specificity = round(metrics[2, best_idx], 3),
    PPV = round(cm$byClass["Pos Pred Value"], 3),
    NPV = round(cm$byClass["Neg Pred Value"], 3),
    Accuracy = round(cm$overall["Accuracy"], 3),
    Youden_Index = round(metrics[3, best_idx], 3)
  )
}

# ===============================================================
# 2️⃣ FUNCTION: Compute Metrics (Optimal Threshold via Youden Index)
#     + 95% CI for AUC using DeLong (pROC::ci.auc)
# ===============================================================
get_cm_metrics_youden_from_probs <- function(pred_probs, true_labels, label = "Model", conf_level = 0.95) {
  thresholds <- seq(0, 1, by = 0.01)
  metrics <- sapply(thresholds, function(t) {
    predicted <- ifelse(pred_probs >= t, 1, 0)
    cm <- table(factor(predicted, levels = c(0,1)),
                factor(true_labels, levels = c(0,1)))
    TP <- cm[2,2]; TN <- cm[1,1]; FP <- cm[2,1]; FN <- cm[1,2]
    sens <- ifelse((TP + FN) > 0, TP / (TP + FN), NA_real_)
    spec <- ifelse((TN + FP) > 0, TN / (TN + FP), NA_real_)
    youden <- sens + spec - 1
    c(sens, spec, youden)
  })
  
  # Identify optimal threshold by Youden Index
  best_idx <- which.max(metrics[3,])
  best_threshold <- thresholds[best_idx]
  
  # Apply optimal threshold
  predicted <- ifelse(pred_probs >= best_threshold, "MetSyn", "No_MetSyn")
  actual_label <- ifelse(true_labels == 1, "MetSyn", "No_MetSyn")
  
  cm <- caret::confusionMatrix(
    factor(predicted, levels = c("No_MetSyn","MetSyn")),
    factor(actual_label, levels = c("No_MetSyn","MetSyn"))
  )
  
  # Compute AUC and 95% CI (DeLong)
  roc_obj <- pROC::roc(true_labels, pred_probs, quiet = TRUE, direction = "<")
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_auc  <- suppressWarnings(pROC::ci.auc(roc_obj, conf.level = conf_level, method = "delong"))
  auc_lower <- as.numeric(ci_auc[1])
  auc_upper <- as.numeric(ci_auc[3])
  
  tibble(
    Model            = label,
    AUC              = round(auc_val, 3),
    AUC_Lower        = round(auc_lower, 3),
    AUC_Upper        = round(auc_upper, 3),
    AUC_CI           = paste0("(", round(auc_lower, 3), ", ", round(auc_upper, 3), ")"),
    Optimal_Threshold= round(best_threshold, 3),
    Sensitivity      = round(metrics[1, best_idx], 3),
    Specificity      = round(metrics[2, best_idx], 3),
    PPV              = round(cm$byClass["Pos Pred Value"], 3),
    NPV              = round(cm$byClass["Neg Pred Value"], 3),
    Accuracy         = round(cm$overall["Accuracy"], 3),
    Youden_Index     = round(metrics[3, best_idx], 3)
  )
}

# ===============================================================
# 3️⃣ ISOTONIC CALIBRATION + FINAL METRICS
# ===============================================================
cat("🚀 GENERATING FINAL METRICS WITH ISOTONIC CALIBRATION\n")

final_metrics_table <- map_dfr(names(train_all_models), function(m) {
  cat("🔍 Processing:", m, "\n")
  
  # Predict probabilities from each model
  probs_original <- predict(train_all_models[[m]], datasets$original$test, type = "prob")[, "MetSyn"]
  
  # Skip models with no variation in probabilities
  if (length(unique(probs_original)) <= 3) {
    cat("⚠️ Skipping", m, "- no probability variation\n")
    return(NULL)
  }
  
  # Apply isotonic calibration (best-performing method)
  iso_model <- isoreg(probs_original, labels_bin)
  probs_calibrated <- fitted(iso_model)
  
  # Clip probabilities to avoid 0 or 1 boundaries
  probs_calibrated <- pmin(pmax(probs_calibrated, 0.01), 0.99)
  
  cat("📊", m, "- Isotonic calibrated range:", round(range(probs_calibrated), 3), "\n")
  
  # Calculate metrics using isotonic-calibrated probabilities
  get_cm_metrics_youden_from_probs(probs_calibrated, labels_bin, m)
})


final_metrics_table <- final_metrics_table %>%
  mutate(
    AUC_with_CI = paste0(
      round(AUC, 3), " (",
      round(AUC_Lower, 3), ", ",
      round(AUC_Upper, 3), ")"
    )
  ) %>%
  select(Model, AUC_with_CI, Optimal_Threshold, Sensitivity, Specificity, PPV, NPV, Accuracy, Youden_Index)


cat("\n🎯 FINAL RESULTS WITH ISOTONIC CALIBRATION:\n")
print(final_metrics_table, width = Inf)

# ===============================================================
# 4️⃣ EXPORT METRICS TO WORD (Auto-fit to Page)
# ===============================================================
cat("\n📝 Exporting metrics table to Word document...\n")

final_metrics_table_clean <- final_metrics_table %>%
  rename_with(~ gsub("_", " ", .x)) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

ft <- flextable(final_metrics_table_clean) %>%
  theme_booktabs() %>%
  autofit() %>%
  set_caption("Final Model Performance Metrics Using Isotonic-Calibrated Probabilities") %>%
  fontsize(size = 10, part = "all") %>%
  align(align = "center", part = "all") %>%
  width(width = 1.2) %>%
  set_table_properties(layout = "autofit", width = 1)

doc <- read_docx() %>%
  body_add_par("Final Model Performance Metrics Report", style = "heading 1") %>%
  body_add_flextable(value = ft)

print(doc, target = "final_model_metrics_isotonic.docx")

cat("✅ Word document 'final_model_metrics_isotonic.docx' successfully created and page-fitted!\n")


# ===============================================================
# 📦 Load Required Packages
# ===============================================================
library(dplyr)
library(purrr)
library(tidyr)
library(ggplot2)
library(ResourceSelection)

# ===============================================================
# 🧭 Helper Functions
# ===============================================================

## ---- 1. Calibration Plot Data ----
plot_calibration_from_probs <- function(pred_probs, true_labels, label = "Model", bins = 10) {
  tibble(Pred = pred_probs, Actual = as.numeric(true_labels)) %>%
    mutate(bin = ntile(Pred, bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(Pred, na.rm = TRUE),
      mean_obs = mean(Actual, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Model = label)
}

## ---- 2. Safe Hosmer–Lemeshow Wrapper ----
safe_hoslem <- function(y, probs, g) {
  tryCatch({
    hoslem.test(y, probs, g = g)
  }, error = function(e) list(p.value = NA, statistic = NA),
  warning = function(w) list(p.value = NA, statistic = NA))
}

## ---- 3. Calibration Quality Interpretation ----
interpret_calibration <- function(p) {
  if (is.na(p)) return(NA)
  if (p > 0.05) "Good calibration" else "Poor calibration"
}

# ===============================================================
# 🔁 Model Recalibration: Platt Scaling + Isotonic Regression
# ===============================================================

recalibrate_and_plot <- function(pred_probs, true_labels, model_name = "Model", bins = 5) {
  labels_bin <- as.numeric(true_labels)
  
  # Skip if probabilities lack variation
  if (length(unique(pred_probs)) <= 3) {
    cat("⚠️ Model", model_name, "has no probability variation - skipping recalibration\n")
    return(list(
      brier_scores = tibble(
        Model = paste(model_name, "Original"),
        Brier_Score = mean((pred_probs - labels_bin)^2, na.rm = TRUE)
      ),
      calibration_data = plot_calibration_from_probs(pred_probs, labels_bin, paste(model_name, "Original"), bins),
      platt_model = NULL, iso_model = NULL
    ))
  }
  
  # ---- Platt Scaling ----
  platt_model <- tryCatch({
    glm(labels_bin ~ pred_probs, family = binomial, control = glm.control(maxit = 100))
  }, error = function(e) {
    cat("❌ Platt failed for", model_name, ":", e$message, "\n")
    NULL
  })
  
  platt_probs <- if (!is.null(platt_model))
    pmin(pmax(predict(platt_model, type = "response"), 0.01), 0.99) else pred_probs
  
  # ---- Isotonic Regression ----
  iso_model <- tryCatch({
    isoreg(pred_probs, labels_bin)
  }, error = function(e) {
    cat("❌ Isotonic failed for", model_name, ":", e$message, "\n")
    NULL
  })
  
  iso_probs <- if (!is.null(iso_model))
    pmin(pmax(fitted(iso_model), 0.01), 0.99) else pred_probs
  
  # ---- Brier Scores ----
  brier_scores <- tibble(
    Model = c(paste(model_name, "Original"),
              paste(model_name, "Platt"),
              paste(model_name, "Isotonic")),
    Brier_Score = c(
      mean((pred_probs - labels_bin)^2, na.rm = TRUE),
      mean((platt_probs - labels_bin)^2, na.rm = TRUE),
      mean((iso_probs - labels_bin)^2, na.rm = TRUE)
    )
  )
  
  # ---- Calibration Data ----
  cal_df <- bind_rows(
    plot_calibration_from_probs(pred_probs, labels_bin, paste(model_name, "Original"), bins),
    plot_calibration_from_probs(platt_probs, labels_bin, paste(model_name, "Platt"), bins),
    plot_calibration_from_probs(iso_probs, labels_bin, paste(model_name, "Isotonic"), bins)
  ) %>%
    mutate(Base_Model = model_name)
  
  list(
    brier_scores = brier_scores,
    calibration_data = cal_df,
    platt_model = platt_model,
    iso_model = iso_model
  )
}


# ===============================================================
# ⚙️ Batch Recalibration Across All Models
# ===============================================================

recalibration_results <- map(names(train_all_models), function(m) {
  probs <- predict(train_all_models[[m]], datasets$original$test, type = "prob")[, "MetSyn"]
  recalibrate_and_plot(probs, labels_bin, model_name = m)
})

# ===============================================================
# 📉 Brier Score Comparison
# ===============================================================

brier_all_models <- bind_rows(map(recalibration_results, "brier_scores"))

brier_summary_table <- brier_all_models %>%
  separate(Model, into = c("Base_Model", "Method"), sep = " ", extra = "merge") %>%
  pivot_wider(names_from = Method, values_from = Brier_Score) %>%
  mutate(
    `Platt Brier` = round(Platt, 3),
    `Isotonic Brier` = round(Isotonic, 3),
    `Better Method` = case_when(
      is.na(Platt) | is.na(Isotonic) ~ "Tie",
      Platt < Isotonic ~ "Platt",
      Isotonic < Platt ~ "Isotonic",
      TRUE ~ "Tie"
    ),
    Improvement = round(abs(Platt - Isotonic), 3)
  ) %>%
  select(Model = Base_Model, `Platt Brier`, `Isotonic Brier`, `Better Method`, Improvement)

print(brier_summary_table)


# Define a Bootstrapping Function
bootstrap_brier_ci <- function(pred_probs, true_labels, n_boot = 1000, conf_level = 0.95) {
  labels_bin <- as.numeric(true_labels)
  n <- length(pred_probs)
  if (n == 0) {
    return(list(mean = NA_real_, lower = NA_real_, upper = NA_real_))
  }
  brier_scores <- replicate(n_boot, {
    idx <- sample.int(n, replace = TRUE)
    mean((pred_probs[idx] - labels_bin[idx])^2, na.rm = TRUE)
  })
  alpha <- (1 - conf_level) / 2
  ci <- quantile(brier_scores, probs = c(alpha, 1 - alpha), na.rm = TRUE)
  list(mean = mean(brier_scores, na.rm = TRUE), lower = ci[1], upper = ci[2])
}

# recalibrate and_plot
recalibrate_and_plot <- function(pred_probs, true_labels, model_name = "Model", bins = 5, n_boot = 1000) {
  labels_bin <- as.numeric(true_labels)
  
  # If probabilities lack variation, return Original only with consistent columns
  if (length(unique(pred_probs)) <= 3) {
    cat("⚠️ Model", model_name, "has low probability variation - recalibration skipped\n")
    brier_orig <- bootstrap_brier_ci(pred_probs, labels_bin, n_boot)
    return(list(
      brier_scores = tibble(
        Model = paste(model_name, "Original"),
        Brier_Mean = brier_orig$mean,
        CI_Lower    = brier_orig$lower,
        CI_Upper    = brier_orig$upper
      ),
      calibration_data = plot_calibration_from_probs(pred_probs, labels_bin, paste(model_name, "Original"), bins) %>%
        mutate(Base_Model = model_name),
      platt_model = NULL,
      iso_model   = NULL
    ))
  }
  
  # ---- Platt scaling ----
  platt_model <- tryCatch({
    glm(labels_bin ~ pred_probs, family = binomial, control = glm.control(maxit = 100))
  }, error = function(e) {
    cat("❌ Platt failed for", model_name, ":", e$message, "\n"); NULL
  })
  
  platt_probs <- if (!is.null(platt_model)) {
    pmin(pmax(predict(platt_model, type = "response"), 0.01), 0.99)
  } else pred_probs
  
  # ---- Isotonic regression ----
  iso_model <- tryCatch({
    isoreg(pred_probs, labels_bin)
  }, error = function(e) {
    cat("❌ Isotonic failed for", model_name, ":", e$message, "\n"); NULL
  })
  
  iso_probs <- if (!is.null(iso_model)) {
    pmin(pmax(fitted(iso_model), 0.01), 0.99)
  } else pred_probs
  
  # ---- Bootstrapped Brier scores ----
  brier_original <- bootstrap_brier_ci(pred_probs, labels_bin, n_boot)
  brier_platt    <- bootstrap_brier_ci(platt_probs, labels_bin, n_boot)
  brier_iso      <- bootstrap_brier_ci(iso_probs, labels_bin, n_boot)
  
  brier_scores <- tibble(
    Model = c(paste(model_name, "Original"),
              paste(model_name, "Platt"),
              paste(model_name, "Isotonic")),
    Brier_Mean = c(brier_original$mean, brier_platt$mean, brier_iso$mean),
    CI_Lower    = c(brier_original$lower, brier_platt$lower, brier_iso$lower),
    CI_Upper    = c(brier_original$upper, brier_platt$upper, brier_iso$upper)
  )
  
  cal_df <- bind_rows(
    plot_calibration_from_probs(pred_probs,     labels_bin, paste(model_name, "Original"), bins),
    plot_calibration_from_probs(platt_probs,    labels_bin, paste(model_name, "Platt"),    bins),
    plot_calibration_from_probs(iso_probs,      labels_bin, paste(model_name, "Isotonic"), bins)
  ) %>%
    mutate(Base_Model = model_name)
  
  list(
    brier_scores     = brier_scores,
    calibration_data = cal_df,
    platt_model      = platt_model,
    iso_model        = iso_model
  )
}

# Batch: Apply across models
recalibration_results <- map(names(train_all_models), function(m) {
  probs <- predict(train_all_models[[m]], datasets$original$test, type = "prob")[, "MetSyn"]
  recalibrate_and_plot(probs, labels_bin, model_name = m, bins = 5, n_boot = 1000)
})


# Update Summary Table
brier_all_models <- bind_rows(map(recalibration_results, "brier_scores"))

# Expect columns: Model, Brier_Mean, CI_Lower, CI_Upper
# Sanity check (optional):
# print(unique(names(brier_all_models)))

brier_summary_table <- brier_all_models %>%
  separate(Model, into = c("Base_Model", "Method"), sep = " ", extra = "merge") %>%
  pivot_wider(names_from = Method, values_from = c(Brier_Mean, CI_Lower, CI_Upper)) %>%
  mutate(
    `Platt Brier`    = round(Brier_Mean_Platt, 3),
    `Platt CI`       = paste0("(", round(CI_Lower_Platt, 3), ", ", round(CI_Upper_Platt, 3), ")"),
    `Isotonic Brier` = round(Brier_Mean_Isotonic, 3),
    `Isotonic CI`    = paste0("(", round(CI_Lower_Isotonic, 3), ", ", round(CI_Upper_Isotonic, 3), ")"),
    `Original Brier` = round(Brier_Mean_Original, 3),
    `Original CI`    = paste0("(", round(CI_Lower_Original, 3), ", ", round(CI_Upper_Original, 3), ")"),
    `Better Method`  = case_when(
      is.na(Brier_Mean_Platt) | is.na(Brier_Mean_Isotonic) ~ "Tie",
      Brier_Mean_Platt < Brier_Mean_Isotonic               ~ "Platt",
      Brier_Mean_Isotonic < Brier_Mean_Platt               ~ "Isotonic",
      TRUE                                                 ~ "Tie"
    ),
    Improvement = round(abs(Brier_Mean_Platt - Brier_Mean_Isotonic), 3)
  ) %>%
  select(Model = Base_Model,
         `Original Brier`, `Original CI`,
         `Platt Brier`,    `Platt CI`,
         `Isotonic Brier`, `Isotonic CI`,
         `Better Method`,  Improvement)

print(brier_summary_table)


# ===============================================================
# 📊 Calibration Curves Visualization
# ===============================================================

calibration_all <- bind_rows(map(recalibration_results, "calibration_data"))

fig_calibration_all <- ggplot(calibration_all,
                              aes(x = mean_pred, y = mean_obs, color = Model)) +
  geom_point(size = 2) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  facet_wrap(~ Base_Model, scales = "free") +
  labs(
    title = "Calibration Curves Across Models (Original vs Recalibrated)",
    x = "Mean Predicted Probability",
    y = "Mean Observed Outcome"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", legend.title = element_text(face = "bold"))

print(fig_calibration_all)

# ===============================================================
# 🧮 Hosmer–Lemeshow Calibration Verification
# ===============================================================

perform_hl_verification_all <- function(original_probs, platt_probs, iso_probs, true_labels, model_name) {
  g <- min(10, max(4, floor(length(true_labels) / 50)))  # ~50 samples/bin
  if (g < 2) {
    return(tibble(Model = model_name, Best_Method = "Insufficient data"))
  }
  
  hl_original <- safe_hoslem(true_labels, original_probs, g)
  hl_platt    <- safe_hoslem(true_labels, platt_probs, g)
  hl_iso      <- safe_hoslem(true_labels, iso_probs, g)
  
  pvals <- c(Original = hl_original$p.value, Platt = hl_platt$p.value, Isotonic = hl_iso$p.value)
  best_method <- names(which.max(pvals))
  
  tibble(
    Model = model_name,
    HL_Original_p = round(hl_original$p.value, 4),
    HL_Original_X2 = round(as.numeric(hl_original$statistic), 3),
    HL_Original_Status = interpret_calibration(hl_original$p.value),
    HL_Platt_p = round(hl_platt$p.value, 4),
    HL_Platt_X2 = round(as.numeric(hl_platt$statistic), 3),
    HL_Platt_Status = interpret_calibration(hl_platt$p.value),
    HL_Isotonic_p = round(hl_iso$p.value, 4),
    HL_Isotonic_X2 = round(as.numeric(hl_iso$statistic), 3),
    HL_Isotonic_Status = interpret_calibration(hl_iso$p.value),
    Best_Method = best_method
  )
}

# ===============================================================
# 🧩 Apply HL Verification Across All Models
# ===============================================================

hl_test_table <- map_dfr(names(train_all_models), function(m) {
  probs <- tryCatch({
    predict(train_all_models[[m]], datasets$original$test, type = "prob")[, "MetSyn"]
  }, error = function(e) {
    message("⚠️ Skipping model ", m, ": prediction failed (", e$message, ")")
    return(rep(NA, length(labels_bin)))
  })
  
  if (all(is.na(probs))) {
    return(tibble(Model = m, Best_Method = "Unavailable"))
  }
  
  result <- recalibrate_and_plot(probs, labels_bin, model_name = m)
  
  if (!is.null(result$platt_model) && !is.null(result$iso_model)) {
    platt_probs <- predict(result$platt_model, type = "response")
    iso_probs <- fitted(result$iso_model)
    perform_hl_verification_all(probs, platt_probs, iso_probs, labels_bin, m)
  } else {
    tibble(Model = m, Best_Method = "Unavailable")
  }
})

# Rank models by best calibration
hl_test_table <- hl_test_table %>%
  mutate(Best_p = pmax(HL_Original_p, HL_Platt_p, HL_Isotonic_p, na.rm = TRUE)) %>%
  arrange(desc(Best_p))

print(hl_test_table)

# ===============================================================
# 🎨 Visualization: HL Test p-values by Calibration Method
# ===============================================================

hl_test_table %>%
  pivot_longer(cols = c(HL_Original_p, HL_Platt_p, HL_Isotonic_p),
               names_to = "Calibration",
               values_to = "p_value") %>%
  filter(!is.na(p_value)) %>%  # Remove NA values
  ggplot(aes(x = Model, y = p_value, fill = Calibration)) +
  geom_col(position = position_dodge()) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
  labs(
    title = "Hosmer–Lemeshow Calibration p-values by Model",
    subtitle = "Red line indicates acceptable calibration (p > 0.05)",
    y = "p-value", x = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Create the plot and assign it to hl_plot
hl_plot <- hl_test_table %>%
  pivot_longer(cols = c(HL_Original_p, HL_Platt_p, HL_Isotonic_p),
               names_to = "Calibration",
               values_to = "p_value") %>%
  filter(!is.na(p_value)) %>%  # Remove NA values
  ggplot(aes(x = Model, y = p_value, fill = Calibration)) +
  geom_col(position = position_dodge()) +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
  labs(
    title = "Hosmer–Lemeshow Calibration p-values by Model",
    subtitle = "Red line indicates acceptable calibration (p > 0.05)",
    y = "p-value", x = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save as high-resolution PNG
ggsave(
  filename = "Figure_HL_Calibration_pvalues.png",
  plot = hl_plot,
  width = 10,
  height = 6,
  dpi = 300   # publication quality
)


## ===============================================================
# 📊 DCA WITH ISOTONIC-CALIBRATED PROBABILITIES ONLY
# ===============================================================

# Load required packages
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)

# Function to get isotonic-calibrated probabilities
get_isotonic_probabilities <- function(model, test_data) {
  # Get original probabilities
  probs_original <- predict(model, test_data, type = "prob")[, "MetSyn"]
  
  # Apply isotonic calibration
  outcome_binary <- ifelse(test_data$Metabolic_Syndrome == "MetSyn", 1, 0)
  
  if (length(unique(probs_original)) > 3) {
    iso_model <- isoreg(probs_original, outcome_binary)
    probs_calibrated <- fitted(iso_model)
    # Clip probabilities to avoid 0 or 1 boundaries
    probs_calibrated <- pmin(pmax(probs_calibrated, 0.01), 0.99)
    return(probs_calibrated)
  } else {
    # Return original if calibration fails
    return(probs_original)
  }
}

# DCA with isotonic-calibrated probabilities
perform_dca_isotonic <- function(all_models, test_data, outcome_var = "Metabolic_Syndrome", 
                                 probability_thresholds = seq(0.01, 0.5, by = 0.01)) {
  
  cat("=== DCA WITH ISOTONIC-CALIBRATED PROBABILITIES ===\n")
  
  # Extract models from the nested structure
  models <- all_models$original
  cat("Models analyzed:", names(models), "\n")
  
  # Convert outcome to binary (0/1)
  outcome_binary <- ifelse(test_data[[outcome_var]] == "MetSyn", 1, 0)
  cat("Test set: Cases =", sum(outcome_binary), "Controls =", sum(!outcome_binary), "\n")
  
  n <- length(outcome_binary)
  prevalence <- mean(outcome_binary)
  
  dca_results <- data.frame()
  
  # Calculate for each threshold
  for (threshold in probability_thresholds) {
    # Calculate net benefit for reference strategies
    net_benefit_all <- prevalence - (1 - prevalence) * (threshold / (1 - threshold))
    net_benefit_none <- 0
    
    # Add reference strategies
    dca_results <- rbind(dca_results,
                         data.frame(threshold = threshold, model = "Treat All", net_benefit = net_benefit_all),
                         data.frame(threshold = threshold, model = "Treat None", net_benefit = net_benefit_none)
    )
    
    # Calculate net benefit for each model using ISOTONIC-CALIBRATED probabilities
    for (model_name in names(models)) {
      tryCatch({
        model <- models[[model_name]]
        
        # Use isotonic-calibrated probabilities
        probs <- get_isotonic_probabilities(model, test_data)
        
        # Calculate classification metrics
        predicted_class <- ifelse(probs >= threshold, 1, 0)
        tp <- sum(predicted_class == 1 & outcome_binary == 1)
        fp <- sum(predicted_class == 1 & outcome_binary == 0)
        
        net_benefit <- (tp / n) - (fp / n) * (threshold / (1 - threshold))
        
        dca_results <- rbind(dca_results,
                             data.frame(threshold = threshold, model = model_name, net_benefit = net_benefit)
        )
        
      }, error = function(e) {
        # Skip models that fail
        cat("⚠️ Skipping", model_name, ":", e$message, "\n")
      })
    }
  }
  
  return(dca_results)
}

# Publication-quality plotting function
create_isotonic_dca_plot <- function(dca_results, title = "Decision Curve Analysis") {
  
  # Define distinct, publication-friendly colors for models
  model_colors <- c(
    "Treat All"     = "#000000",  # Black (reference)
    "Treat None"    = "#7f7f7f",  # Dark gray (reference)
    
    "logistic"      = "#1f77b4",  # Blue
    "svm_linear"    = "#17becf",  # Teal
    "svm_radial"    = "#2ca02c",  # Green
    
    "random_forest" = "#e41a1c",  # Red
    "xgboost"       = "#ff7f00",  # Orange
    "lightgbm"      = "#984ea3",  # Purple
    
    "naive_bayes"   = "#f781bf",  # Pink
    "decision_tree" = "#a65628",  # Brown
    "knn"           = "#dede00"   # Yellow / Gold (distinct from green)
  )
  
  # Create line types
  dca_results <- dca_results %>%
    mutate(
      line_type = ifelse(model %in% c("Treat All", "Treat None"), "dashed", "solid"),
      model = factor(model, levels = c("Treat All", "Treat None", names(model_colors)[3:11]))
    )
  
  # Create the publication-quality plot
  dca_plot <- ggplot(dca_results, aes(x = threshold, y = net_benefit, color = model, linetype = line_type)) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(values = model_colors) +
    scale_linetype_manual(values = c("dashed" = "dashed", "solid" = "solid"), guide = "none") +
    labs(
      title = title,
      x = "Probability Threshold",
      y = "Net Benefit",
      color = "Model"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      panel.grid.minor = element_blank(),
      legend.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 12),
      axis.title = element_text(face = "bold")
    ) +
    guides(color = guide_legend(nrow = 3, byrow = TRUE)) +
    scale_x_continuous(limits = c(0, 0.5), expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.1)))
  
  return(dca_plot)
}

# Clinical impact analysis
calculate_clinical_impact <- function(dca_results, thresholds = c(0.1, 0.2, 0.3)) {
  
  impact_data <- data.frame()
  
  for (threshold in thresholds) {
    threshold_data <- dca_results %>%
      filter(abs(threshold - !!threshold) < 0.005) %>%
      group_by(model) %>%
      summarise(net_benefit = mean(net_benefit), .groups = "drop") %>%
      mutate(Threshold = threshold)
    
    impact_data <- rbind(impact_data, threshold_data)
  }
  
  # Rank models at each threshold
  impact_summary <- impact_data %>%
    group_by(Threshold) %>%
    mutate(
      Rank = rank(-net_benefit),
      Net_Benefit = round(net_benefit, 4)
    ) %>%
    arrange(Threshold, Rank) %>%
    select(Threshold, Model = model, Net_Benefit, Rank)
  
  return(impact_summary)
}

# ===============================================================
# 🚀 EXECUTE DCA WITH ISOTONIC-CALIBRATED PROBABILITIES
# ===============================================================

cat("==================================================\n")
cat("DCA WITH ISOTONIC-CALIBRATED PROBABILITIES\n")
cat("==================================================\n\n")

# Run DCA with isotonic-calibrated probabilities
dca_isotonic_results <- perform_dca_isotonic(
  all_models = all_models,
  test_data = datasets$original$test,
  probability_thresholds = seq(0.01, 0.4, by = 0.01)
)

cat("✅ DCA calculation completed successfully!\n")

# Create publication-quality plot
dca_plot <- create_isotonic_dca_plot(
  dca_isotonic_results,
  title = "Decision Curve Analysis of Isotonic-Calibrated Models"
)

# Display the plot
print(dca_plot)

# Save high-quality figures
ggsave(
  filename = "Figure4_Decision_Curve_Analysis_Isotonic.tiff",
  plot = dca_plot,
  width = 10,
  height = 8,
  dpi = 300,
  compression = "lzw"
)

ggsave(
  filename = "Figure4_Decision_Curve_Analysis_Isotonic.png",
  plot = dca_plot,
  width = 10,
  height = 8,
  dpi = 300
)

cat("✅ DCA plots saved as TIFF and PNG files\n")

# Calculate clinical impact
clinical_impact <- calculate_clinical_impact(dca_isotonic_results)


# Load patchwork if not already loaded
library(patchwork)

# Combine the two plots side by side with labels
figure_5 <- dca_plot + dca_plot_2 +
  plot_annotation(
    title = "Figure 5. Decision Curve Analysis",
    subtitle = "A: Combined decision curve of the calibrated models\nB: Decision curves for predicting Metabolic Syndrome",
    tag_levels = "A"
  )

# Display the combined figure
print(figure_5)

# Save as high-resolution PNG
ggsave("Figure_5_Decision_Curve_Combined.png",
       plot = figure_5,
       width = 12,
       height = 6,
       dpi = 1200)

cat("\n==================================================\n")
cat("CLINICAL IMPACT AT KEY THRESHOLDS\n")
cat("==================================================\n\n")

print(clinical_impact)

# Load libraries
library(dplyr)
library(ggplot2)

# Recreate your dataset
clinical_impact <- tribble(
  ~Threshold, ~Model, ~Net_Benefit, ~Rank,
  0.10, "Random Forest", 0.163, 1,
  0.10, "Logistic", 0.161, 2,
  0.10, "Svm Linear", 0.159, 3,
  0.10, "Naive Bayes", 0.158, 4,
  0.10, "Svm Radial", 0.158, 4,
  0.10, "Xgboost", 0.151, 6,
  0.10, "Lightgbm", 0.148, 7,
  0.10, "Decision Tree", 0.147, 8,
  0.10, "Treat All", 0.140, 9,
  0.10, "Knn", 0.058, 10,
  0.10, "Treat None", 0.000, 11,
  0.20, "Logistic", 0.123, 1,
  0.20, "Svm Linear", 0.120, 2,
  0.20, "Random Forest", 0.118, 3,
  0.20, "Svm Radial", 0.115, 4,
  0.20, "Decision Tree", 0.112, 5,
  0.20, "Naive Bayes", 0.103, 6,
  0.20, "Xgboost", 0.103, 7,
  0.20, "Lightgbm", 0.095, 8,
  0.20, "Knn", 0.037, 9,
  0.20, "Treat All", 0.032, 10,
  0.20, "Treat None", 0.000, 11,
  0.30, "Logistic", 0.102, 1,
  0.30, "Svm Radial", 0.089, 2,
  0.30, "Svm Linear", 0.077, 3,
  0.30, "Naive Bayes", 0.072, 4,
  0.30, "Xgboost", 0.072, 4,
  0.30, "Random Forest", 0.071, 6,
  0.30, "Decision Tree", 0.068, 7,
  0.30, "Lightgbm", 0.060, 8,
  0.30, "Knn", 0.011, 9,
  0.30, "Treat None", 0.000, 10,
  0.30, "Treat All", -0.106, 11
)

# Clean up model names
clinical_impact_clean <- clinical_impact %>%
  mutate(Model = gsub("_", " ", Model) %>% tools::toTitleCase())

# Create the plot
dca_plot_2 <- ggplot(clinical_impact_clean, aes(x = Threshold, y = Net_Benefit, color = Model, group = Model)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Decision Curve Analysis",
    subtitle = "Net Benefit of ML Models for Metabolic Syndrome Prediction",
    x = "Threshold Probability",
    y = "Net Benefit"
  ) +
  theme(
    legend.position = "right",
    legend.title = element_blank(),
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 10)
  )

# Save as PDF
ggsave("decision_curve_analysis.pdf", plot = dca_plot_2, width = 10, height = 6, dpi = 300)


# Identify best performing models
best_models <- clinical_impact %>%
  group_by(Threshold) %>%
  filter(Rank == 1) %>%
  ungroup()

cat("\n==================================================\n")
cat("BEST PERFORMING MODELS BY THRESHOLD\n")
cat("==================================================\n\n")

print(best_models)

# Summary statistics
cat("\n==================================================\n")
cat("DCA SUMMARY\n")
cat("==================================================\n\n")

cat("Probability thresholds analyzed: 0.01 to 0.40\n")
cat("Number of models:", length(unique(dca_isotonic_results$model)) - 2, "ML models + 2 reference strategies\n")
cat("Test set size:", nrow(datasets$original$test), "patients\n")
cat("Metabolic syndrome prevalence:", round(mean(datasets$original$test$Metabolic_Syndrome == "MetSyn"), 3), "\n")

cat("\n🎯 DCA WITH ISOTONIC-CALIBRATED PROBABILITIES COMPLETED!\n")

