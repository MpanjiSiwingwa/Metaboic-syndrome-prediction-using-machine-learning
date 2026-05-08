# ============================================================
# 📊 METABOLIC SYNDROME PREDICTION PIPELINE
# ============================================================
# Version: 11.2
# Last Updated: 07 May 2026
# Author: Mpanji Siwingwa
# ============================================================

# ============================================================
# 1. LOAD REQUIRED LIBRARIES
# ============================================================

required_packages <- c(
  # Core data handling & visualization
  "tidyverse", "dplyr", "tidyr", "purrr", "tibble", "ggplot2", "viridis",
  
  # Machine learning & modeling
  "tidymodels", "caret", "recipes", "glmnet", "randomForest",
  "xgboost", "lightgbm", "e1071", "rpart", "kknn",
  
  # Feature selection & interpretability
  "Boruta", "fastshap",
  
  # Model evaluation & clinical utility
  "pROC", "PRROC", "rmda", "ResourceSelection", "mgcv", "car",
  
  # Reporting & outputs
  "flextable", "officer",
  
  # Additional utilities
  "ppcor", "patchwork"
)

# Install missing packages
installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed) {
    install.packages(pkg, dependencies = TRUE)
  }
}

# Load all packages invisibly
invisible(lapply(required_packages, library, character.only = TRUE))



RANDOM_SEED <- 123
set.seed(RANDOM_SEED)

setwd('/Users/mpanjisiwingwa/Library/Mobile Documents/com~apple~CloudDocs/School/PhD/Articles Publication/Finals/Week_144')


# ===============================================================
# 2. DATA PREPARATION AND CLEANING
# ===============================================================

## 2.1 Load and Inspect Data
cat("Loading data...\n")
df <- read.csv("MetSyn_dataset_Week_144.csv")
df2 <- df # copy for correlation analysis

## 2.2 Data Cleaning
# (drop redundant clinical variables)
df <- df %>%
  dplyr::select(
    -`Blood_Sugar..mmol.L.`,
    -`Triglycerides..mmol.L.`,
    -`Cholesterol_HDL_.mmol.L.`,
    -`BMI.kg.m2.`,
    -`Bp_Systolic..mmHg.`,
    -`Bp_Diastolic..mmHg.`,
    -`Waist_Circumference.cm.`,
    -`Hip_Circumference.cm.`
  )


## 2.3 Feature Engineering
# (rename columns)
df <- df %>%
  rename(
    Alcohol_Consumption = Alcohol_Consuption,
    Regimen_Type        = Drug_Code
  )

# (create log viral load)
df <- df %>%
  mutate(Log_Viral_Load = log10(`Viral_Load.cp.ml.` + 1)) %>%
  dplyr::select(-`Viral_Load.cp.ml.`)

# 2.4 Additional Cleaning
# (trim white space, harmonize Event_Name)
cols_to_clean <- c("Regimen_Type", "Sex", "Diabetes_Mellitus_status",
                   "Tobbacco_Use", "Alcohol_Consumption", "Educational_Level",
                   "Event_Name")

df[cols_to_clean] <- lapply(
  df[cols_to_clean],
  function(x) if (is.character(x)) trimws(x) else x
)

df$Event_Name[df$Event_Name == "Week_ 144"] <- "week_144"

df <- df %>%
  filter(Event_Name != "Baseline") %>%
  dplyr::select(-`Event_Name`)

# 🔎 Check missing values after cleaning
# Count total missing values
sum(is.na(df))

# Count missing values per column
colSums(is.na(df))

# Optional: show percentage missing per column
round(colSums(is.na(df)) / nrow(df) * 100, 2)


## 2.5 Handling Missing Data (Complete-case analysis)
# (drop rows with NA, recode outcome)
df_complete <- df[complete.cases(df), ]

# Recode the outcome factor
df_complete$Metabolic_Syndrome <- factor(
  df_complete$Metabolic_Syndrome,
  levels = c("No MetSyn", "MetSyn"),
  labels = c("No", "Yes")
)

# Count rows before and after complete-case analysis
n_before <- nrow(df)
n_after  <- nrow(df_complete)

rows_dropped <- n_before - n_after
pct_dropped  <- round(rows_dropped / n_before * 100, 2)

cat("Dropped", rows_dropped, "rows (", pct_dropped, "%) due to missing values.\n")

## 2.7 Correlation Analysis (Waist vs Hip)
# (Pearson correlation, CI, p-value, VIF, scatter plot)

cat("Running correlation analysis (Waist vs Hip)...\n")

# --- Safety check for variables ---
if (!all(c("Waist_Circumference.cm.", "Hip_Circumference.cm.") %in% names(df2))) {
  stop("Waist or Hip circumference variable not found in df2")
}

# --- Clean character columns safely ---
expected_cols <- c("Sex", "Diabetes_Mellitus_status",
                   "Tobbacco_Use", "Alcohol_Consuption",
                   "Educational_Level", "Event_Name")

# Keep only those that exist in df2
cols_to_clean_df2 <- intersect(expected_cols, names(df2))

# Warn if some expected columns are missing
missing_cols <- setdiff(expected_cols, names(df2))
if (length(missing_cols) > 0) {
  cat("Warning: These columns were not found in df2:", 
      paste(missing_cols, collapse = ", "), "\n")
}

# Clean only the existing columns
if (length(cols_to_clean_df2) > 0) {
  df2[cols_to_clean_df2] <- lapply(
    df2[cols_to_clean_df2],
    function(x) if (is.character(x)) trimws(x) else x
  )
}

# --- Harmonize Event_Name if present ---
if ("Event_Name" %in% names(df2)) {
  df2$Event_Name[df2$Event_Name == "Week_ 144"] <- "week_144"
  df2 <- df2 %>%
    dplyr::filter(Event_Name != "Baseline") %>%
    dplyr::select(-Event_Name)
}

# --- Pearson Correlation ---
cor_data <- df2 %>%
  dplyr::select(Waist_Circumference.cm., Hip_Circumference.cm.) %>%
  tidyr::drop_na()

cat("N rows for correlation analysis:", nrow(cor_data), "\n")
cat("Dropped", nrow(df2) - nrow(cor_data), "rows due to missing values\n")

cor_test <- cor.test(
  cor_data$Waist_Circumference.cm.,
  cor_data$Hip_Circumference.cm.,
  method = "pearson"
)

cor_value <- round(cor_test$estimate, 3)
p_value   <- signif(cor_test$p.value, 3)
ci_lower  <- round(cor_test$conf.int[1], 3)
ci_upper  <- round(cor_test$conf.int[2], 3)

cat("Pearson correlation (r):", cor_value, "\n")
cat("95% CI:", ci_lower, "to", ci_upper, "\n")
cat("p-value:", p_value, "\n")

correlation_table <- data.frame(
  Variable_1     = "Waist Circumference",
  Variable_2     = "Hip Circumference",
  Correlation_r  = cor_value,
  CI_Lower       = ci_lower,
  CI_Upper       = ci_upper,
  p_value        = p_value
)

print(correlation_table)

# --- Save correlation table to Word ---
library(flextable)
library(officer)

ft_cor <- flextable(correlation_table) %>%
  autofit() %>%
  set_caption("Supplementary Table: Correlation between Waist and Hip Circumference")

doc <- read_docx()
doc <- body_add_flextable(doc, value = ft_cor)
print(doc, target = "Table_Correlation_Waist_Hip.docx")

# --- Recode outcome for VIF and partial correlation ---
df2$Metabolic_Syndrome <- factor(
  ifelse(df2$Metabolic_Syndrome == "MetSyn", "MetSyn", "NoMetSyn"),
  levels = c("NoMetSyn", "MetSyn")
)

cat("Metabolic Syndrome distribution:\n")
print(table(df2$Metabolic_Syndrome))

# --- VIF for Waist and Hip ---
vif_model_waist_hip <- glm(
  Metabolic_Syndrome ~ Waist_Circumference.cm. + Hip_Circumference.cm.,
  data   = df2,
  family = binomial
)

vif_values_waist_hip <- round(vif(vif_model_waist_hip), 3)
print(vif_values_waist_hip)

vif_table_waist_hip <- data.frame(
  Variable = names(vif_values_waist_hip),
  VIF      = vif_values_waist_hip
)

print(vif_table_waist_hip)

# --- Scatter Plot: Waist vs Hip ---

p_cor <- ggplot(cor_data, aes(x = Waist_Circumference.cm., y = Hip_Circumference.cm.)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  theme_minimal(base_size = 14) +
  labs(
    title = paste0("Correlation between Waist and Hip Circumference (r = ", cor_value,
                   ", 95% CI: ", ci_lower, "–", ci_upper, ")"),
    x     = "Waist Circumference (cm)",
    y     = "Hip Circumference (cm)"
  )

ggsave("Figure_Correlation_Waist_Hip.png", plot = p_cor, width = 8, height = 6, dpi = 300)



## 2.8 Partial Correlation Analysis
# (Hip vs MetS controlling for Waist)

cat("Running partial correlation analysis (Hip vs MetS controlling for Waist)...\n")

met_syn_numeric <- ifelse(df2$Metabolic_Syndrome == "MetSyn", 1, 0)

pcor_data <- df2 %>%
  dplyr::select(Hip_Circumference.cm., Waist_Circumference.cm.) %>%
  dplyr::mutate(MetSyn = met_syn_numeric) %>%
  na.omit()

pcor_result <- pcor.test(
  pcor_data$Hip_Circumference.cm.,
  pcor_data$MetSyn,
  pcor_data$Waist_Circumference.cm.
)

pcor_value <- round(pcor_result$estimate, 3)
pcor_p     <- signif(pcor_result$p.value, 3)

cat("Partial correlation (Hip vs MetS controlling for Waist):", pcor_value, "\n")
cat("p-value:", pcor_p, "\n")

partial_table <- data.frame(
  Variable_1       = "Hip Circumference",
  Variable_2       = "Metabolic Syndrome",
  Control_Variable = "Waist Circumference",
  Partial_r        = pcor_value,
  p_value          = pcor_p
)

print(partial_table)

ft_partial <- flextable(partial_table) %>%
  autofit() %>%
  set_caption("Supplementary Table: Partial correlation of Hip Circumference with MetS controlling for Waist Circumference")

doc <- body_add_flextable(doc, value = ft_partial)
print(doc, target = "Table_PartialCorrelation_Hip_MetSyn.docx")

# ===============================================================
# 3. Feature Selection
# ===============================================================

metabolic_data <- df_complete   # use the completed dataset

## 3.1 Random Forest Feature Importance
# (importance scores, threshold, plot)
cat("Running Random Forest importance screening...\n")
predictor_vars <- setdiff(names(metabolic_data), c("Metabolic_Syndrome","Trial_number"))
x <- metabolic_data[, predictor_vars]
y <- as.factor(metabolic_data$Metabolic_Syndrome)
rf_model <- randomForest::randomForest(
  x = x,
  y = y,
  importance = TRUE,
  ntree = 500,
  strata = y,
  sampsize = rep(min(table(y)), 2)
)
importance_scores <- randomForest::importance(rf_model, type = 2)
rf_importance <- data.frame(
  Variable = rownames(importance_scores),
  Importance = importance_scores[,1]
) %>%
  arrange(desc(Importance))
importance_threshold <- max(rf_importance$Importance) * 0.1
rf_importance <- rf_importance %>%
  mutate(Selected_RF = ifelse(Importance >= importance_threshold, "Selected","Not Selected"))
rf_selected <- rf_importance %>%
  filter(Selected_RF == "Selected") %>%
  pull(Variable)

cat("RF selected variables:", paste(rf_selected, collapse=", "), "\n")

# Plotting Random Forest Importance

p_rf <- ggplot(rf_importance, aes(x = reorder(Variable, Importance), y = Importance, fill = Selected_RF)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("Selected" = "steelblue", "Not Selected" = "grey70")) +
  labs(title = "Figure S1: Random Forest Feature Importance (Mean Decrease in Gini)",
       x = "Features", y = "Importance") +
  theme_minimal()

# Save as Figure S5a
ggsave("Figure_S1_RandomForest.png", plot = p_rf, width = 8, height = 6, dpi = 300)

## 3.2 Boruta Feature Selection
# (confirmed variables, plot)

cat("Running Boruta feature selection...\n")

# Ensure reproducibility
set.seed(123)

# Prepare dataset with RF-selected variables + outcome
boruta_data <- metabolic_data %>%
  dplyr::select(all_of(rf_selected), Metabolic_Syndrome)

# Run Boruta with sufficient iterations
boruta_model <- Boruta(
  Metabolic_Syndrome ~ .,
  data = boruta_data,
  doTrace = 0,
  maxRuns = 200
)

# Finalize Tentative features
boruta_model_fixed <- TentativeRoughFix(boruta_model)

# Extract results
boruta_results <- attStats(boruta_model_fixed)

boruta_table <- data.frame(
  Variable   = rownames(boruta_results),
  Importance = boruta_results$meanImp,
  Decision   = boruta_results$decision
) %>%
  arrange(desc(Importance))

# Get confirmed variables after rough fix
boruta_selected <- boruta_table %>%
  filter(Decision == "Confirmed") %>%
  pull(Variable)

cat("Boruta confirmed variables after TentativeRoughFix:", 
    paste(boruta_selected, collapse = ", "), "\n")

# Boruta Feature Selection Plot

# Convert Boruta results into a clean data frame
boruta_plot_df <- boruta_table %>%
  mutate(Decision = factor(Decision, levels = c("Confirmed", "Tentative", "Rejected")))

# Plot Boruta importance with decisions highlighted
ggplot(boruta_plot_df, aes(x = reorder(Variable, Importance), y = Importance, fill = Decision)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("Confirmed" = "forestgreen",
                               "Tentative" = "gold",
                               "Rejected" = "firebrick")) +
  labs(title = "Figure S2: Boruta Feature Selection Results",
       x = "Features", y = "Mean Importance") +
  theme_minimal()

p_boruta <- ggplot(boruta_plot_df, aes(x = reorder(Variable, Importance), y = Importance, fill = Decision)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("Confirmed" = "forestgreen",
                               "Tentative" = "gold",
                               "Rejected" = "firebrick")) +
  labs(title = "Figure S2: Boruta Feature Selection Results",
       x = "Features", y = "Mean Importance") +
  theme_minimal()

# Save as Figure S5b
ggsave("Figure_S2_Boruta.png", plot = p_boruta, width = 8, height = 6, dpi = 300)

## 3.3 SHAP Feature Importance
# (SHAP values, plot)

cat("Computing SHAP importance...\n")

shap_model <- randomForest::randomForest(
  x = metabolic_data[, boruta_selected],
  y = metabolic_data$Metabolic_Syndrome,
  ntree = 500
)

pred_fun <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}

shap_values <- fastshap::explain(
  shap_model,
  X = metabolic_data[, boruta_selected],
  pred_wrapper = pred_fun,
  nsim = 50
)

shap_importance <- data.frame(
  Variable = colnames(shap_values),
  SHAP_Importance = colMeans(abs(shap_values))
) %>%
  arrange(desc(SHAP_Importance))

figure_5 <- ggplot(shap_importance, aes(x = reorder(Variable, SHAP_Importance), y = SHAP_Importance)) +
  geom_bar(stat = "identity", fill = "#0072B2") +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(title = "Figure 5. SHAP Feature Importance",
       x = "Features", y = "Mean |SHAP Value|")
ggsave("Figure_5_SHAP_Importance.png", plot = figure_5, width = 10, height = 8, dpi = 300)

## 3.4 Final Selected Data

selected_data <- metabolic_data %>%
  dplyr::select(Metabolic_Syndrome, all_of(boruta_selected))

## 3.5 Multicollinearity Check (VIF)
# (Boruta predictors, VIF table export)

cat("Checking multicollinearity (Boruta-selected predictors)...\n")

vif_data_boruta <- metabolic_data %>%
  dplyr::select(dplyr::all_of(boruta_selected), Metabolic_Syndrome)

vif_data_boruta$Metabolic_Syndrome_numeric <- ifelse(vif_data_boruta$Metabolic_Syndrome == "Yes", 1, 0)
vif_data_boruta$Alcohol_Consumption        <- as.factor(vif_data_boruta$Alcohol_Consumption)

predictors_boruta <- vif_data_boruta %>%
  dplyr::select(-Metabolic_Syndrome)

vif_model_boruta   <- lm(Metabolic_Syndrome_numeric ~ ., data = predictors_boruta)
vif_results_boruta <- car::vif(vif_model_boruta, type = "terms")

vif_table_boruta <- data.frame(
  Variable = rownames(vif_results_boruta),
  VIF      = round(vif_results_boruta[, 1], 2)
)

print(vif_table_boruta)

ft_vif <- flextable(vif_table_boruta) %>%
  autofit() %>%
  set_caption("Table 2. Variance Inflation Factors (VIF) for Boruta-selected predictors")

doc_vif <- read_docx()
doc_vif <- body_add_flextable(doc_vif, value = ft_vif)
print(doc_vif, target = "Table2_VIF.docx")


## 3.6 Clinical Variable Relationship Diagnostics

cat("Evaluating relationship between CD4 count and viral load...\n")

diagnostic_data <- metabolic_data %>%
  dplyr::select(
    `CD4_count.cells.µl.`,
    `Log_Viral_Load`,
    `Age..yrs.`,
    `Sex`,
    `Alcohol_Consumption`,
    `Metabolic_Syndrome`
  )


### 3.6.1 Non-linear effect of CD4 using GAM
gam_model <- gam(
  Metabolic_Syndrome ~ s(CD4_count.cells.µl.) + Log_Viral_Load +
    Age..yrs. + Sex + Alcohol_Consumption,
  data   = diagnostic_data,
  family = binomial
)

cat("\nGAM model summary:\n")
summary(gam_model)

png("Figure_S3_CD4_GAM_Effect.png", width = 2000, height = 1500, res = 300)
plot(gam_model, select = 1, shade = TRUE,
     main = "Figur S3: Non-linear Effect of CD4 Count on Metabolic Syndrome")
dev.off()


### 3.6.2 Interaction between CD4 and Viral Load

interaction_model <- glm(
  Metabolic_Syndrome ~ Log_Viral_Load * CD4_count.cells.µl. +
    Age..yrs. + Sex + Alcohol_Consumption,
  data   = diagnostic_data,
  family = binomial
)

cat("\nInteraction model summary:\n")
summary(interaction_model)


### 3.6.3 Collinearity between CD4 and Viral Load

vif_model_cd4 <- glm(
  Metabolic_Syndrome ~ CD4_count.cells.µl. + Log_Viral_Load +
    Age..yrs. + Sex + Alcohol_Consumption,
  data   = diagnostic_data,
  family = binomial
)

cat("\nVariance Inflation Factors (CD4 / Viral Load model):\n")
print(round(vif(vif_model_cd4), 2))


# ===============================================================
# 4. DATA PREPROCESSING PIPELINE
# ===============================================================

## 4.1 Variable Standardisation

selected_data <- selected_data %>%
  dplyr::rename(
    Age_years        = Age..yrs.,
    Log10_Viral_Load = Log_Viral_Load,
    Alcohol_Use      = Alcohol_Consumption
  ) %>%
  dplyr::mutate(
    Alcohol_Use = as.factor(Alcohol_Use)
  )

## 4.2 Data Validation Function

validate_data_for_ml <- function(data, data_name = "dataset") {
  cat("\n=== VALIDATING", toupper(data_name), "===\n")
  cat("Dimensions:", dim(data), "\n")
  cat("Missing values:", sum(is.na(data)), "\n")
  if ("Metabolic_Syndrome" %in% names(data)) {
    cat("Class distribution:\n")
    print(table(data$Metabolic_Syndrome))
  }
  nzv_check <- nearZeroVar(data, saveMetrics = TRUE)
  cat("Near-zero variance features:", sum(nzv_check$nzv), "\n")
  cat("===============================\n")
}


## 4.3 Train/Test Split (70/30 stratified)

cat("Splitting data (70/30 stratified)...\n")

set.seed(123)

train_indices <- createDataPartition(
  selected_data$Metabolic_Syndrome,
  p    = 0.7,
  list = FALSE
)

train_set <- selected_data[train_indices, ]
test_set  <- selected_data[-train_indices, ]

cat("Training set size:", nrow(train_set), "\n")
cat("Testing set size:", nrow(test_set), "\n")

write.csv(train_set, "train_set.csv", row.names = FALSE)
write.csv(test_set,  "test_set.csv",  row.names = FALSE)
cat("Train and test sets saved.\n")

train_set$Metabolic_Syndrome <- factor(train_set$Metabolic_Syndrome, levels = c("No", "Yes"))
test_set$Metabolic_Syndrome  <- factor(test_set$Metabolic_Syndrome,  levels = c("No", "Yes"))


## 4.4 Dataset Variants

datasets <- list(
  original = list(
    train = train_set,
    test  = test_set
  )
)

validate_data_for_ml(datasets$original$train, "original training")
validate_data_for_ml(datasets$original$test,  "original testing")


# ===============================================================
# 5. MODEL TRAINING CONFIGURATION
# ===============================================================

## 5.1 Standardize Outcome Levels

cat("=== STANDARDIZING OUTCOME LEVELS ===\n")

datasets$original$train$Metabolic_Syndrome <- factor(
  datasets$original$train$Metabolic_Syndrome, levels = c("No", "Yes")
)
datasets$original$test$Metabolic_Syndrome <- factor(
  datasets$original$test$Metabolic_Syndrome, levels = c("No", "Yes")
)

cat("Outcome levels:", levels(datasets$original$train$Metabolic_Syndrome), "\n")


## 5.2 Training Control Configuration

ctrl_stratified <- caret::trainControl(
  method          = "repeatedcv",   # repeated cross-validation
  number          = 10,             # 10 folds
  repeats         = 5,              # repeat 5 times
  classProbs      = TRUE,           # compute class probabilities
  summaryFunction = twoClassSummary, # gives ROC-AUC, Sensitivity, Specificity
  savePredictions = "final"         # keep final predictions
)


## 5.3 Custom LightGBM Model

caret_lightgbm <- list(
  label      = "LightGBM",
  library    = "lightgbm",
  type       = c("Classification"),
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
    dtrain   <- lgb.Dataset(x_matrix, label = as.numeric(y) - 1)
    model    <- lightgbm::lgb.train(
      params  = list(objective = "binary", num_leaves = param$num_leaves,
                     learning_rate = param$learning_rate, verbose = -1),
      data    = dtrain,
      nrounds = param$nrounds
    )
    result <- list(lgb_model = model, xNames = colnames(x),
                   obsLevels = lev, problemType = "Classification")
    class(result) <- "caretLightGBM"
    return(result)
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    newdata_matrix <- as.matrix(newdata[, modelFit$xNames, drop = FALSE])
    preds <- predict(modelFit$lgb_model, newdata_matrix)
    factor(ifelse(preds > 0.5, modelFit$obsLevels[2], modelFit$obsLevels[1]),
           levels = modelFit$obsLevels)
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    newdata_matrix <- as.matrix(newdata[, modelFit$xNames, drop = FALSE])
    preds <- predict(modelFit$lgb_model, newdata_matrix)
    out   <- cbind(1 - preds, preds)
    colnames(out) <- modelFit$obsLevels
    as.data.frame(out)
  }
)


## 5.4 Model Configurations

make_rf_grid <- function(train_df) {
  p <- ncol(train_df) - 1
  expand.grid(mtry = 1:p)
}

model_configs <- list(
  logistic      = list(method = "glmnet",      grid = expand.grid(alpha = 0.5, lambda = 0.01)),
  svm_linear    = list(method = "svmLinear",   grid = expand.grid(C = c(0.01, 0.1, 1))),
  svm_radial    = list(method = "svmRadial",   grid = expand.grid(sigma = c(0.01, 0.1), C = c(0.5, 1))),
  decision_tree = list(method = "rpart",       grid = NULL),
  random_forest = list(method = "rf",          grid = make_rf_grid(datasets$original$train)),
  knn           = list(method = "kknn",        grid = NULL),
  naive_bayes   = list(method = "naive_bayes", grid = data.frame(laplace = 0, usekernel = FALSE, adjust = 1)),
  lightgbm      = list(method = caret_lightgbm, grid = NULL)
)

## 5.5 Calibration Utility Functions

get_train_test_probs <- function(model, train_data, test_data) {
  train_probs  <- predict(model, train_data, type = "prob")[, "Yes"]
  test_probs   <- predict(model, test_data,  type = "prob")[, "Yes"]
  train_labels <- ifelse(train_data$Metabolic_Syndrome == "Yes", 1, 0)
  test_labels  <- ifelse(test_data$Metabolic_Syndrome  == "Yes", 1, 0)
  list(train_probs = train_probs, test_probs = test_probs,
       train_labels = train_labels, test_labels = test_labels)
}

apply_platt_calibration <- function(train_probs, train_labels, test_probs) {
  platt_model <- glm(train_labels ~ train_probs, family = binomial)
  predict(platt_model, newdata = data.frame(train_probs = test_probs), type = "response")
}

apply_isotonic_calibration <- function(train_probs, train_labels, test_probs) {
  iso_model  <- isoreg(train_probs, train_labels)
  iso_pred   <- approxfun(iso_model$x, iso_model$yf, rule = 2)
  pmin(pmax(iso_pred(test_probs), 0.001), 0.999)
}

# ===============================================================
# 6. UNIFIED MODEL TRAINING PIPELINE
# ===============================================================

## 6.1 Robust Model Training Function

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
  
  tryCatch({
    if (!is.null(tune_grid)) {
      model <- caret::train(
        Metabolic_Syndrome ~ .,
        data = train_data,
        method = method,
        trControl = ctrl_stratified,
        tuneGrid = tune_grid,
        metric = "ROC"
      )
    } else {
      model <- caret::train(
        Metabolic_Syndrome ~ .,
        data = train_data,
        method = method,
        trControl = ctrl_stratified,
        metric = "ROC",
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

## 6.2 Matthews Correlation Coefficient (MCC)

calculate_mcc <- function(cm) {
  if (inherits(cm, "confusionMatrix")) cm <- cm$table
  if (all(dim(cm) == c(2, 2))) {
    TP <- as.numeric(cm[2, 2])
    TN <- as.numeric(cm[1, 1])
    FP <- as.numeric(cm[1, 2])
    FN <- as.numeric(cm[2, 1])
    numerator   <- (TP * TN) - (FP * FN)
    denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if (denominator == 0) return(0)
    return(numerator / denominator)
  }
  return(NA)
}

## 6.3 Train All Models (Original Dataset)

cat("\n=== TRAINING MODELS: ORIGINAL DATASET ONLY ===\n")

library(progressr)
handlers(global = TRUE)  # enable progress handlers

cat("\n=== STARTING MODEL TRAINING ===\n")

with_progress({
  total_steps <- length(names(datasets)) * length(names(model_configs))
  p <- progressor(steps = total_steps)
  
  train_df <- datasets[["original"]]$train
  
  all_models <- list()
  results <- list()
  
  for (model_name in names(model_configs)) {
    config <- model_configs[[model_name]]
    
    # Adjust RF grid dynamically
    if (model_name == "random_forest") {
      config$grid <- make_rf_grid(train_df)
    }
    
    # 🔹 increment progress here
    p(message = paste("Training", model_name, "on original dataset"))
    
    results[[model_name]] <- train_robust_model(
      train_data = train_df,
      method     = config$method,
      tune_grid  = config$grid,
      model_name = paste("original", model_name, sep = "_")
    )
  }
  
  all_models[["original"]] <- results
})

cat("\n=== MODEL TRAINING COMPLETE (ORIGINAL) ===\n")

## 6.4 Metrics Function (ROC, PR AUC, Youden, MCC)

get_cm_metrics_youden_from_probs <- function(pred_probs, true_labels,
                                             label = "Model", conf_level = 0.95) {
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
  
  best_idx <- which.max(metrics[3,])
  best_threshold <- thresholds[best_idx]
  
  predicted <- ifelse(pred_probs >= best_threshold, "Yes", "No")
  actual_label <- ifelse(true_labels == 1, "Yes", "No")
  
  cm <- caret::confusionMatrix(
    factor(predicted, levels = c("No","Yes")),
    factor(actual_label, levels = c("No","Yes"))
  )
  
  mcc_val <- calculate_mcc(cm)
  
  roc_obj <- pROC::roc(true_labels, pred_probs, quiet = TRUE, direction = "<")
  auc_val <- as.numeric(pROC::auc(roc_obj))
  ci_auc <- suppressWarnings(pROC::ci.auc(roc_obj, conf.level = conf_level, method = "delong"))
  
  # --- PR AUC Calculation ---
  pr_obj <- PRROC::pr.curve(
    scores.class0 = pred_probs[true_labels == 1],
    scores.class1 = pred_probs[true_labels == 0],
    curve = FALSE
  )
  auc_pr_val <- pr_obj$auc.integral
  
  tibble::tibble(
    Model = label,
    AUC = round(auc_val,3),
    AUC_Lower = round(ci_auc[1],3),
    AUC_Upper = round(ci_auc[3],3),
    PR_AUC = round(auc_pr_val,3),
    Optimal_Threshold = round(best_threshold,3),
    Sensitivity = round(metrics[1, best_idx],3),
    Specificity = round(metrics[2, best_idx],3),
    PPV = round(cm$byClass["Pos Pred Value"],3),
    NPV = round(cm$byClass["Neg Pred Value"],3),
    Accuracy = round(cm$overall["Accuracy"],3),
    Youden_Index = round(metrics[3, best_idx],3),
    MCC = round(mcc_val,3)
  )
}


## 6.5 XGBoost Standalone Function
train_xgboost_standalone <- function(train_data, test_data, dataset_name = "original") {
  
  cat("\n=== TRAINING XGBOOST (", dataset_name, ") ===\n")
  
  # Build design matrices
  x_train <- model.matrix(Metabolic_Syndrome ~ ., data = train_data)[, -1]
  x_test  <- model.matrix(Metabolic_Syndrome ~ ., data = test_data)[, -1]
  
  # Align columns between train and test
  train_df <- as.data.frame(x_train)
  test_df  <- as.data.frame(x_test)
  
  missing_cols <- setdiff(names(train_df), names(test_df))
  for (col in missing_cols) {
    test_df[[col]] <- 0
  }
  test_df <- test_df[, names(train_df)]  # reorder
  
  # Convert back to matrices
  x_train <- data.matrix(train_df)
  x_test  <- data.matrix(test_df)
  
  # Labels
  y_train <- factor(train_data$Metabolic_Syndrome, levels = c("No","Yes"))
  y_test  <- factor(test_data$Metabolic_Syndrome, levels = c("No","Yes"))
  
  train_label <- as.numeric(y_train) - 1
  test_label  <- as.numeric(y_test) - 1
  
  dtrain <- xgboost::xgb.DMatrix(data = x_train, label = train_label)
  dtest  <- xgboost::xgb.DMatrix(data = x_test,  label = test_label)
  
  # Parameters
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 3,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  # Train model
  model <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100,
    verbose = 0
  )
  
  # Predictions
  probs <- predict(model, dtest)
  
  # Collect results
  results <- get_cm_metrics_youden_from_probs(
    pred_probs = probs,
    true_labels = test_label,
    label = paste(dataset_name, "xgboost", sep = "_")
  )
  
  return(list(model = model, results = results))
}



## 6.6 Run XGBoost (Original Dataset)
xgb_results_list <- list()

xgb_results_list[["original"]] <- train_xgboost_standalone(
  datasets[["original"]]$train,
  datasets[["original"]]$test,
  dataset_name = "original"
)


# ===============================================================
# 7. MODEL EVALUATION
# ===============================================================

## 7.1 Binary Label Helper

get_labels <- function(dataset_name) {
  ifelse(datasets[[dataset_name]]$test$Metabolic_Syndrome == "Yes", 1, 0)
}

## 7.2 Probability Extraction Helper

extract_probabilities <- function(probs_raw, model_name = "model") {
  cat("Class of probs_raw for", model_name, ":", class(probs_raw), "\n")
  
  if (is.data.frame(probs_raw) || is.matrix(probs_raw)) {
    if ("Yes" %in% colnames(probs_raw)) {
      probs_col <- probs_raw[, "Yes"]
      if (is.list(probs_col)) probs_col <- unlist(probs_col)
      return(as.numeric(probs_col))
    } else stop("Column 'Yes' not found")
  }
  if (is.list(probs_raw)) {
    cat("⚠️ List detected in", model_name, "\n")
    if ("Yes" %in% names(probs_raw)) return(as.numeric(unlist(probs_raw[["Yes"]])))
    if (length(probs_raw) >= 2)       return(as.numeric(unlist(probs_raw[[2]])))
    return(as.numeric(unlist(probs_raw)))
  }
  if (is.numeric(probs_raw))                      return(as.numeric(probs_raw))
  if (is.factor(probs_raw) || is.character(probs_raw)) {
    cat("⚠️ Converting class predictions for", model_name, "\n")
    return(ifelse(probs_raw == "Yes", 1, 0))
  }
  stop(paste("Unsupported probability format in", model_name))
}


## 7.3 Evaluate Single Model (Youden Threshold)

evaluate_model_youden <- function(model, test_data, model_name = "model",
                                  outcome_col = "Metabolic_Syndrome") {
  if (is.null(model)) {
    warning(paste("Model is NULL:", model_name))
    return(list(Accuracy = NA, Sensitivity = NA, Specificity = NA,
                PPV = NA, NPV = NA, MCC = NA, AUC = NA, Threshold = NA))
  }
  
  tryCatch({
    cat("\n🔍 Evaluating:", model_name, "\n")
    test_data <- test_data[complete.cases(test_data), ]
    rownames(test_data) <- NULL
    truth <- factor(test_data[[outcome_col]], levels = c("No", "Yes"))
    
    probs_raw <- tryCatch(
      predict(model, newdata = test_data, type = "prob"),
      error = function(e) {
        cat("⚠️ No prob support for", model_name, "- using raw predictions\n")
        predict(model, newdata = test_data)
      }
    )
    
    probs <- extract_probabilities(probs_raw, model_name)
    if (length(probs) != length(truth)) stop("Length mismatch")
    if (any(is.na(probs)))             stop("NA probabilities detected")
    
    roc_obj          <- pROC::roc(truth, probs, levels = c("No", "Yes"), direction = "<")
    youden_threshold <- as.numeric(pROC::coords(roc_obj, x = "best",
                                                best.method = "youden", ret = "threshold"))
    pred_labels      <- ifelse(probs > youden_threshold, "Yes", "No")
    preds            <- factor(pred_labels, levels = c("No", "Yes"))
    cm               <- caret::confusionMatrix(preds, truth, positive = "Yes")
    
    return(list(
      Accuracy    = cm$overall["Accuracy"],
      Sensitivity = cm$byClass["Sensitivity"],
      Specificity = cm$byClass["Specificity"],
      PPV         = cm$byClass["Pos Pred Value"],
      NPV         = cm$byClass["Neg Pred Value"],
      MCC         = calculate_mcc(cm),
      AUC         = as.numeric(pROC::auc(roc_obj)),
      Threshold   = round(youden_threshold, 4)
    ))
  }, error = function(e) {
    cat("❌ FAILED:", model_name, "\n", "Reason:", e$message, "\n")
    return(list(Accuracy = NA, Sensitivity = NA, Specificity = NA,
                PPV = NA, NPV = NA, MCC = NA, AUC = NA, Threshold = NA))
  })
}


## 7.4 Batch Evaluation Across All Datasets

evaluate_all_models_youden <- function(models, datasets) {
  results <- list()
  for (data_name in names(models)) {
    results[[data_name]] <- list()
    test_data <- datasets[[data_name]]$test
    for (model_name in names(models[[data_name]])) {
      full_model_name <- paste(data_name, model_name, sep = "_")
      cat("\n➡️ Evaluating:", full_model_name, "\n")
      results[[data_name]][[model_name]] <- evaluate_model_youden(
        models[[data_name]][[model_name]], test_data, full_model_name
      )
    }
  }
  return(results)
}


## 7.5 Execute Evaluation

cat("\n=== EVALUATING MODELS WITH YOUDEN INDEX THRESHOLD ===\n")
all_results_youden <- evaluate_all_models_youden(
  models = list(original = all_models$original),
  datasets = list(original = datasets$original)
)

# ===============================================================
# 8. RESULTS ANALYSIS AND VISUALIZATION
# ===============================================================

## 8.1 Loop Over Datasets and Models
dataset_names <- names(datasets)

all_metrics <- map_dfr(dataset_names, function(dname) {
  cat("\n📊 Processing dataset:", dname, "\n")
  models     <- all_models[[dname]]
  labels_bin <- get_labels(dname)
  
  map_dfr(names(models), function(m) {
    cat("   🔹 Model:", m, "\n")
    probs <- predict(models[[m]], datasets[[dname]]$test, type = "prob")[, "Yes"]
    get_cm_metrics_youden_from_probs(probs, labels_bin, paste(dname, m, sep = "_"))
  })
})

## 8.2 Add XGBoost Results
xgb_metrics <- dplyr::bind_rows(lapply(xgb_results_list, function(x) x$results)) %>%
  mutate(Dataset = sub("_xgboost", "", Model), Model_Type = "xgboost")

dataset_results <- all_metrics %>%
  separate(Model, into = c("Dataset", "Model_Type"), sep = "_", extra = "merge")

final_results <- bind_rows(dataset_results, xgb_metrics) %>%
  arrange(desc(AUC))

cat("\n=== MASTER METRICS TABLE ===\n")
print(final_results, width = Inf)

cat("\n=== RESULTS: ORIGINAL DATASET ===\n")
original_results <- final_results %>%
  filter(Dataset == "original") %>%
  arrange(desc(AUC))
print(original_results, width = Inf)

cat("\n=== RESULTS BY MODEL TYPE ===\n")
algorithm_tables <- dataset_results %>%
  group_by(Model_Type) %>%
  arrange(desc(AUC), .by_group = TRUE) %>%
  ungroup()
print(algorithm_tables, width = Inf)

# Utility: Safe Probability Extraction

safe_predict_probs <- function(model, newdata) {
  out <- tryCatch({
    probs <- predict(model, newdata, type = "prob")
    
    if (is.data.frame(probs) || is.matrix(probs)) {
      if ("Yes" %in% colnames(probs)) {
        return(as.numeric(probs[, "Yes"]))
      } else {
        return(as.numeric(probs[, 1]))
      }
    } else if (is.numeric(probs)) {
      return(as.numeric(probs))
    } else {
      # fallback: convert class predictions to 0/1
      preds <- predict(model, newdata)
      return(ifelse(preds == "Yes", 1, 0))
    }
  }, error = function(e) {
    cat("⚠️ Probability extraction failed:", e$message, "\n")
    return(rep(NA, nrow(newdata)))
  })
  
  return(out)
}

## 8.3 ROC Curves
library(viridis)

generate_roc_curves <- function(models_list, dataset_list) {
  roc_data <- data.frame()
  for (dname in names(models_list)) {
    labels_bin <- get_labels(dname)
    test_data  <- dataset_list[[dname]]$test
    for (m in names(models_list[[dname]])) {
      model <- models_list[[dname]][[m]]
      if (!is.null(model)) {
        probs <- tryCatch({
          if (m == "xgboost") {
            x_test <- model.matrix(Metabolic_Syndrome ~ ., data = test_data)[, -1]
            predict(model, data.matrix(x_test))
          } else {
            predict(model, test_data, type = "prob")[, "Yes"]
          }
        }, error = function(e) {
          cat("Error in", m, ":", e$message, "\n")
          return(NULL)
        })
        if (!is.null(probs)) {
          roc_obj <- pROC::roc(labels_bin, probs, quiet = TRUE)
          roc_data <- rbind(roc_data, data.frame(
            FPR   = 1 - roc_obj$specificities,
            TPR   = roc_obj$sensitivities,
            Model = paste(dname, m, sep = "_"),
            AUC   = round(pROC::auc(roc_obj), 3)
          ))
        }
      }
    }
  }
  return(roc_data)
}

roc_data <- generate_roc_curves(all_models, datasets)

auc_labels <- roc_data %>%
  group_by(Model) %>%
  summarise(AUC = first(AUC)) %>%
  mutate(label_text = paste0(Model, " (AUC = ", sprintf("%.3f", AUC), ")"))

figure_3 <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    title = "Figure 3: ROC Curves for Metabolic Syndrome Prediction Models",
    subtitle = "Comparison of Machine Learning Algorithms",
    x = "False Positive Rate (1 - Specificity)", 
    y = "True Positive Rate (Sensitivity)",
    caption = "Diagonal line represents random classifier (AUC = 0.5)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = c(0.75, 0.25),
    legend.background = element_rect(fill = "white", color = "gray80"),
    legend.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA, color = "gray70")
  ) +
  scale_color_viridis_d(labels = auc_labels$label_text) +
  coord_equal()

ggsave("Figure_3_ROC_Curves.png", plot = figure_3, width = 10, height = 8, dpi = 300)

## 8.4 Precision-Recall Curves (BMC-ready)
plot_pr_from_probs <- function(pred_probs, true_labels, label = "Model") {
  pr <- PRROC::pr.curve(
    scores.class0 = pred_probs[true_labels == 1],
    scores.class1 = pred_probs[true_labels == 0],
    curve = TRUE
  )
  baseline <- mean(true_labels)
  tibble(
    Recall = pr$curve[, 1],
    Precision = pr$curve[, 2],
    Threshold = pr$curve[, 3],
    AUC_PR = pr$auc.integral,
    Baseline = baseline,
    Model = label
  )
}

pr_data <- map_dfr(dataset_names, function(dname) {
  labels_bin <- get_labels(dname)
  models <- all_models[[dname]]
  map_dfr(names(models), function(m) {
    if (!is.null(models[[m]])) {
      probs <- tryCatch({
        predict(models[[m]], datasets[[dname]]$test, type = "prob")[, "Yes"]
      }, error = function(e) {
        cat("Error with model", m, ":", e$message, "\n")
        return(NULL)
      })
      if (!is.null(probs)) plot_pr_from_probs(probs, labels_bin, paste(dname, m, sep = "_"))
    }
  })
})

auc_pr_labels <- pr_data %>%
  group_by(Model) %>%
  summarise(AUC_PR = first(AUC_PR), Baseline = first(Baseline)) %>%
  mutate(Label = paste0(Model, " (AUC-PR = ", sprintf("%.3f", AUC_PR), ")"))

bmc_colors <- c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
                "#FFFF33", "#A65628", "#F781BF", "#999999")

figure_4 <- ggplot(pr_data, aes(x = Recall, y = Precision, color = Model)) +
  geom_line(linewidth = 1.2) +
  geom_hline(data = auc_pr_labels, aes(yintercept = Baseline, color = Model),
             linetype = "dashed", linewidth = 0.5, alpha = 0.6) +
  labs(
    title = "Figure 4: Precision-Recall Curves for Metabolic Syndrome Prediction",
    subtitle = paste("Comparison of", length(unique(pr_data$Model)), "Machine Learning Models"),
    x = "Recall (Sensitivity)", y = "Precision (PPV)",
    caption = "Dashed lines represent baseline prevalence"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = c(0.75, 0.35),
    legend.background = element_rect(fill = "white", color = "gray70"),
    legend.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = NA, color = "gray50")
  ) +
  scale_color_manual(values = bmc_colors[1:n_distinct(pr_data$Model)],
                     labels = auc_pr_labels$Label) +
  coord_equal()

ggsave("Figure_4_PR_Curves_BMC.tiff", plot = figure_4,
       width = 7.5, height = 6.5, dpi = 300, compression = "lzw", device = "tiff")
ggsave("Figure_4_PR_Curves_BMC.png", plot = figure_4,
       width = 7.5, height = 6.5, dpi = 300, bg = "white")

## 8.5 Build Model Objects
# Build model objects, apply Platt & Isotonic calibration, export tables

results_summary <- dataset_results %>%
  mutate(Dataset    = str_replace_all(Dataset, "\\s+", "_"),
         Dataset    = case_when(Dataset == "feature" ~ "feature_engineered", TRUE ~ Dataset),
         Model_Type = str_replace(Model_Type, "^engineered_", ""))

all_model_objects <- list()

for (i in 1:nrow(results_summary)) {
  dataset_name <- results_summary$Dataset[i]
  model_type   <- results_summary$Model_Type[i]
  model_obj    <- all_models[[dataset_name]][[model_type]]
  
  if (!is.null(model_obj)) {
    test_data  <- datasets[[dataset_name]]$test
    labels_bin <- get_labels(dataset_name)
    probs <- safe_predict_probs(model_obj, test_data)
    model_name <- paste(dataset_name, model_type, sep = "_")
    all_model_objects[[model_name]] <- list(model = model_obj, probs = probs, labels = labels_bin)
  } else {
    cat("⚠️ Skipping", dataset_name, model_type, "- model not found\n")
  }
}

cat("\n=== BUILT MODEL OBJECTS ===\n")
print(names(all_model_objects))


##  8.6 Calibration for All Models

calibration_results <- list()

for (dataset_name in names(all_models)) {
  train_data <- datasets[[dataset_name]]$train
  test_data  <- datasets[[dataset_name]]$test
  
  for (model_name in names(all_models[[dataset_name]])) {
    model <- all_models[[dataset_name]][[model_name]]
    if (is.null(model)) next
    
    cat("Calibrating:", dataset_name, model_name, "\n")
    
    train_probs <- safe_predict_probs(model, train_data)
    test_probs  <- safe_predict_probs(model, test_data)
    
    if (all(is.na(train_probs)) || all(is.na(test_probs))) {
      cat("⚠️ Skipping", model_name, "- no valid probabilities\n")
      next
    }
    
    train_probs <- pmin(pmax(train_probs, 0.001), 0.999)
    test_probs  <- pmin(pmax(test_probs, 0.001), 0.999)
    
    train_labels <- ifelse(train_data$Metabolic_Syndrome == "Yes", 1, 0)
    test_labels  <- ifelse(test_data$Metabolic_Syndrome == "Yes", 1, 0)
    
    platt_probs <- apply_platt_calibration(train_probs, train_labels, test_probs)
    iso_probs   <- suppressWarnings(apply_isotonic_calibration(train_probs, train_labels, test_probs))
    
    model_id <- paste(dataset_name, model_name, sep = "_")
    
    calibration_results[[model_id]] <- bind_rows(
      get_cm_metrics_youden_from_probs(test_probs,  test_labels, paste(model_id, "Raw")),
      get_cm_metrics_youden_from_probs(platt_probs, test_labels, paste(model_id, "Platt")),
      get_cm_metrics_youden_from_probs(iso_probs,   test_labels, paste(model_id, "Isotonic"))
    )
  }
}

calibration_table <- bind_rows(calibration_results)
cat("\n=== CALIBRATION RESULTS ===\n")
print(calibration_table, width = Inf)


# Export selected columns to Word
ft_calibration <- flextable(
  calibration_table %>%
    dplyr::select(Model, AUC, AUC_CI, PR_AUC, Sensitivity, Specificity)
) %>%
  autofit() %>%
  theme_vanilla() %>%
  set_caption("Table X. Calibration results for models (Original dataset)")

doc <- read_docx() %>%
  body_add_par("Calibration Results", style = "heading 1") %>%
  body_add_flextable(ft_calibration)

print(doc, target = "calibration_results_BMC.docx")

# Format Supplementary Table 2
ft_table2 <- flextable(table2) %>%
  autofit() %>%
  width(j = ~., width = 1.5) %>%   # shrink columns if needed
  set_caption("Supplementary Table 2. Expanded performance metrics.")

# Format Table 3
ft_table3 <- flextable(table3) %>%
  autofit() %>%
  width(j = ~., width = 1.5) %>%
  set_caption("Table 3. Summary metrics")

# Build Word document
doc_perf <- read_docx() %>%
  body_add_par("Table 3. Summary metrics", style = "heading 1") %>%
  body_add_flextable(ft_table3) %>%
  body_add_break() %>%
  body_add_par("Supplementary Table 2. Expanded metrics", style = "heading 1") %>%
  body_add_flextable(ft_table2)

# Save Word file
print(doc_perf, target = "Model_Performance_Tables.docx")

# 8.7 Calibration Curves Visualization
# ---------------------------------------
make_calibration_curve <- function(probs, labels, model_name) {
  df <- data.frame(prob = probs, label = labels)
  df$bin <- cut(df$prob, breaks = seq(0, 1, 0.1), include.lowest = TRUE)
  df %>%
    group_by(bin) %>%
    summarise(mean_prob = mean(prob), obs_rate = mean(label), .groups = "drop") %>%
    mutate(Model = model_name)
}

calibration_curve_data <- map_dfr(names(all_model_objects), function(m) {
  make_calibration_curve(all_model_objects[[m]]$probs, all_model_objects[[m]]$labels, m)
})

calibration_plot <- ggplot(calibration_curve_data, aes(x = mean_prob, y = obs_rate, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "Calibration Curves - All Models",
       x = "Predicted Probability", y = "Observed Frequency") +
  theme_minimal()

print(calibration_plot)
ggsave("calibration_curves_all_models.png", plot = calibration_plot, width = 12, height = 8)


# 8.8 Brier Scores + Recalibration (with clipping)
# ----------------------------------
bootstrap_brier_ci <- function(pred_probs, true_labels, n_boot = 1000, conf_level = 0.95) {
  labels_bin   <- as.numeric(true_labels)
  n            <- length(pred_probs)
  if (n == 0) return(list(mean = NA_real_, lower = NA_real_, upper = NA_real_))
  brier_scores <- replicate(n_boot, {
    idx <- sample.int(n, replace = TRUE)
    mean((pred_probs[idx] - labels_bin[idx])^2, na.rm = TRUE)
  })
  alpha <- (1 - conf_level) / 2
  ci    <- quantile(brier_scores, probs = c(alpha, 1 - alpha), na.rm = TRUE)
  list(mean = mean(brier_scores, na.rm = TRUE), lower = ci[1], upper = ci[2])
}

plot_calibration_from_probs <- function(probs, labels, model_name, bins = 5) {
  df     <- data.frame(prob = probs, label = labels)
  df$bin <- cut(df$prob, breaks = seq(0, 1, length.out = bins + 1), include.lowest = TRUE)
  df %>%
    group_by(bin) %>%
    summarise(mean_pred = mean(prob), mean_obs = mean(label), .groups = "drop") %>%
    mutate(Model = model_name)
}

recalibrate_and_plot <- function(pred_probs, true_labels, model_name = "Model",
                                 bins = 5, n_boot = 1000) {
  labels_bin <- as.numeric(true_labels)
  
  # 🔹 Clip probabilities here to avoid 0/1 extremes
  pred_probs <- pmin(pmax(pred_probs, 0.001), 0.999)
  
  if (length(unique(pred_probs)) <= 3) {
    cat("⚠️ Low probability variation for", model_name, "- recalibration skipped\n")
    brier_orig <- bootstrap_brier_ci(pred_probs, labels_bin, n_boot)
    return(list(
      brier_scores     = tibble(Model = paste(model_name, "Original"),
                                Brier_Mean = brier_orig$mean,
                                CI_Lower = brier_orig$lower, CI_Upper = brier_orig$upper),
      calibration_data = plot_calibration_from_probs(pred_probs, labels_bin,
                                                     paste(model_name, "Original"), bins) %>%
        mutate(Base_Model = model_name),
      platt_model = NULL, iso_model = NULL
    ))
  }
  
  # Platt scaling with tryCatch
  platt_model <- tryCatch(
    glm(labels_bin ~ pred_probs, family = binomial, control = glm.control(maxit = 100)),
    error = function(e) { cat("❌ Platt failed for", model_name, "\n"); NULL }
  )
  platt_probs <- if (!is.null(platt_model))
    pmin(pmax(predict(platt_model, type = "response"), 0.001, 0.999)) else pred_probs
  
  # Isotonic regression with tryCatch
  iso_model <- tryCatch(isoreg(pred_probs, labels_bin),
                        error = function(e) { cat("❌ Isotonic failed for", model_name, "\n"); NULL })
  iso_probs <- if (!is.null(iso_model))
    pmin(pmax(fitted(iso_model), 0.001, 0.999)) else pred_probs
  
  # Bootstrapped Brier scores
  brier_original <- bootstrap_brier_ci(pred_probs,  labels_bin, n_boot)
  brier_platt    <- bootstrap_brier_ci(platt_probs, labels_bin, n_boot)
  brier_iso      <- bootstrap_brier_ci(iso_probs,   labels_bin, n_boot)
  
  brier_scores <- tibble(
    Model      = c(paste(model_name, "Original"), paste(model_name, "Platt"), paste(model_name, "Isotonic")),
    Brier_Mean = c(brier_original$mean, brier_platt$mean, brier_iso$mean),
    CI_Lower   = c(brier_original$lower, brier_platt$lower, brier_iso$lower),
    CI_Upper   = c(brier_original$upper, brier_platt$upper, brier_iso$upper)
  )
  
  cal_df <- bind_rows(
    plot_calibration_from_probs(pred_probs,  labels_bin, paste(model_name, "Original"), bins),
    plot_calibration_from_probs(platt_probs, labels_bin, paste(model_name, "Platt"),    bins),
    plot_calibration_from_probs(iso_probs,   labels_bin, paste(model_name, "Isotonic"), bins)
  ) %>% mutate(Base_Model = model_name)
  
  list(brier_scores = brier_scores, calibration_data = cal_df,
       platt_model = platt_model, iso_model = iso_model)
}

# Single run of recalibration (no duplicate)
recalibration_results <- map(names(all_model_objects), function(m) {
  recalibrate_and_plot(
    all_model_objects[[m]]$probs,
    all_model_objects[[m]]$labels,
    model_name = m, bins = 5, n_boot = 1000
  )
})
names(recalibration_results) <- names(all_model_objects)

brier_all_models <- bind_rows(map(recalibration_results, "brier_scores"))

brier_summary_table <- brier_all_models %>%
  separate(Model, into = c("Base_Model", "Method"), sep = " ", extra = "merge") %>%
  pivot_wider(names_from = Method, values_from = c(Brier_Mean, CI_Lower, CI_Upper)) %>%
  mutate(
    `Original Brier` = round(Brier_Mean_Original, 3),
    `Original CI`    = paste0("(", round(CI_Lower_Original, 3), ", ", round(CI_Upper_Original, 3), ")"),
    `Platt Brier`    = round(Brier_Mean_Platt, 3),
    `Platt CI`       = paste0("(", round(CI_Lower_Platt, 3), ", ", round(CI_Upper_Platt, 3), ")"),
    `Isotonic Brier` = round(Brier_Mean_Isotonic, 3),
    `Isotonic CI`    = paste0("(", round(CI_Lower_Isotonic, 3), ", ", round(CI_Upper_Isotonic, 3), ")"),
    `Better Method`  = case_when(
      is.na(Brier_Mean_Platt) | is.na(Brier_Mean_Isotonic) ~ "Tie",
      Brier_Mean_Platt < Brier_Mean_Isotonic               ~ "Platt",
      Brier_Mean_Isotonic < Brier_Mean_Platt               ~ "Isotonic",
      TRUE                                                 ~ "Tie"
    ),
    Improvement = round(abs(Brier_Mean_Platt - Brier_Mean_Isotonic), 3)
  ) %>%
  dplyr::select(Model = Base_Model,
                `Original Brier`, `Original CI`,
                `Platt Brier`,    `Platt CI`,
                `Isotonic Brier`, `Isotonic CI`,
                `Better Method`,  Improvement)

cat("\n=== BRIER SUMMARY TABLE ===\n")
print(brier_summary_table, width = Inf)

# Calibration curve visualization
calibration_all <- bind_rows(map(recalibration_results, "calibration_data"))

fig_calibration_all <- ggplot(calibration_all, aes(x = mean_pred, y = mean_obs, color = Model)) +
  geom_point(size = 2) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  facet_wrap(~ Base_Model, scales = "free") +
  labs(title = "Calibration Curves Across Models (Original vs Recalibrated)",
       x = "Mean Predicted Probability", y = "Mean Observed Outcome") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", legend.title = element_text(face = "bold"))

print(fig_calibration_all)


# 8.9 Hosmer-Lemeshow Test
# --------------------------
safe_hl_test <- function(labels, probs) {
  g      <- min(10, max(4, floor(length(unique(probs)) / 2)))
  result <- tryCatch(
    ResourceSelection::hoslem.test(labels, probs, g = g),
    warning = function(w) NULL, error = function(e) NULL
  )
  if (is.null(result)) return(list(statistic = NA, p.value = NA))
  list(statistic = as.numeric(result$statistic), p.value = result$p.value)
}

perform_hl_verification <- function(model_name, all_model_objects) {
  probs  <- all_model_objects[[model_name]]$probs
  labels <- all_model_objects[[model_name]]$labels
  
  hl_original <- safe_hl_test(labels, probs)
  
  platt_model <- glm(labels ~ probs, family = binomial)
  platt_probs <- predict(platt_model, type = "response")
  hl_platt    <- safe_hl_test(labels, platt_probs)
  
  iso_model <- tryCatch(isoreg(probs, labels), error = function(e) NULL)
  iso_probs <- if (!is.null(iso_model)) fitted(iso_model) else probs
  hl_iso    <- safe_hl_test(labels, iso_probs)
  
  tibble(Model          = model_name,
         HL_Original_p  = hl_original$p.value,
         HL_Original_X2 = hl_original$statistic,
         HL_Platt_p     = hl_platt$p.value,
         HL_Platt_X2    = hl_platt$statistic,
         HL_Isotonic_p  = hl_iso$p.value,
         HL_Isotonic_X2 = hl_iso$statistic)
}

hl_test_table <- map_dfr(names(all_model_objects),
                         ~ perform_hl_verification(.x, all_model_objects)) %>%
  mutate(Best_p = pmax(HL_Original_p, HL_Platt_p, HL_Isotonic_p, na.rm = TRUE)) %>%
  arrange(desc(Best_p))

cat("\n=== HOSMER-LEMESHOW TEST RESULTS ===\n")
print(hl_test_table, width = Inf)

# HL plot — original dataset only
make_hl_plot <- function(hl_table, dataset_label, fig_title) {
  hl_table %>%
    pivot_longer(cols = c(HL_Original_p, HL_Platt_p, HL_Isotonic_p),
                 names_to = "Calibration", values_to = "p_value") %>%
    filter(!is.na(p_value)) %>%
    ggplot(aes(x = Model, y = p_value, fill = Calibration)) +
    geom_col(position = position_dodge()) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
    labs(title = fig_title,
         subtitle = paste(dataset_label, "- Red line = acceptable calibration (p > 0.05)"),
         y = "p-value", x = "Model") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

hl_original_only <- hl_test_table %>% filter(grepl("original", Model))
hl_plot_original <- make_hl_plot(hl_original_only, "Original dataset", "Figure 2. HL Calibration p-values (Original)")
ggsave("Figure_2_HL_Original.png", hl_plot_original, width = 10, height = 6, dpi = 300)


# ===============================================================
# 9. DECISION CURVE ANALYSIS (DCA)
# ===============================================================

get_isotonic_probabilities <- function(model, test_data) {
  if (inherits(model, "xgb.Booster")) {
    # Special handling for XGBoost
    x_test <- model.matrix(Metabolic_Syndrome ~ ., data = test_data)[, -1]
    probs_original <- predict(model, xgboost::xgb.DMatrix(data = x_test))
  } else {
    probs_original <- predict(model, test_data, type = "prob")[, "Yes"]
  }
  
  outcome_binary <- ifelse(test_data$Metabolic_Syndrome == "Yes", 1, 0)
  
  if (length(unique(probs_original)) > 3) {
    iso_model <- isoreg(probs_original, outcome_binary)
    return(pmin(pmax(fitted(iso_model), 0.01), 0.99))
  } else {
    return(probs_original)
  }
}


calculate_clinical_impact <- function(dca_results, thresholds = c(0.1, 0.2, 0.3)) {
  
  impact_data <- data.frame()
  
  for (thr in thresholds) {
    
    threshold_data <- dca_results %>%
      dplyr::filter(abs(threshold - thr) < 0.005) %>%
      dplyr::group_by(model) %>%
      dplyr::summarise(
        net_benefit = mean(net_benefit),
        .groups = "drop"
      ) %>%
      dplyr::mutate(Threshold = thr)
    
    impact_data <- dplyr::bind_rows(impact_data, threshold_data)
  }
  
  impact_data %>%
    dplyr::group_by(Threshold) %>%
    dplyr::mutate(
      Rank = rank(-net_benefit),
      Net_Benefit = round(net_benefit, 4)
    ) %>%
    dplyr::arrange(Threshold, Rank) %>%
    dplyr::rename(Model = model) %>%
    dplyr::select(Threshold, Model, Net_Benefit, Rank) %>%
    dplyr::ungroup()
}


perform_dca_isotonic <- function(all_models, test_data, outcome_var = "Metabolic_Syndrome",
                                 probability_thresholds = seq(0.01, 0.5, by = 0.01)) {
  cat("=== DCA WITH ISOTONIC-CALIBRATED PROBABILITIES ===\n")
  models         <- all_models$original
  outcome_binary <- ifelse(test_data[[outcome_var]] == "Yes", 1, 0)
  n              <- length(outcome_binary)
  prevalence     <- mean(outcome_binary)
  dca_results    <- data.frame()
  
  for (threshold in probability_thresholds) {
    net_benefit_all  <- prevalence - (1 - prevalence) * (threshold / (1 - threshold))
    dca_results <- rbind(dca_results,
                         data.frame(threshold = threshold, model = "Treat All",  net_benefit = net_benefit_all),
                         data.frame(threshold = threshold, model = "Treat None", net_benefit = 0))
    
    for (model_name in names(models)) {
      tryCatch({
        probs           <- get_isotonic_probabilities(models[[model_name]], test_data)
        predicted_class <- ifelse(probs >= threshold, 1, 0)
        tp              <- sum(predicted_class == 1 & outcome_binary == 1)
        fp              <- sum(predicted_class == 1 & outcome_binary == 0)
        net_benefit     <- (tp / n) - (fp / n) * (threshold / (1 - threshold))
        dca_results     <- rbind(dca_results,
                                 data.frame(threshold = threshold, model = model_name, net_benefit = net_benefit))
      }, error = function(e) cat("⚠️ Skipping", model_name, ":", e$message, "\n"))
    }
  }
  return(dca_results)
}

create_isotonic_dca_plot <- function(dca_results, title = "Decision Curve Analysis") {
  model_colors <- c(
    "Treat All"     = "#000000", "Treat None"    = "#7f7f7f",
    "logistic"      = "#1f77b4", "svm_linear"    = "#17becf",
    "svm_radial"    = "#2ca02c", "random_forest" = "#e41a1c",
    "xgboost"       = "#ff7f00", "lightgbm"      = "#984ea3",
    "naive_bayes"   = "#f781bf", "decision_tree" = "#a65628",
    "knn"           = "#dede00"
  )
  dca_results <- dca_results %>%
    mutate(line_type = ifelse(model %in% c("Treat All", "Treat None"), "dashed", "solid"),
           model     = factor(model, levels = names(model_colors)))
  
  ggplot(dca_results, aes(x = threshold, y = net_benefit, color = model, linetype = line_type)) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(values = model_colors) +
    scale_linetype_manual(values = c("dashed" = "dashed", "solid" = "solid"), guide = "none") +
    labs(title = title, x = "Probability Threshold", y = "Net Benefit", color = "Model") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "bottom", plot.title = element_text(face = "bold", hjust = 0.5),
          panel.grid.minor = element_blank()) +
    guides(color = guide_legend(nrow = 3, byrow = TRUE)) +
    scale_x_continuous(limits = c(0, 0.5), expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.1)))
}


# Execute DCA
cat("Running DCA...\n")
dca_isotonic_results <- perform_dca_isotonic(
  all_models           = all_models,
  test_data            = datasets$original$test,
  probability_thresholds = seq(0.01, 0.4, by = 0.01)
)

dca_plot <- create_isotonic_dca_plot(dca_isotonic_results,
                                     title = "Decision Curve Analysis of Isotonic-Calibrated Models")
print(dca_plot)

ggsave("Figure4_Decision_Curve_Analysis_Isotonic.tiff", plot = dca_plot, width = 10, height = 8, dpi = 300, compression = "lzw")
ggsave("Figure4_Decision_Curve_Analysis_Isotonic.png",  plot = dca_plot, width = 10, height = 8, dpi = 300)

clinical_impact <- calculate_clinical_impact(dca_isotonic_results)

# Combined Figure 4


clinical_impact_clean <- clinical_impact %>%
  mutate(Model = tolower(Model)) %>%   # force lowercase
  mutate(Model = gsub(" ", "_", Model)) %>%  # replace spaces with underscores
  mutate(Model = factor(Model, levels = names(model_colors)))

# Define consistent colors for all models

model_colors <- c(
  "Treat All"     = "#000000", 
  "Treat None"    = "#7f7f7f",
  "logistic"      = "#1f77b4", 
  "svm_linear"    = "#17becf",
  "svm_radial"    = "#2ca02c", 
  "random_forest" = "#e41a1c",
  "xgboost"       = "#ff7f00", 
  "lightgbm"      = "#984ea3",
  "naive_bayes"   = "#f781bf", 
  "decision_tree" = "#a65628",
  "knn"           = "#dede00"
)

dca_plot_bar <- ggplot(clinical_impact_clean,
                       aes(x = factor(Threshold), y = Net_Benefit, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge2(width = 0.8, preserve = "single")) +
  geom_text(aes(label = Net_Benefit),
            position = position_dodge2(width = 0.8, preserve = "single"),
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = model_colors) +   # <-- ensures colors match panel A
  theme_minimal(base_size = 12) +
  labs(title = "Clinical Impact at Selected Thresholds",
       x = "Threshold Probability", y = "Net Benefit") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 9),
    plot.title = element_text(face = "bold")
  ) +
  guides(fill = guide_legend(nrow = 3, byrow = TRUE))




figure_4 <- dca_plot + dca_plot_bar +
  plot_annotation(
    title    = "Figure 4. Decision Curve Analysis",
    subtitle = "A: Combined decision curve  |  B: Clinical impact at selected thresholds",
    tag_levels = "A"
  )

print(figure_4)
ggsave("Figure_4_Decision_Curve_Analysis.png", plot = figure_4, width = 12, height = 6, dpi = 300)
ggsave("decision_curve_analysis.pdf",          plot = dca_plot_bar dca_, width = 10, height = 6, dpi = 300)

cat("\n=== CLINICAL IMPACT AT KEY THRESHOLDS ===\n")
print(clinical_impact)

best_models <- clinical_impact %>%
  group_by(Threshold) %>%
  filter(Rank == 1) %>%
  ungroup()

cat("\n=== BEST PERFORMING MODELS BY THRESHOLD ===\n")
print(best_models)


#========================================================
# Sensitivity analysis test evaluation WITH hip circumference
#=========================================================
# Reload raw dataset
df_raw <- df2

# Keep hip circumference, drop other diagnostic variables only
df_with_hip <- df_raw %>%
  dplyr::select(
    -dplyr::any_of(c(
      "Blood_Sugar..mmol.L.",
      "Triglycerides..mmol.L.",
      "Cholesterol_HDL_.mmol.L.",
      "BMI.kg.m2.",
      "Bp_Systolic..mmHg.",
      "Bp_Diastolic..mmHg.",
      "Waist_Circumference.cm."
      # ⚠️ Hip_Circumference.cm. intentionally retained
    ))
  ) %>%
  dplyr::rename(
    Alcohol_Consumption = Alcohol_Consuption,
    Regimen_Type = Drug_Code
  )

# Trim whitespace in categorical fields
cols_to_clean <- c("Regimen_Type", "Sex", "Diabetes_Mellitus_status", 
                   "Tobbacco_Use", "Alcohol_Consumption", "Educational_Level", 
                   "Event_Name")

df_with_hip[cols_to_clean] <- lapply(
  df_with_hip[cols_to_clean], 
  function(x) if (is.character(x)) trimws(x) else x
)

# Standardize Event_Name values
df_with_hip$Event_Name[df_with_hip$Event_Name == "Week_ 144"] <- "week_144"

# Filter out Baseline and drop Event_Name safely
df_with_hip <- df_with_hip %>%
  dplyr::filter(Event_Name != "Baseline") %>%
  dplyr::select(-dplyr::any_of("Event_Name"))


# Impute missing values with MICE
library(mice)
imp_with_hip <- mice(df_with_hip, m = 5, method = "pmm", maxit = 10, seed = 123)
df_with_hip_imputed <- complete(imp_with_hip, 1)

# Convert categorical variables to factors
metabolic_data_with_hip <- df_with_hip_imputed %>%
  mutate(across(c("Regimen_Type", "Sex", "Diabetes_Mellitus_status",
                  "Alcohol_Consumption", "Educational_Level",  
                  "Tobbacco_Use", "Metabolic_Syndrome"), as.factor),
         Educational_Level = factor(Educational_Level,
                                    levels = c("none", "primary", "secondary", "Terciary"),
                                    ordered = TRUE))

# Rename age variable
metabolic_data_with_hip <- metabolic_data_with_hip %>%
  rename(Age_years = Age..yrs.)

# Drop unwanted columns
metabolic_data_with_hip <- metabolic_data_with_hip %>%
  dplyr::select(-dplyr::any_of(c(
    "Tobbacco_Use",
    "Regimen_Type",
    "Diabetes_Mellitus_status",
    "CD4_count.cells.µl.",
    "Trial_number",
    "Educational_Level"
  )))


## ✅ Fix factor levels for Metabolic_Syndrome
# Recode to valid names ("No"/"Yes") before training
metabolic_data_with_hip$Metabolic_Syndrome <- dplyr::recode(
  metabolic_data_with_hip$Metabolic_Syndrome,
  "No MetSyn" = "No",
  "MetSyn"    = "Yes"
)

metabolic_data_with_hip$Metabolic_Syndrome <- factor(
  metabolic_data_with_hip$Metabolic_Syndrome,
  levels = c("No","Yes")
)

# Log-transform Viral Load
metabolic_data_with_hip <- metabolic_data_with_hip %>%
  dplyr::mutate(Log_Viral_Load = log10(Viral_Load.cp.ml. + 1)) %>%
  dplyr::select(-dplyr::any_of("Viral_Load.cp.ml."))

# ✅ Split WITH hip circumference dataset
set.seed(123)
train_indices_hip <- createDataPartition(
  metabolic_data_with_hip$Metabolic_Syndrome,
  p = 0.7,
  list = FALSE
)

train_set_with_hip <- metabolic_data_with_hip[train_indices_hip, ]
test_set_with_hip  <- metabolic_data_with_hip[-train_indices_hip, ]

# ✅ Ensure consistent levels in train/test
train_set_with_hip$Metabolic_Syndrome <- factor(train_set_with_hip$Metabolic_Syndrome,
                                                levels = c("No","Yes"))
test_set_with_hip$Metabolic_Syndrome  <- factor(test_set_with_hip$Metabolic_Syndrome,
                                                levels = c("No","Yes"))

# ✅ Train and evaluate WITH hip circumference
set.seed(123)
model_with_hip <- train(
  Metabolic_Syndrome ~ ., 
  data = train_set_with_hip,
  method = "rf",
  trControl = ctrl_stratified,
  metric = "ROC"
)
results_with_hip <- evaluate_model_youden(model_with_hip, test_set_with_hip, "rf_with_hip")

# Train and evaluate WITHOUT hip circumference
set.seed(123)
model_without_hip <- train(
  Metabolic_Syndrome ~ ., 
  data = train_set,      
  method = "rf",
  trControl = ctrl_stratified,
  metric = "ROC"
)

results_withno_hip <- evaluate_model_youden(
  model_without_hip, 
  test_set,             
  "rf_without_hip"
)

# Compare against primary model (without hip circumference)
sensitivity_results <- data.frame(
  Model = c("With Hip Circumference", "Without Hip Circumference"),
  AUC = c(results_with_hip$AUC, results_withno_hip$AUC),
  MCC = c(results_with_hip$MCC, results_withno_hip$MCC)
)

print(sensitivity_results)


library(ggplot2)

# Results data frame
sensitivity_results <- data.frame(
  Model = c("With Hip Circumference", "Without Hip Circumference"),
  AUC = c(0.739, 0.560),
  MCC = c(0.405, 0.174)
)

# Plot AUC comparison
ggplot(sensitivity_results, aes(x = Model, y = AUC, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(AUC, 3)), vjust = -0.5) +
  theme_minimal(base_size = 14) +
  labs(title = "Sensitivity Analysis: Effect of Hip Circumference",
       y = "AUC", x = "") +
  theme(legend.position = "none")

# Plot MCC comparison
ggplot(sensitivity_results, aes(x = Model, y = MCC, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(MCC, 3)), vjust = -0.5) +
  theme_minimal(base_size = 14) +
  labs(title = "Sensitivity Analysis: Effect of Hip Circumference",
       y = "MCC", x = "") +
  theme(legend.position = "none")


library(tidyr)

# Reshape data
sensitivity_long <- sensitivity_results %>%
  pivot_longer(cols = c(AUC, MCC), names_to = "Metric", values_to = "Value")

# Create plot object with figure label in the title
sensitivity_plot <- ggplot(sensitivity_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = round(Value, 3)), 
            position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5) +
  theme_minimal(base_size = 14) +
  labs(title = "Figure 3. Sensitivity Analysis: Effect of Hip Circumference",
       y = "Performance Metric", x = "") +
  theme(legend.position = "bottom", legend.title = element_blank())


# ============================================================
# SUPPLEMENTARY TABLES S2 AND S3 FOR BMC PAPER
# S2: Correlation between Waist and Hip Circumference
# S3: Partial correlation (Hip vs MetS controlling for Waist)
# ============================================================

# ---------- SUPPLEMENTARY TABLE S2: Correlation (Waist vs Hip) ----------
cat("\n=== Supplementary Table S2 ===\n")

# Assuming cor_data exists from your scatter plot code
cor_test <- cor.test(
  cor_data$Waist_Circumference.cm., 
  cor_data$Hip_Circumference.cm.,
  method = "pearson"
)

cor_table_s2 <- data.frame(
  Variable_1 = "Waist circumference",
  Variable_2 = "Hip circumference",
  Correlation_r = round(cor_test$estimate, 3),
  CI_Lower = round(cor_test$conf.int[1], 3),
  CI_Upper = round(cor_test$conf.int[2], 3),
  p_value = ifelse(cor_test$p.value < 0.001, "< 0.001", 
                   round(cor_test$p.value, 3))
)

# Create flextable for Table S2
ft_s2 <- flextable(cor_table_s2) %>%
  theme_vanilla() %>%
  bold(part = "header") %>%
  set_header_labels(
    Variable_1 = "Variable 1",
    Variable_2 = "Variable 2",
    Correlation_r = "Correlation r",
    CI_Lower = "95% CI Lower",
    CI_Upper = "95% CI Upper",
    p_value = "p-value"
  ) %>%
  align(align = "center", part = "all") %>%
  width(width = c(3.5, 3.5, 1.8, 1.5, 1.5, 1.5)) %>%
  set_caption(
    caption = "Supplementary Table S2 Correlation between waist and hip circumference",
    style = "TableCaption"
  ) %>%
  add_footer_lines(
    "CI confidence interval. Correlation calculated using Pearson's method."
  ) %>%
  fontsize(size = 10, part = "all") %>%
  font(fontname = "Arial", part = "all")

# Print to console for preview
print(ft_s2)

# ---------- SUPPLEMENTARY TABLE S3: Partial Correlation ----------
cat("\n=== Supplementary Table S3 ===\n")

# Your existing partial correlation code
met_syn_numeric <- ifelse(df2$Metabolic_Syndrome == "MetSyn", 1, 0)

pcor_data <- df2 %>%
  dplyr::select(Hip_Circumference.cm., Waist_Circumference.cm.) %>%
  dplyr::mutate(MetSyn = met_syn_numeric) %>%
  na.omit()

pcor_result <- pcor.test(
  pcor_data$Hip_Circumference.cm.,
  pcor_data$MetSyn,
  pcor_data$Waist_Circumference.cm.
)

partial_table_s3 <- data.frame(
  Variable_1 = "Hip circumference",
  Variable_2 = "Metabolic syndrome (MetS)",
  Control_Variable = "Waist circumference",
  Partial_r = round(pcor_result$estimate, 3),
  p_value = ifelse(pcor_result$p.value < 0.001, "< 0.001",
                   round(pcor_result$p.value, 3))
)

# Create flextable for Table S3
ft_s3 <- flextable(partial_table_s3) %>%
  theme_vanilla() %>%
  bold(part = "header") %>%
  set_header_labels(
    Variable_1 = "Variable 1",
    Variable_2 = "Variable 2",
    Control_Variable = "Control variable",
    Partial_r = "Partial r",
    p_value = "p-value"
  ) %>%
  align(align = "center", part = "all") %>%
  width(width = c(3.5, 4, 3.5, 1.8, 1.5)) %>%
  set_caption(
    caption = "Supplementary Table S3 Partial correlation of hip circumference with metabolic syndrome (MetS) controlling for waist circumference",
    style = "TableCaption"
  ) %>%
  add_footer_lines(
    "MetS defined according to standard criteria [specify your criteria]. Partial correlation adjusted for waist circumference as indicated."
  ) %>%
  fontsize(size = 10, part = "all") %>%
  font(fontname = "Arial", part = "all")

# Print to console
print(ft_s3)

# ---------- EXPORT BOTH TABLES TO WORD DOCUMENT ----------
cat("\n=== Exporting Supplementary Tables S2 and S3 to Word ===\n")

doc <- read_docx() %>%
  body_add_par("Supplementary Materials", style = "heading 1") %>%
  body_add_par(" ", style = "Normal") %>%
  
  # Table S2
  body_add_par("Supplementary Table S2", style = "heading 2") %>%
  body_add_flextable(ft_s2) %>%
  body_add_par(" ", style = "Normal") %>%
  body_add_par(" ", style = "Normal") %>%
  
  # Table S3
  body_add_par("Supplementary Table S3", style = "heading 2") %>%
  body_add_flextable(ft_s3)

# Save the document
print(doc, target = "S)

cat("✓ Tables exported to 'Supplementary_Tables_S2_S3_BMC_Style.docx'\n")

# ---------- OPTIONAL: Print both tables side by side in console ----------
cat("\n\n========== SUMMARY OF SUPPLEMENTARY TABLES ==========\n\n")
cat("SUPPLEMENTARY TABLE S2\n")
cat("Correlation between waist and hip circumference\n")
cat("------------------------------------------------------------\n")
print(cor_table_s2)
cat("\n\nSUPPLEMENTARY TABLE S3\n")
cat("Partial correlation of hip circumference with MetS controlling for waist\n")
cat("------------------------------------------------------------\n")
print(partial_table_s3)

