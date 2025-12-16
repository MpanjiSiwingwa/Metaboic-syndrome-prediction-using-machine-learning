# =============================================
# METABOLIC SYNDROME PREDICTION MODEL PIPELINE
# =============================================
# Version: 1.0
# Last Updated: 15th November 2025
# Author: Mpanji Siwingwa
# 
# This script provides a complete pipeline for:
# 1. Data preparation and cleaning
# 2. Feature selection
# 3. Model training (8 algorithms)
# 4. Performance evaluation
# 5. Result visualization
# =============================================

# 1. INITIAL SETUP AND CONFIGURATION
# ==================================

# 1.1 Load Required Libraries
# --------------------------
# 📦 Data Manipulation & Pre-processing
library(tidyverse)      # Unified collection of data tools (ggplot2, dplyr, tidyr, etc.)
library(dplyr)          # Data manipulation and transformation
library(rsample)        # Train/test splitting and resampling strategies
library(recipes)        # Preprocessing pipelines for modeling (normalization, encoding, etc.)
library(purrr)          # Functional programming tools for iteration
library(reshape2)       # Convert between wide and long formats (legacy; superseded by tidyr)
library(themis)         # Data balancing with SMOTE and other techniques



# 📊 Evaluation & Reporting
library(pROC)             # ROC curve analysis and AUC computation
library(rmda)             # Decision curve analysis for evaluating clinical usefulness
library(knitr)            # Report generation via R Markdown
library(kableExtra)       # Advanced formatting for LaTeX/HTML tables
library(tableone)         # Create descriptive summary tables for clinical datasets
library(flextable)        # Export styled tables to Word/PowerPoint
library(officer)          # Edit and write Word/PowerPoint documents programmatically
library(car)              # Companion to Applied Regression — includes VIF, outlier tests, linear model diagnostics
library(ResourceSelection) # Model calibration diagnostics including Hosmer-Lemeshow test for logistic models
library(gtsummary)
# 🎨 Visualization
library(ggrepel)        # Repels overlapping text labels in ggplot2 plots
library(ggpubr)         # Publication-ready enhancements to ggplot2
library(webshot2)       # Capture and export web-based visual content and widgets as images (screenshot engine)

#===============================================================================
# Table 1: Baseline Characteristics of the Study Population by MetS Status
#===============================================================================

# Load and prepare datasets
train_set <- read.csv("train_set.csv", stringsAsFactors = FALSE) %>%
  dplyr::mutate(Set = "Training Set")

test_set <- read.csv("test_set.csv", stringsAsFactors = FALSE) %>%
  dplyr::mutate(Set = "Test Set")

# Harmonize columns and combine
common_cols <- intersect(names(train_set), names(test_set))
combined_data <- rbind(train_set[, common_cols], test_set[, common_cols])

# Create grouping variables
combined_data <- combined_data %>%
  dplyr::mutate(
    Age_group = factor(ifelse(Age..yrs. < 40, "< 40 years", "≥ 40 years")),
    CD4_group = cut(CD4_count.cells.µl., breaks = c(-Inf, 100, 350, Inf),
                    labels = c("100 cells/µL", "100–350 cells/µL", ">350 cells/µL")),
    ViralLoad_group = cut(Viral_Load.cp.ml., breaks = c(-Inf, 1000, Inf),
                          labels = c("<1000 cp/mL", "≥1000 cp/mL")),
    MetS_status = factor(Metabolic_Syndrome,
                         levels = c("No MetSyn", "MetSyn"),
                         labels = c("Non MetS", "MetS")),
    Sex = factor(Sex),
    Alcohol_Consumption = factor(Alcohol_Consumption),
    Educational_Level = factor(Educational_Level)
  )

# Generate Table 1 for each set
generate_table1 <- function(data, set_label) {
  data %>%
    dplyr::filter(Set == set_label) %>%
    dplyr::select(MetS_status, Age_group, CD4_group, ViralLoad_group,
                  Sex, Alcohol_Consumption, Educational_Level) %>%
    gtsummary::tbl_summary(
      by = MetS_status,
      label = list(
        Age_group ~ "Age",
        CD4_group ~ "CD4 Count",
        ViralLoad_group ~ "Viral Load",
        Sex ~ "Sex",
        Alcohol_Consumption ~ "Alcohol Use",
        Educational_Level ~ "Level of Education"
      ),
      missing = "no"
    ) %>%
    gtsummary::add_p() %>%
    gtsummary::modify_header(label = "**Characteristics**") %>%
    gtsummary::modify_caption(paste0("**Table 1: Baseline Characteristics by MetS Status — ", set_label, "**"))
}

table_train <- generate_table1(combined_data, "Training Set")
table_test  <- generate_table1(combined_data, "Test Set")

# Merge tables side-by-side
table_combined <- gtsummary::tbl_merge(
  tbls = list(table_train, table_test),
  tab_spanner = c("**Training Set**", "**Test Set**")
)

# Convert to flextable
table1_flex <- gtsummary::as_flex_table(table_combined)

# Create MetS summary row
summary_counts <- combined_data %>%
  dplyr::group_by(Set, MetS_status) %>%
  dplyr::summarise(n = n(), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = MetS_status, values_from = n, values_fill = 0) %>%
  dplyr::mutate(
    Total = `Non MetS` + `MetS`,
    Non_MetS_pct = paste0(`Non MetS`, " (", round(`Non MetS` / Total * 100, 1), "%)"),
    MetS_pct = paste0(`MetS`, " (", round(`MetS` / Total * 100, 1), "%)")
  )

p_val <- fisher.test(matrix(c(
  summary_counts$`Non MetS`[summary_counts$Set == "Training Set"],
  summary_counts$`MetS`[summary_counts$Set == "Training Set"],
  summary_counts$`Non MetS`[summary_counts$Set == "Test Set"],
  summary_counts$`MetS`[summary_counts$Set == "Test Set"]
), nrow = 2))$p.value

summary_row <- data.frame(
  Characteristics     = "MetS, n (%)",
  Training_Non_MetS   = summary_counts$Non_MetS_pct[summary_counts$Set == "Training Set"],
  Training_MetS       = summary_counts$MetS_pct[summary_counts$Set == "Training Set"],
  Training_p          = format.pval(p_val, digits = 3, eps = 0.001),
  Test_Non_MetS       = summary_counts$Non_MetS_pct[summary_counts$Set == "Test Set"],
  Test_MetS           = summary_counts$MetS_pct[summary_counts$Set == "Test Set"],
  Test_p              = format.pval(p_val, digits = 3, eps = 0.001)
)

# Add summary row below tertiary header
table1_flex <- flextable::add_body_row(
  x = table1_flex,
  values = as.character(unlist(summary_row)),
  colwidths = rep(1, ncol(summary_row))
)

# Export to Word
doc <- officer::read_docx() %>%
  flextable::body_add_flextable(value = table1_flex)

print(doc, target = "Table1_MetS_by_Set.docx")


#===============================================================================
# Table 1. Baseline Characteristics of Study Participants Stratified by Metabolic Syndrome Status
#===============================================================================

library(dplyr)
library(tidyr)
library(gtsummary)
library(flextable)
library(officer)

# Load and label datasets
train_set <- read.csv("train_set.csv", stringsAsFactors = FALSE) %>%
  mutate(Set = "Training Set")

test_set <- read.csv("test_set.csv", stringsAsFactors = FALSE) %>%
  mutate(Set = "Test Set")

# Harmonize columns and combine
common_cols <- intersect(names(train_set), names(test_set))
combined_data <- bind_rows(train_set[, common_cols], test_set[, common_cols])

# Create grouping variables
combined_data <- combined_data %>%
  mutate(
    Age_group = factor(ifelse(Age..yrs. < 40, "< 40 years", "≥ 40 years")),
    CD4_group = cut(CD4_count.cells.µl., breaks = c(-Inf, 100, 350, Inf),
                    labels = c("100 cells/µL", "100–350 cells/µL", ">350 cells/µL")),
    ViralLoad_group = cut(Viral_Load.cp.ml., breaks = c(-Inf, 1000, Inf),
                          labels = c("<1000 cp/mL", "≥1000 cp/mL")),
    MetS_status = factor(Metabolic_Syndrome,
                         levels = c("No_MetSyn", "MetSyn"),
                         labels = c("Non-MetS", "MetS")),
    Sex = factor(Sex),
    Alcohol_Consumption = factor(Alcohol_Consumption),
    Educational_Level = factor(Educational_Level)
  )

# Function to generate Table 1 for each set
generate_table1 <- function(data, set_label) {
  data %>%
    filter(Set == set_label) %>%
    select(MetS_status, Age_group, CD4_group, ViralLoad_group,
           Sex, Alcohol_Consumption, Educational_Level) %>%
    tbl_summary(
      by = MetS_status,
      label = list(
        Age_group ~ "Age",
        CD4_group ~ "CD4 Count",
        ViralLoad_group ~ "Viral Load",
        Sex ~ "Sex",
        Alcohol_Consumption ~ "Alcohol Use",
        Educational_Level ~ "Level of Education"
      ),
      missing = "no"
    ) %>%
    add_p() %>%
    modify_header(label = "**Characteristics**") %>%
    modify_caption(paste0("**Table 1. Characteristics of Study Participants Stratified by Metabolic Syndrome Status in Training and Test Sets ", set_label, "**"))
}

# Generate tables
table_train <- generate_table1(combined_data, "Training Set")
table_test  <- generate_table1(combined_data, "Test Set")

# Merge side-by-side
table_combined <- tbl_merge(
  tbls = list(table_train, table_test),
  tab_spanner = c("**Training Set**", "**Test Set**")
)

# Convert to flextable
table1_flex <- as_flex_table(table_combined)

# Create MetS summary row
summary_counts <- combined_data %>%
  group_by(Set, MetS_status) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = MetS_status, values_from = n, values_fill = 0) %>%
  mutate(
    Total = `Non-MetS` + `MetS`,
    Non_MetS_pct = paste0(`Non-MetS`, " (", round(`Non-MetS` / Total * 100, 1), "%)"),
    MetS_pct = paste0(`MetS`, " (", round(`MetS` / Total * 100, 1), "%)")
  )

# Fisher's test across sets
p_val <- fisher.test(matrix(c(
  summary_counts$`Non-MetS`[summary_counts$Set == "Training Set"],
  summary_counts$`MetS`[summary_counts$Set == "Training Set"],
  summary_counts$`Non-MetS`[summary_counts$Set == "Test Set"],
  summary_counts$`MetS`[summary_counts$Set == "Test Set"]
), nrow = 2))$p.value

# Create summary row
summary_row <- data.frame(
  Characteristics     = "MetS, n (%)",
  Training_Non_MetS   = summary_counts$Non_MetS_pct[summary_counts$Set == "Training Set"],
  Training_MetS       = summary_counts$MetS_pct[summary_counts$Set == "Training Set"],
  Training_p          = format.pval(p_val, digits = 3, eps = 0.001),
  Test_Non_MetS       = summary_counts$Non_MetS_pct[summary_counts$Set == "Test Set"],
  Test_MetS           = summary_counts$MetS_pct[summary_counts$Set == "Test Set"],
  Test_p              = format.pval(p_val, digits = 3, eps = 0.001)
)

# Ensure column names match flextable
names(summary_row) <- names(table1_flex$body$dataset)

# Append MetS row at bottom
table1_flex <- flextable::add_body(
  x = table1_flex,
  values = summary_row
)

# Export to Word
doc <- read_docx() %>%
  flextable::body_add_flextable(value = table1_flex)

print(doc, target = "Table1_MetS_by_Set.docx")


# Convert flextable to image
save_as_image(
  x = table1_flex,
  path = "Table1_MetS_by_Set.png",
  zoom = 2,       # Increase zoom for clarity
  res = 300       # Set resolution to 300 DPI
)
