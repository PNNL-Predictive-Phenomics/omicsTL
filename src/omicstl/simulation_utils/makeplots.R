library(stringr)
library(dplyr)
library(tidyverse)
library(here)

# Configuration
dataset <- "06_13_2025"
cat("=== SCRIPT STARTED ===\n")
cat(sprintf("Using dataset: %s\n", dataset))

# Load or create scenario data
cat("Loading scenario data...\n")
scenario_data_path <- here::here("data", "simulated_data", paste0(dataset, "_scenario_data.rds"))

if(file.exists(scenario_data_path)) {
  cat(sprintf("Found cached scenario data at %s\n", scenario_data_path))
  result <- readRDS(scenario_data_path)
  cat(sprintf("Loaded %d rows of scenario data\n", nrow(result)))
} else {
  cat("No cached scenario data found. Building from individual simulation files...\n")
  sim_data_dir <- here::here("data", "simulated_data", dataset)
  files <- list.files(path = sim_data_dir, pattern = "sim_info_.*\\.RDS$", full.names = TRUE)
  cat(sprintf("Found %d simulation files to process\n", length(files)))

  data_list <- list()
  for(i in seq_along(files)) {
    f <- files[i]
    if(i %% 10 == 0 || i == 1 || i == length(files)) {
      cat(sprintf("Processing scenario file %d/%d: %s\n", i, length(files), basename(f)))
    }

    data <- readRDS(f)
    replicate_scenario <- str_match(f, "sim_info_(\\d+)_(\\w+)\\.RDS")
    data <- data %>%
      mutate(replicate = replicate_scenario[, 2], scenario = replicate_scenario[, 3])
    data_list[[i]] <- data
  }

  result <- bind_rows(data_list)
  cat(sprintf("Combined %d files into scenario data with %d rows\n", length(files), nrow(result)))

  cat(sprintf("Saving combined scenario data to %s\n", scenario_data_path))
  saveRDS(result, scenario_data_path)
}

# Define path to results
path <- here("data", "results", dataset)
cat(sprintf("Results directory: %s\n", path))

# Function to combine CSV files
combine_csv_files <- function(directory_path) {
  cat(sprintf("Scanning directory for CSV files: %s\n", directory_path))
  csv_files <- list.files(path = directory_path, pattern = "\\.csv$", full.names = TRUE)

  if (length(csv_files) == 0) {
    cat("No CSV files found in the directory\n")
    return(data.frame())
  }

  cat(sprintf("Found %d CSV files to process\n", length(csv_files)))
  df_list <- list()

  pb <- txtProgressBar(min = 0, max = length(csv_files), style = 3)
  num_errors <- 0

  for (i in seq_along(csv_files)) {
    if(i %% 50 == 0 || i == 1 || i == length(csv_files)) {
      cat(sprintf("\nProcessing result file %d/%d\n", i, length(csv_files)))
    }

    tryCatch({
      df_list[[i]] <- read.csv(csv_files[i])
    }, error = function(e) {
      num_errors <- num_errors + 1
      cat(sprintf("\nError reading file %s: %s\n", basename(csv_files[i]), e$message))
    })
    setTxtProgressBar(pb, i)
  }
  close(pb)

  cat(sprintf("\nFinished reading CSV files. Encountered %d errors.\n", num_errors))

  if (length(df_list) > 0) {
    cat("Combining data frames...\n")
    combined_df <- bind_rows(df_list)
    cat(sprintf("Combined data frame has %d rows and %d columns\n", 
                nrow(combined_df), ncol(combined_df)))
    return(combined_df)
  } else {
    cat("No data frames to combine, returning empty data frame\n")
    return(data.frame())
  }
}

# Load and combine results
cat("Loading and combining result files...\n")
results <- combine_csv_files(path)

# Prepare results dataframe
cat("Joining results with scenario data and formatting...\n")
results <- results %>% 
  mutate(
    scenario = as.numeric(scenario),
    replicate = as.numeric(replicate)
  ) %>%
  left_join(
    result %>% mutate(scenario = as.numeric(scenario), replicate = as.numeric(replicate)), 
    by = c("scenario", "replicate")
  )

cat(sprintf("Joined data has %d rows\n", nrow(results)))

# Extract response functions
response_fns <- results$response_fn_source %>% unique()
cat(sprintf("Found %d unique response functions\n", length(response_fns)))

# Clean up and restructure results
cat("Cleaning and restructuring results...\n")
results <- results %>% 
  select(
    -dropout,
    -hidden_dim_size,
    -source_epochs,
    -target_epochs,
    -freeze,
    -z_dim
  ) %>%
  mutate(
    response_fn_source = ifelse(sapply(response_fn_source, \(x) identical(x, response_fns[[1]])), "simple", "complex"),
    response_fn_target = ifelse(sapply(response_fn_target, \(x) identical(x, response_fns[[1]])), "simple", "complex"),
    num_features = pmax(num_features_source, num_features_target, 1),
    prop_features = round(num_features / pmax(num_samples_source, num_samples_target), 1)
  )

# Filter and create model_id
cat("Filtering results and creating model identifiers...\n")
results <- results %>%
  filter(
    !((model == "rf" & split != "test_0")) & model_type != "pred_ensemble"
  ) %>% 
  mutate(
    model_id = ifelse(model == "rf" & model_type == "pred_0", "target_nosource", model_id),
    model_id = ifelse(model == "rf" & model_type != "pred_0", 
                      paste0("target", sapply(model_type, \(x) strsplit(x, split = '_')[[1]][2])), 
                      model_id)
  )

cat(sprintf("After filtering, data has %d rows\n", nrow(results)))

# Generate ACC plots
cat("=== GENERATING ACC PLOTS ===\n")
metric <- 'acc'
snr_target_vec <- results$snr_target |> unique()
snr_source_vec <- results$snr_source |> unique()
prop_features_vec <- results$prop_features |> unique()
response_fn_source_vec <- results$response_fn_source |> unique()

cat(sprintf("Parameter combinations: %d target SNR levels × %d source SNR levels × %d feature proportion levels × %d response function types\n",
            length(snr_target_vec), length(snr_source_vec), length(prop_features_vec), length(response_fn_source_vec)))

params <- expand.grid(
  snr_target = snr_target_vec,
  snr_source = snr_source_vec,
  prop_features = prop_features_vec,
  response_fn_source = response_fn_source_vec
)

plots_dir <- here::here("plots", dataset)
if(!dir.exists(plots_dir)) {
  cat(sprintf("Creating plots directory: %s\n", plots_dir))
  dir.create(plots_dir, recursive = TRUE)
}

for(param_idx in seq_len(nrow(params))) {
  snr_target_val <- params$snr_target[param_idx]
  snr_source_val <- params$snr_source[param_idx]
  prop_features_val <- params$prop_features[param_idx]
  response_fn_source_val <- params$response_fn_source[param_idx]

  cat(sprintf("\nGenerating MCC plot %d/%d with parameters:\n", param_idx, nrow(params)))
  cat(sprintf("  - Source SNR: %s\n", snr_source_val))
  cat(sprintf("  - Target SNR: %s\n", snr_target_val))
  cat(sprintf("  - Feature Proportion: %s\n", prop_features_val))
  cat(sprintf("  - Response Function: %s\n", response_fn_source_val))

  filtered_data <- results %>% filter(
    snr_target == snr_target_val,
    snr_source == snr_source_val,
    prop_features == prop_features_val,
    response_fn_source == response_fn_source_val,
    model_id != 'source',
    split == "test_0"
  )

  cat(sprintf("  - Filtered data has %d rows\n", nrow(filtered_data)))

  if(nrow(filtered_data) > 0) {
    p <- filtered_data %>%
      ggplot(
        aes(
          x = as.factor(num_samples_source),
          y = acc,
          color = model_id
        )
      ) + 
      geom_boxplot() +
      facet_grid(model ~ num_samples_target) + 
      theme_bw() + 
      ggtitle(
        paste0(
          "Source SNR = ", snr_source_val, "\n",
          "Target SNR = ", snr_target_val, "\n",
          "Num Features / Sample Size = ", prop_features_val, "\n",
          "Response Type = ", response_fn_source_val, "\n"
        )
      ) + 
      xlab("Source Sample Size") +
      ylab("ACC Score")
    
    plot_name <- paste0(paste(metric, snr_source_val, snr_target_val, prop_features_val, response_fn_source_val, sep = '_'), '.png')
    plot_path <- file.path(plots_dir, plot_name)
    
    cat(sprintf("  - Saving plot to: %s\n", plot_path))
    ggsave(plot_path, p, width = 10, height = 8)
  } else {
    cat("  - WARNING: No data available for this parameter combination, skipping plot\n")
  }
}

# Generate MAE plots
cat("\n=== GENERATING MAE PLOTS ===\n")
metric <- 'mae'

for(param_idx in seq_len(nrow(params))) {
  snr_target_val <- params$snr_target[param_idx]
  snr_source_val <- params$snr_source[param_idx]
  prop_features_val <- params$prop_features[param_idx]
  response_fn_source_val <- params$response_fn_source[param_idx]

  cat(sprintf("\nGenerating MAE plot %d/%d with parameters:\n", param_idx, nrow(params)))
  cat(sprintf("  - Source SNR: %s\n", snr_source_val))
  cat(sprintf("  - Target SNR: %s\n", snr_target_val))
  cat(sprintf("  - Feature Proportion: %s\n", prop_features_val))
  cat(sprintf("  - Response Function: %s\n", response_fn_source_val))

  filtered_data <- results %>% filter(
    snr_target == snr_target_val,
    snr_source == snr_source_val,
    prop_features == prop_features_val,
    response_fn_source == response_fn_source_val,
    model_id != 'source',
    model_type != "pred_ensemble" & model_type != "pred_0",
    split == "test_0"
  )

  cat(sprintf("  - Filtered data has %d rows\n", nrow(filtered_data)))

  if(nrow(filtered_data) > 0) {
    p <- filtered_data %>%
      ggplot(
        aes(
          x = as.factor(num_samples_source),
          y = mae,
          color = model_id
        )
      ) +
      geom_boxplot() +
      facet_grid(model ~ num_samples_target) +
      theme_bw() +
      ggtitle(
        paste0(
          "Source SNR = ", snr_source_val, "\n",
          "Target SNR = ", snr_target_val, "\n",
          "Num Features / Sample Size = ", prop_features_val, "\n",
          "Response Type = ", response_fn_source_val, "\n"
        )
      ) + 
      xlab("Source Sample Size") +
      ylab("MAE Score")

    plot_name <- paste0(paste(metric, snr_source_val, snr_target_val, prop_features_val, response_fn_source_val, sep = '_'), '.png')
    plot_path <- file.path(plots_dir, plot_name)

    cat(sprintf("  - Saving plot to: %s\n", plot_path))
    ggsave(plot_path, p, width = 10, height = 8)
  } else {
    cat("  - WARNING: No data available for this parameter combination, skipping plot\n")
  }
}

cat("\n=== SCRIPT FINISHED SUCCESSFULLY ===\n")
