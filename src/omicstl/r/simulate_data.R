library(dplyr)

## Load simulation utils
source(here::here("src", "omicsTL", "r", "data_simulation_utils.R"))

## Define simulation scenarios
source(here::here("src", "omicsTL", "r", "simulation_params.R"))

## Check for multi-node multiprocessing
max_workers <- 1
taskid <- NULL
if (Sys.getenv("SLURM_ARRAY_TASK_ID") != "") {
  max_workers <- as.numeric(Sys.getenv("MAX_WORKERS"))
  taskid <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID")) - 1
}

## Combine scenaro parameters into final scenario list
variable_scenario_parameters <- expand.grid(
    num_samples_source = num_samples_source,
    snr_source = snr_source,
    num_samples_target = num_samples_target,
    snr_target = snr_target
  )

simulation_scenarios <- expand.grid(
  variable_param_idx = seq_len(nrow(variable_scenario_parameters)),
  response_param_idx = seq_len(nrow(response_scenarios)),
  n_features = n_features
)

simulation_scenarios <- simulation_scenarios %>%
  as_tibble() %>%
  left_join(
    variable_scenario_parameters %>%
      mutate(variable_param_idx = seq_len(nrow(variable_scenario_parameters)))
  ) %>%
  left_join(
    response_scenarios %>%
      mutate(response_param_idx = seq_len(nrow(response_scenarios)))
  ) %>%
  mutate(
    response_fn_source = response_fn,
    response_fn_target = response_fn,
    response_parameters_source = response_parameters,
    response_parameters_target = response_parameters
  ) %>%
  dplyr::select(
    -variable_param_idx,
    -response_param_idx,
    -response_fn,
    -response_parameters
  )

## Load and process source data
source_data <- readRDS("/qfs/projects/ppi_timed/datagen/Datasets/Virus/LHV_MHAE001/Processed_Data/mhae001_pepdata_24hr.RDS")

source_data <- source_data %>%
  normalize_global(
    norm_fn = 'median',
    subset_fn = 'all',
    apply_norm = TRUE
  )

source_data <- source_data %>%
  molecule_filter() %>%
  applyFilt(source_data) %>%
  protein_quant(
    method = 'rrollup',
    combine_fn = "median"
  )

source_data <- impute_pmart_em(source_data)

## Load and process target data
target_data <- readRDS("/qfs/projects/ppi_timed/datagen/Datasets/Virus/LHV_MMVE002/Processed_Data/mmve002_pepdata_24hr.RDS")

target_data <- source_data %>%
  normalize_global(
    norm_fn = 'median',
    subset_fn = 'all',
    apply_norm = TRUE
  )

target_data <- target_data %>%
  molecule_filter() %>%
  applyFilt(target_data) %>%
  protein_quant(
    method = 'rrollup',
    combine_fn = "median"
  )

target_data <- impute_pmart_em(target_data)

# Ensure 1 to 1 mapping of biomolecules across sets and ensure they are in the
# same order
aligned_datasets <- align_datasets(
  source_data = source_data,
  target_data = target_data
)

long_source_data <- get_long_data(aligned_datasets$source_data)
long_target_data <- get_long_data(aligned_datasets$target_data)

selected_scenarios <- seq_len(nrow(simulation_scenarios))

if (!is.null(taskid)) {
  selected_scenarios <- which(
    sapply(
      seq_len(nrow(simulation_scenarios)),
      \(x) (x - 1) %% max_workers == taskid
    )
  )
  simulation_scenarios$original_index <- seq_len(nrow(simulation_scenarios))
}


## Simulate data
generate_from_data(
  init_source_data = long_source_data %>% dplyr::select(-subject_ids),
  init_target_data = long_target_data %>% dplyr::select(-subject_ids),
  scenarios = simulation_scenarios[selected_scenarios, ],
  replicates = 30,
  original_scenario_count = nrow(simulation_scenarios),
  samples_target_test = 50,
  seed = 42,
  out_dir = "./data/simulated_data/fixed_samplesize_large/",
  verbose = TRUE,
  n_cores = 60
)
