generate_from_data <- function(
    init_source_data,
    init_target_data,
    scenarios,
    replicates,
    original_scenario_count,
    samples_target_test,
    seed,
    out_dir,
    verbose = FALSE,
    n_cores = parallel::detectCores() - 1,
    method  = c("pca", "lc"),
    ...
) {
  method <- match.arg(method)
  # Create output directory if it doesn't exist
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  # Load required packages
  library(furrr)
  library(future)
  library(progressr)
  library(tibble)

  # Set seed for reproducibility
  set.seed(seed)
  seeds <- sample(seq_len(1e7), replicates * original_scenario_count)

  # Create all combinations of replicate and scenario
  jobs <- expand.grid(
    replicate = seq_len(replicates),
    scenario = seq_len(nrow(scenarios))
  )

  # Safely store additional arguments
  additional_args <- list(...)

  # Setup cleanup on exit
  old_plan <- future::plan()
  on.exit({
    message("Resetting to original plan...")
    future::plan(old_plan)
  }, add = TRUE)

  # Set up parallel processing
  future::plan(future::multisession, workers = n_cores)
  options(future.globals.maxSize = 500 * 1024^2)

  # Configure progress reporting
  if (verbose) {
    # Make sure a progress handler is registered
    progressr::handlers(global = TRUE)
    progressr::handlers("progress")
  }

  # Total number of jobs
  total_jobs <- nrow(jobs)
  completed_jobs <- 0

  # For progress tracking
  if (verbose) {
    pb <- txtProgressBar(min = 0, max = total_jobs, style = 3)
  }

  # Process in chunks to better manage progress
  chunk_size <- min(100, ceiling(total_jobs / 10))  # Process in reasonable chunks
  job_chunks <- split(seq_len(total_jobs), ceiling(seq_len(total_jobs) / chunk_size))

  results <- character(total_jobs)

  # Process each chunk
  for (chunk_idx in seq_along(job_chunks)) {
    current_indices <- job_chunks[[chunk_idx]]

    if (verbose) {
      message(sprintf("Processing chunk %d of %d (jobs %d-%d of %d)...",
                      chunk_idx, length(job_chunks),
                      min(current_indices), max(current_indices), total_jobs))
    }

    # Process current chunk in parallel
    chunk_results <- future_map(
      current_indices,
      function(i) {
        tryCatch({
          # Extract job parameters
          replicate <- jobs$replicate[i]
          scenario <- jobs$scenario[i]
          original_scenario <- scenarios$original_index[scenario]
          job_index <- original_scenario + (replicate - 1) * original_scenario_count
          replicate_seed <- seeds[job_index]

          # File paths
          source_path <- paste0(
            out_dir,
            paste("source_data", replicate, original_scenario, sep = "_"),
            ".csv"
          )
          target_path <- paste0(
            out_dir,
            paste("target_data", replicate, original_scenario, sep = "_"),
            ".csv"
          )
          target_test_path <- paste0(
            out_dir,
            paste("target_test_data", replicate, original_scenario, sep = "_"),
            ".csv"
          )
          sim_info_path <- paste0(
            out_dir,
            paste("sim_info", replicate, original_scenario, sep = "_"),
            ".RDS"
          )

          if (verbose) {
            message(sprintf("Processing replicate %d, scenario %d: %s | %s", replicate, original_scenario, target_path, sim_info_path))
          }

          # Skip if files already exist
          if (file.exists(source_path) && file.exists(target_path) && file.exists(sim_info_path)) {
            if (verbose) {
              message(sprintf("Files for replicate %d, scenario %d already exist, skipping", replicate, original_scenario))
            }
            return("skipped")
          }

          # Set seed for this specific job
          set.seed(replicate_seed)

          # Extract scenario parameters safely
          num_samples_source <- scenarios$num_samples_source[scenario]
          num_features_source <- scenarios$n_features[scenario]
          response_fn_source <- scenarios$response_fn_source[[scenario]]
          response_parameters_source <- scenarios$response_parameters_source[[scenario]]
          snr_source <- scenarios$snr_source[scenario]
          num_samples_target <- scenarios$num_samples_target[scenario]
          num_features_target <- scenarios$n_features[scenario]
          response_fn_target <- scenarios$response_fn_target[[scenario]]
          response_parameters_target <- scenarios$response_parameters_target[[scenario]]
          snr_target <- scenarios$snr_target[scenario]
          num_features <- max(num_features_source, num_features_target)

          if (method == "pca") {
            # ── PCA path: joint simulation preserves covariance structure ──────
            pca_out <- dat_generator_pca(
              source_data           = init_source_data,
              target_data           = init_target_data,
              n_output_samps_source = num_samples_source,
              n_output_samps_target = num_samples_target + samples_target_test,
              n_output_features     = num_features
            )

            # Helper: resolve cut-point / ncats from response_parameters list
            resolve_resp_params <- function(rp) {
              list(
                ncats  = if (!is.null(rp) && "ncats"    %in% names(rp)) as.numeric(rp[["ncats"]])    else NULL,
                thresh = if (!is.null(rp) && "quantile" %in% names(rp)) rp[["quantile"]] else NULL
              )
            }

            src_rp <- resolve_resp_params(response_parameters_source)
            src_resp <- response_generator(
              data   = pca_out$synth_source,
              func   = response_fn_source,
              SNR    = snr_source,
              ncats  = src_rp$ncats,
              thresh = src_rp$thresh
            )
            src_id <- ifelse(any(names(src_resp$response_data) %in% "cat_y"), "cat_y", "y")

            # Apply source cut points to target to keep category boundaries consistent
            tgt_resp <- response_generator(
              data   = pca_out$synth_target,
              func   = response_fn_target,
              SNR    = snr_target,
              ncats  = src_rp$ncats,
              thresh = src_resp$cut_points
            )
            tgt_id <- ifelse(any(names(tgt_resp$response_data) %in% "cat_y"), "cat_y", "y")

            source_data_out      <- cbind(response = src_resp$response_data[, src_id], pca_out$synth_source)
            target_data_full_out <- cbind(response = tgt_resp$response_data[, tgt_id], pca_out$synth_target)
            cut_point_out        <- src_resp$cut_points

          } else {
            # ── LC path (legacy): correlation-attenuated linear-combination sim ─
            source_args <- c(
              list(
                data                = init_source_data,
                num_features        = num_features,
                num_samples         = num_samples_source,
                response_fn         = response_fn_source,
                response_parameters = response_parameters_source,
                snr                 = snr_source
              ),
              additional_args
            )
            source_lc <- do.call(data_generator_wrapper, source_args)

            target_args <- c(
              list(
                data                = init_target_data,
                num_features        = num_features,
                num_samples         = num_samples_target + samples_target_test,
                response_fn         = response_fn_target,
                response_parameters = response_parameters_target,
                snr                 = snr_target,
                cut_point           = source_lc$cut_point,
                prior_lc_info       = source_lc$lc_info
              ),
              additional_args
            )
            target_lc <- do.call(data_generator_wrapper, target_args)

            source_data_out      <- source_lc$data
            target_data_full_out <- target_lc$data
            cut_point_out        <- source_lc$cut_point
          }

          # Split target data into train and test
          target_data_train <- target_data_full_out[seq_len(num_samples_target), ]
          target_data_test  <- target_data_full_out[seq(num_samples_target + 1, num_samples_target + samples_target_test), ]

          # Create simulation info
          sim_info <- tibble::tibble(
            num_features_source = num_features_source,
            num_samples_source = num_samples_source,
            snr_source = snr_source,
            response_fn_source = list(response_fn_source),
            num_features_target = num_features_target,
            num_samples_target = num_samples_target,
            snr_target = snr_target,
            response_fn_target = list(response_fn_target),
            replicate_seed = replicate_seed
          )

          # Save files
          write.csv(source_data_out, source_path, row.names = FALSE)
          write.csv(target_data_train, target_path, row.names = FALSE)
          write.csv(target_data_test, target_test_path, row.names = FALSE)
          saveRDS(sim_info, sim_info_path)

          if (verbose) {
            message(sprintf("Completed replicate %d, scenario %d", replicate, original_scenario))
          }

          return("success")
        }, error = function(e) {
          message(sprintf("Error in replicate %d, scenario %d: %s", replicate, original_scenario, e$message))
          return(paste("error:", e$message))
        })
      },
      .options = furrr_options(seed = TRUE, globals = TRUE)
    )

    # Store results
    results[current_indices] <- chunk_results

    # Update progress
    completed_jobs <- completed_jobs + length(current_indices)
    if (verbose) {
      setTxtProgressBar(pb, completed_jobs)
    }
  }

  # Close progress bar
  if (verbose) {
    close(pb)
  }

  # Count results
  success_count <- sum(results == "success", na.rm = TRUE)
  error_count <- sum(grepl("^error:", results), na.rm = TRUE)
  skipped_count <- sum(results == "skipped", na.rm = TRUE)

  # Print a more detailed error summary if there were errors
  if (error_count > 0) {
    error_messages <- unique(results[grepl("^error:", results)])
    message("\nError summary:")
    for (err_msg in error_messages) {
      count <- sum(results == err_msg, na.rm = TRUE)
      message(sprintf("  - %s: %d occurrences", gsub("^error: ", "", err_msg), count))
    }
  }

  message(sprintf("\nData generation complete: %d successful, %d failed, %d skipped",
                  success_count, error_count, skipped_count))

  return(invisible(NULL))
}

data_generator_wrapper <- function(
  data,
  num_features,
  num_samples,
  response_fn,
  response_parameters,
  snr,
  cut_point = NULL,
  ...
) {
  out <- dat_generator(
    source_data = data,
    n_output_features = num_features,
    n_output_samps = num_samples,
    ...
  )
  data <- out$synth_data

  if (!is.null(cut_point)) {
	ncats <- length(cut_point) + 1
	thresh <- cut_point
  } else {
	if (any(names(response_parameters) %in% "ncats")) {
		ncats <- as.numeric(response_parameters[["ncats"]])
	} else {
		ncats <- NULL
	}

	if (any(names(response_parameters) %in% "quantile")) {
		thresh <- response_parameters[["quantile"]]
	} else {
		thresh <- NULL
	}
  }

  response <- response_generator(
    data = data,
    func = response_fn,
    SNR = snr,
    ncats = ncats,
    thresh = thresh
  )

  response_id <- ifelse(
    any(names(response$response_data) %in% "cat_y"),
    "cat_y",
    "y"
  )

  data <- cbind(response = response$response_data[, response_id], data)
  return(list(data = data, lc_info = out$lc_info, cut_point = response$cut_points))
}

get_long_data <- function(pmart_object, ...) {
  subject_ids <- pmart_object$f_data[, pmartR::get_fdata_cname(pmart_object)]
  feature_ids <- pmart_object$e_data[, pmartR::get_edata_cname(pmart_object)]
  feature_id_idx <- which(names(pmart_object$e_data) == pmartR::get_edata_cname(pmart_object))

  long_data <- t(pmart_object$e_data[, -feature_id_idx])
  colnames(long_data) <- feature_ids

  args <- list(...)
  if (length(args) != 0) {
    meta_feature_idx <- which(names(pmart_object$f_data) %in% args)
    meta_data <- pmart_object$f_data %>% dplyr::select(dplyr::all_of(meta_feature_idx))
    long_data <- cbind(subject_ids = subject_ids, data.frame(meta_data), data.frame(long_data))
  } else {
    long_data <- cbind(subject_ids = subject_ids, data.frame(long_data))
  }

  return(long_data)
}

impute_pmart_em <- function(data) {
  feature_label_idx <- which(names(data$e_data) == get_edata_cname(data))
  subject_label_idx <- which(names(data$e_data) != get_edata_cname(data))
  data_for_imputation <- data$e_data[, -feature_label_idx]
  impute_res <- mvdalab::imputeEM(data_for_imputation)
  data$e_data[, subject_label_idx] <- impute_res$Imputed.DataFrames[[1]]
  return(data)
}

align_datasets <- function(source_data, target_data) {
  source_id <- get_edata_cname(source_data)
  target_id <- get_edata_cname(target_data)

  source_data_ids <- source_data$e_data[, source_id]
  target_data_ids <- target_data$e_data[, target_id]

  source_keep_ids <- which(source_data_ids %in% target_data_ids)
  target_keep_ids <- which(target_data_ids %in% source_data_ids)

  source_data$e_data <- source_data$e_data[source_keep_ids, ]
  target_data$e_data <- target_data$e_data[target_keep_ids, ]

  source_data$e_data <- source_data$e_data[order(source_data$e_data[, source_id]), ]
  target_data$e_data <- target_data$e_data[order(target_data$e_data[, target_id]), ]

  if (!all.equal(source_data$e_data[,source_id], target_data$e_data[,1])) stop("Alignment failed")

  return(list(source_data = source_data, target_data = target_data))
}


#' Generate new datasets using patterns in an existing dataset
#'
#' @param source_data A matrix with predictor information only OR a pmartR object.
#' For a matrix, rows should be samples, and columns should be features.
#' @param n_output_features A non-zero, positive integer value. Dictates how many
#' synthetic features are generated
#' @param n_output_samps A non-zero, positive integer value. Dictates how many
#' synthetic samples are generated
#' @param scale_sample_sd A boolean value. Controls whether or not the synthetic data
#' sample standard deviations should be scaled to randomly sampled sample
#' standard deviations from the source data
#' @param weight_func A sampling function (default = runif) to be used
#' in generating weights for the linear combinations of source data predictors that
#' generate synthetic predictors.
#' @param max_n_weights A positive integer value greater than 1. Dictates how many
#' of the source features are randomly selected for defining the linear combination
#' to generate a new synthetic feature
#' @param normalize_weights A boolean value. Controls whether the weights generated for
#' the linear combination should be normalized such that they sum to 1.
#' @param group_data If input is a matrix, you may provide a vector of strings
#' indicating the group of each of the source_data samples. Default is NULL, indicating
#' no grouping information is known. Currently assumes only one grouping variable but,
#' in theory, multiple groups could be accommodated by pasting group labels together,
#' e.g. Sex (M, F), Race (White, Nonwhite) ->  M_White, M_Nonwhite, F_White, F_Nonwhite.
#' If input is a pmartR object, you must provide a single string which specifies
#' the name of the column of f_data to read grouping information from or a vector
#' containing the group of each of the samples in the pmartR object.
#' @param prior_lc_info A dataframe containing the information used to generate a prior
#' set of synthetic biomolecules. Default is NULL if there has been no prior generation
#' of synthetic data
#' @param reuse_scaling_centering A boolean value. Controls whether the scaling and
#' centering data stored in prior_lc_info should be used to rescale and recenter the
#' synthetic data.
#' @param ... additional arguments passed as parameters to the sampling function specified
#' for weight_func
#' @return A named list. Element "synth_data" contains the generated data. This is a
#' matrix if source_data is a matrix or a pmartR object if source_data is a pmartR
#' object. Element "lc_info" contains information about the linear combination
#' selections. If the group_data parameter is specified, element "assigned_groups"
#' contains the groups specified for each sample.
#'
#' @examples
#'
#' library(pmartRdata)
#' library(ggplot2)
#' library(pmartR)
#' library(pcaMethods)
#' library(patchwork)
#' library(moments)
#'
#' set.seed(2)
#'
#' data("udn_metab_qc")
#'
#' og_subset <- udn_metab_qc$f_data %>%
#'   dplyr::filter(BatchName %in% c("Pr01", "Pr09")) %>%
#'   .$SampleID
#' og_dat <- udn_metab_qc$e_data %>%
#'   dplyr::select(Metabolite, dplyr::all_of(og_subset))
#' rownames(og_dat) <- make.names(og_dat$Metabolite)
#'
#' # Goal: Generate simulated biomolecules as linear combinations of existing
#' # biomolecules. Linear combinations should be such that the coefficients
#' # sum to 1, e.g. SynthBioMol1 = 0.1*OGBioMol1 + 0.5*OGBioMol2 + 0.4*OGBioMol3.
#'
#' tog_dat <- og_dat %>%
#'   dplyr::select(-Metabolite) %>%
#'   t(.)
#' tog_dat <- tog_dat[, -which(apply(!is.na(tog_dat), 2, sum) == 0)]
#' # For the sake of convenience, will impute all remaining missing observations with 0
#' tog_dat[is.na(tog_dat)] <- 0
#'
#' synth_dat_info1 <- dat_generator(source_data = tog_dat,
#'                                  n_output_features = 10,
#'                                  n_output_samps = 10,
#'                                  scale_sample_sd = FALSE,
#'                                  weight_func = runif,
#'                                  # max_n_weights = 224,
#'                                  normalize_weights = TRUE,
#'                                  # group_data = udn_metab_qc$f_data %>%
#'                                  #   dplyr::filter(BatchName %in% c("Pr01", "Pr09")) %>%
#'                                  #   .$BatchName,
#'                                  prior_lc_info = NULL)
#'
#' synth_dat_info2 <- dat_generator(source_data = tog_dat,
#'                                  n_output_features = 20,
#'                                  n_output_samps = 10,
#'                                  scale_sample_sd = FALSE,
#'                                  weight_func = runif,
#'                                  # max_n_weights = 224,
#'                                  normalize_weights = TRUE,
#'                                  # group_data = udn_metab_qc$f_data %>%
#'                                  #   dplyr::filter(BatchName %in% c("Pr01", "Pr09")) %>%
#'                                  #   .$BatchName,
#'                                  prior_lc_info = synth_dat_info1$lc_info)
#'
#' View(synth_dat_info1$lc_info)
#' View(synth_dat_info2$lc_info)
#'
#' # Note that correlation may still be low despite the predictor being "the same"
#' # See example below that uses partitions of source data for proof.
#' cor(synth_dat_info1$synth_data[,3], synth_dat_info2$synth_data[,3])
#'
#'
#' tog_subset1_idx <- sample(1:nrow(tog_dat), 6)
#' tog_subset2_idx <- (1:nrow(tog_dat))[-tog_subset1_idx]
#' tog_subset2_idx <- tog_subset2_idx[-length(tog_subset2_idx)]
#'
#' tog_subset1 <- tog_dat[sample(tog_subset1_idx),]
#' tog_subset2 <- tog_dat[sample(tog_subset2_idx),]
#'
#' # Correlation may still be low despite the predictor being "the same"
#' cor(tog_subset1[,2], tog_subset2[,2])
#'
#' synth_metabData <- dat_generator(source_data = pmartRdata::metab_object,
#'                                  n_output_features = 10,
#'                                  n_output_samps = 10,
#'                                  scale_sample_sd = FALSE,
#'                                  weight_func = runif,
#'                                  normalize_weights = TRUE,
#'                                  group_data = "Phenotype")
#'
#' @export
dat_generator <- function(source_data,
                          n_output_features,
                          n_output_samps,
                          scale_sample_sd = FALSE,
                          weight_func = runif,
                          max_n_weights = max(2, floor(ncol(source_data)*0.05)),
                          normalize_weights = TRUE,
                          group_data = NULL,
                          prior_lc_info = NULL,
                          reuse_centering_scaling = TRUE,
                          ...){

  # Check to make sure group_data is vector of character
  if (!is.null(group_data)) {
    if (!inherits(group_data, "character")) {
      error("group_data must either be a vector of strings or NULL")
    }
  }

  # Check for pmartR data object
  allowed_pmart_types <- c("isobaricpepData", "pepData", "proData", "lipidData", "nmrData", "metabData")
  pmartR_object_type <- allowed_pmart_types[which(inherits(source_data, allowed_pmart_types, TRUE) == 1)]
  if (length(pmartR_object_type) == 1) {
    # Require group_data when pmartR source_data provided
    if (is.null(group_data)) {
      error("group_data parameter must be specified if source_data is a pmartR object")
    }

    # Extract groups from f_data if column name is specified
    pmartR_group_colname <- "Group"
    if (length(group_data) == 1) {
      pmartR_group_colname <- group_data
      group_data <- source_data$f_data[,which(names(source_data$f_data) == group_data)]
    }

    # Use e_data as source data, but don't include edata_cname
    pmartR_object <- source_data
    edata_cname <- pmartR::get_edata_cname(pmartR_object)
    source_data <- pmartR_object$e_data[
      ,-which(names(pmartR_object$e_data) == edata_cname)]
    source_data <- t(as.matrix(source_data))
    colnames(source_data) <- pmartR_object$e_data[,which(names(pmartR_object$e_data) == edata_cname)]
  } else {
    pmartR_object_type <- NULL
  }

  if(!is.null(group_data)){
    # This randomly assigns groups to each synthetic sample, with
    # sampling probabilities based on the empirical probabilities
    # of the source data
    grp_randomizer <- sample(unique(group_data), n_output_samps, replace = TRUE,
                             prob = as.numeric(table(group_data)/sum(table(group_data))))
  }

  synth_dat <- matrix(NA, nrow = n_output_samps, ncol = n_output_features)
  # Object to store the source predictor names, indices, sample indices, and weights used for
  # each new synthetic feature instance.
  lc_info <- vector("list", length = n_output_features)
  for(i in 1:n_output_features){

    # Determine if the current number of sythentic predictors
    # is greater than the amount in prior_lc_info
    num_prior_synthpreds <- -Inf
    if(!is.null(prior_lc_info)){
      num_prior_synthpreds <- max(as.numeric(gsub("synthpred_", "", prior_lc_info$synthpred)))
    }

    if(i <= num_prior_synthpreds){
      # Only if the current amount of synthetic predictors is less than or equal to
      # the amount in prior_lc_info will we take the weights in
      # prior_lc_info to generate new predictors. Otherwise, we will
      # generate synthetic predictors as usual.

      # Extract info from relevant predictor
      temp_prior_lc_info <- prior_lc_info %>%
        dplyr::select(synthpred, og_inds, og_names, lc_weights, sd, meanval) %>%
        dplyr::filter(synthpred == paste0("synthpred_", i))

      n_weights <- nrow(temp_prior_lc_info)
    } else {
      # Randomize the number of original biomolecules used for the combination.
      # Should be at least 2, but to avoid excessively high correlations between synthetic
      # features, we need to set a max number of biomols for a linear combination to be a
      # lower proportion of total biomolecules
      if(max_n_weights < 2) {
        stop("max_n_weights must be a positive integer value greater than 1")
      }

      n_weights <- sample(2:max_n_weights, 1)
    }

    if (i <= num_prior_synthpreds) {
      # Use same weights as those in the prior_lc_info
      norm_weights <- temp_prior_lc_info$lc_weights

      # Extract the same variables as those used in the prior_lc_info
      og_inds <- as.numeric(sapply(temp_prior_lc_info$og_names, function(x){
        which(colnames(source_data) == x)
      }))

    } else {

      if(normalize_weights){
        # Sample n_weights from a uniform distribution and normalize to the sum so that
        # the weights sum to 1
        raw_weights <- weight_func(n_weights, ...)
        # raw_weights <- weight_func(n_weights, shape1 = 0.5, shape2 = 0.5)
        norm_weights <- raw_weights/sum(raw_weights)

      } else{
        # Sample n_weights from a uniform distribution and normalize to the sum so that
        # the weights sum to 1
        raw_weights <- weight_func(n_weights, ...)
        norm_weights <- raw_weights
      }

      # Sample original biomol indices to determine which are used in the computation
      # of the new, synthetic biomol.
      # Thinking about the transfer learning scenario, these indices must be saved
      # so that we are able to recreate the "same" synthetic predictors based on a
      # different source dataset with shared predictors as the current source dataset.

      # Only sample from biomol indices which have at least two nonmissing values for each group
      nonmissing_pred_inds <- 1:ncol(source_data)

      if (!is.null(group_data)) {
        unique_groups <- unique(group_data)
      } else {
        unique_groups <- 1
      }

      for (group in unique_groups) {
        if (!is.null(group_data)) {
          group_samp_inds <- which(group_data == group)
        } else {
          group_samp_inds <- 1:nrow(source_data)
        }

        current_group_nonmissingness <- !is.na(source_data[group_samp_inds,])
        nonmissing_pred_inds <- intersect(
          nonmissing_pred_inds,
          which(colMeans(current_group_nonmissingness) >= 2 / length(group_samp_inds))
        )
      }

      og_inds <- sample(nonmissing_pred_inds, n_weights)

    }

    if(!is.null(group_data)){

      # In the presence of groups, sample indices must be sampled by group to avoid
      # taking a linear combination of predictor values across different groups
      weight_samp_inds <- Reduce("rbind", lapply(grp_randomizer, function(x, data, groups){
        tempinds_group <- which(groups == x)

        # Exclude rows which have missing values
        new_weight_samp_inds <- NULL
        for (weight in 1:n_weights) {
          # Get the column indices which have missing values in the row (og_inds)
          # which will be used to draw source data values from later
          og_inds_na <- which(is.na(data[,og_inds[weight]]))

          # Exclude the columns not in the group and columns which will sample
          # a missing value
          tempinds <- setdiff(tempinds_group, og_inds_na)

          # Sample a column index
          new_weight_samp_inds <- c(new_weight_samp_inds, sample(tempinds, 1))
        }

        new_weight_samp_inds
      }, data = source_data, groups = group_data))

    } else{
      # If there are no groups, then sample indices can be freely sampled
      weight_samp_inds <- t(replicate(n_output_samps,
                                      sample(1:nrow(source_data), n_weights, replace = TRUE)))
    }

    # Generate synthetic predictor
    for(j in 1:nrow(weight_samp_inds)){
      # Pull the set of source sample and feature pairs that will
      # be used in the linear combination to generate a new sample
      temp_row_inds <- weight_samp_inds[j,]
      temp_col_inds <- og_inds
      temp_ind_df <- matrix(c(temp_row_inds, temp_col_inds), ncol = 2)

      synth_dat[j,i] <- source_data[temp_ind_df] %*% norm_weights

    }

    # Demonstration of how things should be multiplied to get the synthetic predictor
    # (source_data[temp_info$synthsamp_100[1], temp_info$og_inds[1]]*temp_info$lc_weights[1] +
    #     source_data[temp_info$synthsamp_100[2], temp_info$og_inds[2]]*temp_info$lc_weights[2])
    # synth_dat[100, 1]

    # Note that the above syntheticpredictor may have an inflated variance
    # by virtue of being a linear combination of predictors with
    # unknown variance. To remedy this, we can rescale according
    # to a randomly sampled variance from the source distribution
    # of variances
    og_sds <- apply(source_data, 2, sd, na.rm = TRUE)

    if (!is.null(prior_lc_info) && reuse_centering_scaling) {
      synth_sd_set <- temp_prior_lc_info[1, ]$sd
    } else {
      synth_sd_set <- sample(og_sds, 1)
    }

    new_synth_dat_col <- (synth_dat[,i]/sd(synth_dat[,i]))*synth_sd_set
    synth_dat[,i] <- new_synth_dat_col

    # Note that the above predictor may have mean(s) not reflective
    # of the mean(s) (plural in the presence of groups) within the data.
    # Therefore, we center the predictor and randomly sample among the
    # original data group means and re-center the predictor accordingly
    if(!is.null(group_data)) {

      # Center synthetic predictor so it has mean 0
      synthpred <- ave(new_synth_dat_col, grp_randomizer, FUN = \(x) scale(x, scale = FALSE))

      # Reuse centering if desired
      if (!is.null(prior_lc_info) && reuse_centering_scaling) {
        meanval <- temp_prior_lc_info[1, ]$meanval
      } else {
        # Determine all group means in original data
        gdat <- aggregate(data.frame(source_data), list(group_data), \(x) mean(x, na.rm = TRUE))

        # Sample from group means for each group
        meanval <- as.list(as.numeric(apply(gdat[,2:length(gdat)], 1, sample, size = 1)))
        names(meanval) <- gdat[,1]
        meanval <- meanval[[grp_randomizer[j]]]
      }

      # Re-center the 0-mean synthetic predictor to the sampled mean value(s)
      synth_dat[, i] <- sapply(1:length(synthpred), \(j) synthpred[j] + meanval)

    } else {

      # Center synthetic predictor so it has mean 0
      synthpred <- scale(new_synth_dat_col, scale = FALSE)

      if (!is.null(prior_lc_info) && reuse_centering_scaling) {
        meanval <- temp_prior_lc_info[1, ]$meanval
      } else {
        # Determine all means in original data
        gdat <- apply(source_data, 2, \(x) mean(x, na.rm = TRUE))

        # Sample from means
        meanval <- as.numeric(sample(gdat, 1))
      }

      # Re-center the 0-mean synthetic predictor to the sampled mean value
      synth_dat[, i] <- synthpred + meanval

    }

    # Package all of the information used for generating the synthetic data for
    # the current synthetic predictor
    temp_info <- data.frame(og_inds = og_inds,
                            og_names = colnames(source_data)[og_inds],
                            lc_weights = norm_weights,
                            sd = unname(synth_sd_set),
                            meanval = meanval)
    weight_samp_inds <- weight_samp_inds %>%
      t(.) %>%
      as.data.frame(.) %>%
      setNames(paste0("synthsamp_", 1:ncol(.))) %>%
      dplyr::mutate(og_names = temp_info$og_names)

    # Assign values to lc_info slot
    temp_info <- temp_info %>%
      dplyr::left_join(weight_samp_inds, by = "og_names")

    lc_info[[i]] <- temp_info %>%
      dplyr::mutate(synthpred = paste0("synthpred_", i)) %>%
      dplyr::relocate(synthpred)
  }

  # Not sure of most efficient way of storing this information.
  # This dataframe could grow very large depending on the
  # number of features, samples, and weights per feature.
  lc_info <- Reduce("rbind", lc_info)

  if(scale_sample_sd){
    # Rescale the sample variances so that they are in line with the empirical
    # sample variances
    og_samp_sds <- apply(source_data, 1, sd, na.rm = TRUE)

    for(j in 1:nrow(synth_dat)){
      # since the empirical distribution may have very few observations,
      # assume the sds follow a normal distribution with mean and variability
      # dictated by the empirical mean and variance of the sds
      synth_samp_sd_set <- rnorm(1, mean = mean(og_samp_sds), sd = sd(og_samp_sds))
      synth_dat[j,] <- (synth_dat[j,]/sd(synth_dat[j,]))*synth_samp_sd_set
    }
  }

  if (!is.null(pmartR_object_type)) {
    # Generate names for the samples
    edata <- data.frame(t(synth_dat))
    samp_names <- sapply(1:length(edata), \(x) paste0("Sample", x))
    names(edata) <- samp_names

    # Generate names for the features
    feat_names <- sapply(1:nrow(edata), \(x) paste0("Feature", x))
    edata <- cbind(edata_cname = feat_names, edata)
    names(edata)[1] <- pmartR::get_edata_cname(pmartR_object)

    # Generate fdata
    fdata_cname <- pmartR::get_fdata_cname(pmartR_object)
    fdata <- data.frame(Sample = samp_names, Group = grp_randomizer)
    names(fdata) <- c(fdata_cname, pmartR_group_colname)

    synth_pmartR <- do.call(
      paste0("as.", pmartR_object_type),
      list(
        e_data = edata,
        f_data = fdata,
        edata_cname = edata_cname,
        fdata_cname = fdata_cname
      ))
    attr(synth_pmartR, "synthetic_data") <- TRUE

    return(list(synth_data = synth_pmartR,
                assigned_groups = grp_randomizer,
                lc_info = lc_info))
  }

  if(!is.null(group_data)){

    return(list(synth_data = synth_dat,
                assigned_groups = grp_randomizer,
                lc_info = lc_info))

  } else{

    return(list(synth_data = synth_dat,
                lc_info = lc_info))

  }

}

#' Generate response data
#' @param data A dataframe for which you want to generate response data for
#' @param func a user-specified function dictating how the response mean should
#' be generated using predictors in data
#' @param SNR A number. Specifies the desired signal to noise ratio
#' @param ncats A number. Specifies the number of desired categories (assuming a
#' categorical outcome is desired). Defaults to NULL.
#' @param thresh Either a numeric vector of custom cutpoints or one of the character
#' values "random" or "quantile". Defaults to NULL.
#' Applicable only if ncats is specified (i.e. if the outcome should be categorical)
#' If "random", then ncats-1 values will be sampled from the generated signal and
#' used as cutpoints for categories.
#' If "quantile" cutpoints will be determined through every 1/ncats quantile of the
#' generated signal.
#' If a numeric vector of custom cutpoints, the user must specify ncats-1 values to
#' be used as cutpoints.
#'
#' @examples
#'
#' library(pmartRdata)
#' library(ggplot2)
#' library(pmartR)
#' library(pcaMethods)
#' library(patchwork)
#' library(moments)
#'
#' set.seed(2)
#'
#' data("udn_metab_qc")
#'
#' og_subset <- udn_metab_qc$f_data %>%
#'   dplyr::filter(BatchName %in% c("Pr01", "Pr09")) %>%
#'   .$SampleID
#' og_dat <- udn_metab_qc$e_data %>%
#'   dplyr::select(Metabolite, dplyr::all_of(og_subset))
#' rownames(og_dat) <- make.names(og_dat$Metabolite)
#'
#' tog_dat <- og_dat %>%
#'   dplyr::select(-Metabolite) %>%
#'   t(.)
#' tog_dat <- tog_dat[, -which(apply(!is.na(tog_dat), 2, sum) == 0)]
#' # For the sake of convenience, will impute all remaining missing observations with 0
#' tog_dat[is.na(tog_dat)] <- 0
#'
#' synth_dat_info1 <- dat_generator(source_data = tog_dat,
#'                                  n_output_features = 10,
#'                                  n_output_samps = 10,
#'                                  scale_sample_sd = FALSE,
#'                                  weight_func = runif,
#'                                  # max_n_weights = 224,
#'                                  normalize_weights = TRUE,
#'                                  # group_data = udn_metab_qc$f_data %>%
#'                                  #   dplyr::filter(BatchName %in% c("Pr01", "Pr09")) %>%
#'                                  #   .$BatchName,
#'                                  prior_lc_info = NULL)
#'
#' func <- function(data){
#'   y <- 2^(1/(3*data[,1])-1/(data[,2]))
#'   return(y)
#' }
#'
#' tempy_reg <- response_generator(data = synth_dat_info1$synth_data,
#'                             func = func,
#'                             SNR = 10)
#'
#' var(tempy_reg$response_data$y)/var(tempy_reg$response_data$noise)
#'
#' tempy_thresh1 <- response_generator(data = synth_dat_info1$synth_data,
#'                                     func = func,
#'                                     SNR = 10,
#'                                     ncats = 3,
#'                                     thresh = "random")
#' View(tempy_thresh1$response_data)
#'
#' tempy_thresh2 <- response_generator(data = synth_dat_info1$synth_data,
#'                                     func = func,
#'                                     SNR = 10,
#'                                     ncats = 3,
#'                                     thresh = "quantile")
#' View(tempy_thresh2$response_data)
#'
#' tempy_thresh3 <- response_generator(data = synth_dat_info1$synth_data,
#'                                     func = func,
#'                                     SNR = 10,
#'                                     ncats = 3,
#'                                     thresh = c(0.95, 0.99))
#' View(tempy_thresh3$response_data)
#'
#' @export
response_generator <- function(data, func, SNR,
                               ncats = NULL,
                               thresh = NULL){

  # uncorrupted signal
  pure_y <- func(data)

  # Compute variability of uncorrupted signal
  signal_var <- var(pure_y)

  # To achieve desired SNR, determine appropriate noise variance based
  # on observed signal_var: SNR = signal_var/noise_var -> noise_var = signal_var/SNR
  noise_var <- signal_var/SNR

  # Generate a vector of random noise with variance = noise_var
  # and add to the uncorrupted signal
  noise_vec <- rnorm(length(pure_y), mean = 0, sd = sqrt(noise_var))
  y <- pure_y + noise_vec

  if(!is.null(ncats) | !is.null(thresh)){
    if(is.null(ncats) | is.null(thresh)){
      stop("Both ncats and thresh must be specified.")
    }

    if(class(thresh) == "character"){
      if(thresh == "random"){
        # Remove min and max of y from consideration
        tempy <- y[-which(y == max(y) | y == min(y))]

        cuts <- sample(tempy, ncats-1)

      } else if(thresh == "quantile"){
        # determine quantiles percentages to use
        yquants <- (1:(ncats-1))*1/ncats

        cuts <- quantile(y, yquants)

      } else{
        stop("thresh must be either a numeric vector of cutpoints, or one of 'random' or 'quantile'.")
      }
    } else{
      cuts <- as.numeric(thresh[order(thresh)])
    }

    cuts <- as.numeric(cuts[order(cuts)])
    cat_y <- vector("numeric", length = length(y))
    for(i in 1:ncats){
      if(i == 1){
        cat_y[which(y <= cuts[i])] <- i
      } else if(i == ncats){
        cat_y[which(y > cuts[i-1])] <- i
      } else{
        cat_y[which(y > cuts[i-1] & y <= cuts[i])] <- i
      }
    }

    return(list(response_data = data.frame(y = y,
                                           cat_y = cat_y,
                                           noise = noise_vec,
                                           SNR = SNR),
                cut_points = cuts))

  } else{
    return(list(response_data = data.frame(y = y,
                                           noise = noise_vec,
                                           SNR = SNR)))
  }

}




# ── PCA-based synthetic data generation ───────────────────────────────────────

#' Generate synthetic source and target data using separate per-domain PCA
#'
#' Fits independent PCA on each domain so each domain's loading matrix captures
#' only its own covariance structure. New observations are sampled from
#' domain-specific Gaussian distributions in each domain's PC space and
#' projected back to the original feature space. Residual noise from the real
#' data's PCA complement is added to restore variance lost by truncation.
#'
#' Using separate SVDs avoids the source dominating the target's PC basis when
#' \code{n_source >> n_target}, which was the primary cause of target variance
#' inflation under the previous joint-SVD + shrinkage design.
#'
#' @param source_data  data.frame of real source data (samples × features).
#' @param target_data  data.frame of real target data (samples × features).
#'   Must share the same column names as \code{source_data}.
#' @param n_output_samps_source  Number of synthetic source samples to generate.
#' @param n_output_samps_target  Number of synthetic target samples to generate.
#' @param n_components  Number of PCs to retain per domain.  Defaults to
#'   \code{min(n_domain - 1, p)} for each domain independently, which is the
#'   maximum rank supported by that domain's sample size.  If specified, the
#'   value is silently clamped to each domain's maximum rank.
#' @param regularization  Small ridge constant added to the diagonal of each
#'   PC-space covariance matrix before sampling, to guard against near-
#'   singularity when \code{n_components} approaches the sample count.
#'   Default \code{1e-6}.
#'
#' @return A list with elements:
#' \describe{
#'   \item{synth_source}{data.frame of synthetic source data (samples × features).}
#'   \item{synth_target}{data.frame of synthetic target data (samples × features).}
#'   \item{pca_info}{List containing \code{V_source}, \code{V_target},
#'     per-domain means, component counts, and (if K > p) LC weights.}
#' }
#'
#' @export
dat_generator_pca <- function(
  source_data,
  target_data,
  n_output_samps_source,
  n_output_samps_target,
  n_components      = NULL,
  regularization    = 1e-6,
  n_output_features = NULL,
  snr               = NULL
) {

  # ── 0. Align feature sets ──────────────────────────────────────────────────
  common_features <- intersect(colnames(source_data), colnames(target_data))
  if (length(common_features) == 0)
    stop("source_data and target_data share no common column names.")
  if (length(common_features) < ncol(source_data) || length(common_features) < ncol(target_data))
    warning(sprintf(
      "Retaining %d common features (source had %d, target had %d).",
      length(common_features), ncol(source_data), ncol(target_data)
    ))
  source_data <- as.matrix(source_data[, common_features])
  target_data <- as.matrix(target_data[, common_features])
  storage.mode(source_data) <- "double"
  storage.mode(target_data) <- "double"

  n_source <- nrow(source_data)
  n_target <- nrow(target_data)
  p        <- length(common_features)

  # ── 1. Determine n_components ─────────────────────────────────────────────
  # Only source rank matters — the shared loading matrix V_src is fitted on
  # source data only; target is projected into the same space.
  max_comp_src <- min(n_source - 1L, p)

  if (is.null(n_components)) {
    n_comp_src <- max_comp_src
  } else {
    n_comp_src <- min(as.integer(n_components), max_comp_src)
  }

  # ── 2. Mean-impute missing values ──────────────────────────────────────────
  impute_col_means <- function(mat) {
    col_means <- colMeans(mat, na.rm = TRUE)
    col_means[!is.finite(col_means)] <- 0
    for (j in seq_len(ncol(mat))) {
      bad_idx <- which(!is.finite(mat[, j]))
      if (length(bad_idx)) mat[bad_idx, j] <- col_means[j]
    }
    mat
  }
  source_data <- impute_col_means(source_data)
  target_data <- impute_col_means(target_data)

  # ── 3. Source-only centering and scaling ──────────────────────────────────
  # Fit mean and SD on source only, then apply to both domains. After this,
  # source is zero-mean unit-variance per feature. Target is expressed relative
  # to the source distribution, so its nonzero PC-score mean encodes domain shift.
  source_mean_feat <- colMeans(source_data)
  source_sd_feat   <- apply(source_data, 2, sd)
  source_sd_feat[source_sd_feat < 1e-10] <- 1  # guard constant features

  source_cs <- sweep(sweep(source_data, 2, source_mean_feat, "-"), 2, source_sd_feat, "/")
  target_cs <- sweep(sweep(target_data, 2, source_mean_feat, "-"), 2, source_sd_feat, "/")

  # ── 4. PCA on scaled source — shared loading matrix ───────────────────────
  # SVD is fitted on source only; V_src is the shared loading matrix used for
  # both domains. Using the same V_src ensures source and target features live
  # in the same latent space, which is required for coherent transfer learning.
  svd_src <- svd(source_cs, nu = n_comp_src, nv = n_comp_src)
  V_src   <- svd_src$v   # p × n_comp_src

  # ── 5. Project both domains into source PC space ───────────────────────────
  Z_source <- source_cs %*% V_src   # n_source × n_comp_src  (zero-mean by construction)
  Z_target <- target_cs %*% V_src   # n_target × n_comp_src  (nonzero mean = domain shift)

  # ── 6. Estimate separate PC-score distributions ────────────────────────────
  # Source scores are zero-mean; target scores have nonzero mean (mu_target)
  # encoding the covariate shift between domains.
  mu_target    <- colMeans(Z_target)
  Sigma_source <- cov(Z_source) + diag(regularization, n_comp_src)
  Sigma_target <- cov(Z_target) + diag(regularization, n_comp_src)

  # ── 7. Sample synthetic scores from each domain's distribution ────────────
  Z_synth_source <- MASS::mvrnorm(n_output_samps_source, mu = rep(0, n_comp_src), Sigma = Sigma_source)
  Z_synth_target <- MASS::mvrnorm(n_output_samps_target, mu = mu_target,           Sigma = Sigma_target)

  if (!is.matrix(Z_synth_source)) Z_synth_source <- matrix(Z_synth_source, nrow = 1)
  if (!is.matrix(Z_synth_target)) Z_synth_target <- matrix(Z_synth_target, nrow = 1)

  # ── 8. Reconstruct in scaled space, add residual noise, then unscale ──────
  # All reconstruction work happens in the standardised space (source_cs /
  # target_cs) so that V_src and the residual complement are consistent.
  # Final sweep restores original feature units.
  add_pca_residual_noise <- function(synth_mat, real_mat, V, Sigma_pc, n_samps_out) {
    n_comps <- ncol(V)
    n_real  <- nrow(real_mat)
    p_feat  <- ncol(real_mat)

    pop_recon_var <- rowSums((V %*% Sigma_pc) * V)
    real_var      <- apply(real_mat, 2, var)
    if (all(pop_recon_var >= real_var * (1 - 1e-8))) return(synth_mat)

    real_c     <- sweep(real_mat, 2, colMeans(real_mat), "-")
    real_proj  <- real_c %*% V %*% t(V)
    real_resid <- real_c - real_proj

    rank_resid <- max(1L, min(n_real - 1L - n_comps, p_feat))
    svd_r      <- svd(real_resid, nu = 0L, nv = rank_resid)
    d_r        <- svd_r$d[seq_len(rank_resid)] / sqrt(max(n_real - 1L, 1L))
    Vr         <- svd_r$v[, seq_len(rank_resid), drop = FALSE]

    z_r   <- matrix(rnorm(n_samps_out * rank_resid), nrow = n_samps_out)
    z_r   <- sweep(z_r, 2L, d_r, "*")
    noise <- z_r %*% t(Vr)

    synth_mat + noise
  }

  recon_source_cs <- Z_synth_source %*% t(V_src)
  recon_target_cs <- Z_synth_target %*% t(V_src)

  recon_source_cs <- add_pca_residual_noise(recon_source_cs, source_cs, V_src,
                                            Sigma_source, n_output_samps_source)
  recon_target_cs <- add_pca_residual_noise(recon_target_cs, target_cs, V_src,
                                            Sigma_target, n_output_samps_target)

  # Reverse source-fitted standardisation for both domains
  recon_source <- sweep(sweep(recon_source_cs, 2, source_sd_feat, "*"), 2, source_mean_feat, "+")
  recon_target <- sweep(sweep(recon_target_cs, 2, source_sd_feat, "*"), 2, source_mean_feat, "+")

  # ── 8. Generate output features ────────────────────────────────────────────
  if (is.null(n_output_features)) {
    synth_source <- recon_source
    synth_target <- recon_target
    colnames(synth_source) <- common_features
    colnames(synth_target) <- common_features
    lc_weights <- NULL

  } else {
    n_output_features <- as.integer(n_output_features)
    if (n_output_features < 1)
      stop("n_output_features must be a positive integer.")

    if (n_output_features <= p) {
      # Option A: K ≤ p — random feature subsampling (exact statistics preserved)
      feat_idx     <- sample.int(p, n_output_features, replace = FALSE)
      synth_source <- recon_source[, feat_idx, drop = FALSE]
      synth_target <- recon_target[, feat_idx, drop = FALSE]
      colnames(synth_source) <- common_features[feat_idx]
      colnames(synth_target) <- common_features[feat_idx]
      lc_weights <- NULL

    } else {
      # Option B: K > p — variance-weighted sparse LC + dominant-contributor rescaling
      feat_var <- pmax(apply(source_data, 2, var, na.rm = TRUE), 1e-8)
      prob     <- feat_var / sum(feat_var)

      nnz <- max(3L, min(p, ceiling(sqrt(p))))
      W   <- matrix(0.0, nrow = p, ncol = n_output_features)
      for (k in seq_len(n_output_features)) {
        idx       <- sample.int(p, nnz, replace = FALSE, prob = prob)
        w         <- feat_var[idx]
        W[idx, k] <- w / sum(w)
      }

      synth_source <- recon_source %*% W
      synth_target <- recon_target %*% W

      ref_idx <- apply(W, 2, which.max)

      for (k in seq_len(n_output_features)) {
        j <- ref_idx[k]

        src_sd <- sd(synth_source[, k])
        if (src_sd > 0)
          synth_source[, k] <- (synth_source[, k] - mean(synth_source[, k])) /
                                src_sd * sd(source_data[, j]) +
                                mean(source_data[, j])

        tgt_sd <- sd(synth_target[, k])
        if (tgt_sd > 0)
          synth_target[, k] <- (synth_target[, k] - mean(synth_target[, k])) /
                                tgt_sd * sd(target_data[, j]) +
                                mean(target_data[, j])
      }

      colnames(synth_source) <- paste0("synthpred_", seq_len(n_output_features))
      colnames(synth_target) <- paste0("synthpred_", seq_len(n_output_features))
      lc_weights <- W
    }
  }

  # ── 9. Optional SNR noise injection ──────────────────────────────────────
  # Adds zero-mean Gaussian noise so that noise_var_j = signal_var_j / snr
  # for each feature j, giving the target SNR per feature. snr = NULL skips.
  if (!is.null(snr) && is.finite(snr) && snr > 0) {
    inject_snr_noise <- function(mat, snr_val) {
      feat_var <- pmax(apply(mat, 2, var), 0)
      noise_sd <- sqrt(feat_var / snr_val)
      noise_mat <- matrix(rnorm(nrow(mat) * ncol(mat)), nrow = nrow(mat))
      mat + sweep(noise_mat, 2, noise_sd, "*")
    }
    synth_source <- inject_snr_noise(synth_source, snr)
    synth_target <- inject_snr_noise(synth_target, snr)
  }

  list(
    synth_source      = data.frame(synth_source),
    synth_target      = data.frame(synth_target),
    source_scores_syn = Z_synth_source,
    target_scores_syn = Z_synth_target,
    pca_info = list(
      V_source         = V_src,
      source_mean_feat = source_mean_feat,
      source_sd_feat   = source_sd_feat,
      n_comp_source    = n_comp_src,
      lc_weights       = lc_weights
    )
  )
}


#' Wrapper: generate paired synthetic source + target data via PCA with responses
#'
#' Combines \code{\link{dat_generator_pca}} with \code{\link{response_generator}}
#' to produce feature matrices and response columns for both domains in a single
#' call.  The same response function and (for categorical responses) the same
#' cut points derived from the source are applied to the target, preserving the
#' source–target link.
#'
#' @inheritParams dat_generator_pca
#' @param response_fn      R function created by \code{response_function()}.
#' @param response_parameters  Named list with optional keys \code{"ncats"}
#'   (integer) and \code{"quantile"} ("quantile", "random", or numeric vector
#'   of cut points).  Pass \code{NULL} for a continuous response.
#' @param snr              Signal-to-noise ratio applied when generating the
#'   response column.  Default \code{1}.
#' @param cut_point        Pre-computed cut points (numeric vector) from a
#'   previous source-data call, used to impose the same category boundaries on
#'   the target response.  Pass \code{NULL} to derive cut points from the
#'   source synthetic data.
#'
#' @return A list with:
#' \describe{
#'   \item{source_data}{data.frame with response prepended to synthetic source features.}
#'   \item{target_data}{data.frame with response prepended to synthetic target features.}
#'   \item{cut_point}{Numeric vector of category cut points (NULL for continuous).}
#'   \item{pca_info}{PCA metadata from \code{dat_generator_pca}.}
#' }
#'
#' @export
data_generator_wrapper_pca <- function(
  source_data,
  target_data,
  n_output_samps_source,
  n_output_samps_target,
  response_fn,
  response_parameters = NULL,
  snr               = 1,
  cut_point         = NULL,
  n_components      = NULL,
  regularization    = 1e-6,
  n_output_features = NULL
) {

  out          <- dat_generator_pca(
    source_data           = source_data,
    target_data           = target_data,
    n_output_samps_source = n_output_samps_source,
    n_output_samps_target = n_output_samps_target,
    n_components          = n_components,
    regularization        = regularization,
    n_output_features     = n_output_features
  )
  source_synth <- out$synth_source
  target_synth <- out$synth_target

  # ── Resolve response parameters ────────────────────────────────────────────
  if (!is.null(cut_point)) {
    ncats <- length(cut_point) + 1
    thresh <- cut_point
  } else {
    ncats  <- if (!is.null(response_parameters) &&
                   any(names(response_parameters) %in% "ncats"))
                as.numeric(response_parameters[["ncats"]]) else NULL
    thresh <- if (!is.null(response_parameters) &&
                   any(names(response_parameters) %in% "quantile"))
                response_parameters[["quantile"]] else NULL
  }

  # ── Source response ────────────────────────────────────────────────────────
  src_resp <- response_generator(
    data  = source_synth,
    func  = response_fn,
    SNR   = snr,
    ncats = ncats,
    thresh = thresh
  )
  src_id <- if (any(names(src_resp$response_data) %in% "cat_y")) "cat_y" else "y"

  # ── Target response — domain-specific cut points ───────────────────────────
  tgt_resp <- response_generator(
    data   = target_synth,
    func   = response_fn,
    SNR    = snr,
    ncats  = ncats,
    thresh = thresh   # NULL for continuous; domain-specific quantile cuts for categorical
  )
  tgt_id <- if (any(names(tgt_resp$response_data) %in% "cat_y")) "cat_y" else "y"

  # ── Prepend response column ────────────────────────────────────────────────
  source_out <- cbind(response = src_resp$response_data[, src_id], source_synth)
  target_out <- cbind(response = tgt_resp$response_data[, tgt_id], target_synth)

  list(
    source_data       = source_out,
    target_data       = target_out,
    cut_point         = src_resp$cut_points,
    target_cut_point  = tgt_resp$cut_points,
    pca_info          = out$pca_info
  )
}


# ── Multi-omics PCA simulation ────────────────────────────────────────────────

#' Generate synthetic multi-omics source and target data using PCA
#'
#' Standardizes each omic layer to unit variance per feature, concatenates all
#' layers into a single matrix, then runs \code{\link{dat_generator_pca}} on
#' the concatenated data so that cross-omic correlations are captured in the
#' shared PC space.  The output is split back into per-omic blocks and
#' un-standardized to restore original scales.
#'
#' @param source_omics  Named list of data.frames (samples x features), one
#'   per omic layer (e.g. \code{list(prot = ..., rna = ...)}).
#' @param target_omics  Named list of data.frames with the same omic names.
#' @param n_output_samps_source  Number of synthetic source samples.
#' @param n_output_samps_target  Number of synthetic target samples.
#' @param n_output_features  Optional named list of integers specifying the
#'   desired output feature count per omic
#'   (e.g. \code{list(prot = 100, rna = 200)}).  Values \eqn{\le p} subsample
#'   randomly; values \eqn{> p} use variance-weighted LC mixing.  Omics not
#'   listed keep all their features.  Default \code{NULL} (keep all).
#' @param n_components  PCA components retained per domain (passed to
#'   \code{dat_generator_pca}).  Default \code{NULL} (max rank per domain).
#' @param regularization  Ridge constant added to PC-space covariances.
#'   Default \code{1e-6}.
#'
#' @return A list with:
#' \describe{
#'   \item{synth_source}{Named list of synthetic source data.frames, one per omic.}
#'   \item{synth_target}{Named list of synthetic target data.frames, one per omic.}
#'   \item{pca_info}{PCA metadata from the concatenated run.}
#' }
#'
#' @export
dat_generator_multiomics_pca <- function(
  source_omics,
  target_omics,
  n_output_samps_source,
  n_output_samps_target,
  n_output_features = NULL,
  n_components      = NULL,
  regularization    = 1e-6
) {

  # ── 0. Validate ────────────────────────────────────────────────────────────
  omic_names <- names(source_omics)
  if (is.null(omic_names) || !setequal(omic_names, names(target_omics)))
    stop("source_omics and target_omics must be named lists with the same omic names.")
  if (!is.null(n_output_features)) {
    unknown <- setdiff(names(n_output_features), omic_names)
    if (length(unknown))
      stop(sprintf("n_output_features contains unknown omic names: %s",
                   paste(unknown, collapse = ", ")))
  }

  # ── 1. Per-omic alignment and per-feature SD scaling ──────────────────────
  # SD is computed from the pooled real data so that neither domain's scale
  # inflates or deflates any omic's contribution to PCA.
  scale_sds          <- list()
  omic_widths        <- setNames(integer(length(omic_names)), omic_names)
  source_scaled_list <- list()
  target_scaled_list <- list()
  real_source_list   <- list()
  real_target_list   <- list()

  for (k in omic_names) {
    src_k <- as.matrix(source_omics[[k]]); storage.mode(src_k) <- "double"
    tgt_k <- as.matrix(target_omics[[k]]); storage.mode(tgt_k) <- "double"

    common_k <- intersect(colnames(src_k), colnames(tgt_k))
    if (length(common_k) == 0)
      stop(sprintf("Omic '%s': source and target share no common features.", k))

    src_k <- src_k[, common_k, drop = FALSE]
    tgt_k <- tgt_k[, common_k, drop = FALSE]

    pooled <- rbind(src_k, tgt_k)
    sds_k  <- apply(pooled, 2, sd, na.rm = TRUE)
    sds_k[sds_k < 1e-10] <- 1   # guard constant features

    scale_sds[[k]]          <- sds_k
    omic_widths[k]          <- length(common_k)
    source_scaled_list[[k]] <- sweep(src_k, 2, sds_k, "/")
    target_scaled_list[[k]] <- sweep(tgt_k, 2, sds_k, "/")
    real_source_list[[k]]   <- src_k
    real_target_list[[k]]   <- tgt_k
  }

  # ── 2. Concatenate all omic layers ─────────────────────────────────────────
  source_concat <- as.data.frame(do.call(cbind, source_scaled_list))
  target_concat <- as.data.frame(do.call(cbind, target_scaled_list))

  # ── 3. Separate per-domain PCA on concatenated scaled matrix ───────────────
  # Cross-omic correlations are preserved because all features from all omics
  # are present in a single matrix — the PC axes capture variation that spans
  # omic boundaries.
  pca_out <- dat_generator_pca(
    source_data           = source_concat,
    target_data           = target_concat,
    n_output_samps_source = n_output_samps_source,
    n_output_samps_target = n_output_samps_target,
    n_components          = n_components,
    regularization        = regularization,
    n_output_features     = NULL   # per-omic selection handled below
  )

  synth_src_mat <- as.matrix(pca_out$synth_source)
  synth_tgt_mat <- as.matrix(pca_out$synth_target)

  # ── 4. Split, un-scale, per-omic feature selection ─────────────────────────
  synth_source_omics <- list()
  synth_target_omics <- list()
  col_start <- 1L

  for (k in omic_names) {
    p_k     <- omic_widths[k]
    col_idx <- seq(col_start, col_start + p_k - 1L)
    col_start <- col_start + p_k

    sds_k      <- scale_sds[[k]]
    feat_names <- colnames(source_scaled_list[[k]])
    real_src_k <- real_source_list[[k]]
    real_tgt_k <- real_target_list[[k]]

    # Reverse the SD scaling
    src_block <- sweep(synth_src_mat[, col_idx, drop = FALSE], 2, sds_k, "*")
    tgt_block <- sweep(synth_tgt_mat[, col_idx, drop = FALSE], 2, sds_k, "*")
    colnames(src_block) <- feat_names
    colnames(tgt_block) <- feat_names

    n_out_k <- if (!is.null(n_output_features) && k %in% names(n_output_features))
                 as.integer(n_output_features[[k]]) else p_k

    if (n_out_k == p_k) {
      synth_source_omics[[k]] <- data.frame(src_block)
      synth_target_omics[[k]] <- data.frame(tgt_block)

    } else if (n_out_k < p_k) {
      # Random subsampling — same indices for source and target
      feat_idx <- sample.int(p_k, n_out_k, replace = FALSE)
      synth_source_omics[[k]] <- data.frame(src_block[, feat_idx, drop = FALSE])
      synth_target_omics[[k]] <- data.frame(tgt_block[, feat_idx, drop = FALSE])

    } else {
      # LC expansion: K > p_k — variance-weighted sparse mixing
      feat_var <- pmax(apply(real_src_k, 2, var, na.rm = TRUE), 1e-8)
      prob     <- feat_var / sum(feat_var)
      nnz      <- max(3L, min(p_k, ceiling(sqrt(p_k))))

      W <- matrix(0.0, nrow = p_k, ncol = n_out_k)
      for (j in seq_len(n_out_k)) {
        idx       <- sample.int(p_k, nnz, replace = FALSE, prob = prob)
        w         <- feat_var[idx]
        W[idx, j] <- w / sum(w)
      }

      src_exp <- src_block %*% W
      tgt_exp <- tgt_block %*% W
      ref_idx <- apply(W, 2, which.max)

      for (j in seq_len(n_out_k)) {
        r <- ref_idx[j]

        src_sd <- sd(src_exp[, j])
        if (src_sd > 0)
          src_exp[, j] <- (src_exp[, j] - mean(src_exp[, j])) /
                           src_sd * sd(real_src_k[, r]) + mean(real_src_k[, r])

        tgt_sd <- sd(tgt_exp[, j])
        if (tgt_sd > 0)
          tgt_exp[, j] <- (tgt_exp[, j] - mean(tgt_exp[, j])) /
                           tgt_sd * sd(real_tgt_k[, r]) + mean(real_tgt_k[, r])
      }

      colnames(src_exp) <- paste0(k, "_synthpred_", seq_len(n_out_k))
      colnames(tgt_exp) <- paste0(k, "_synthpred_", seq_len(n_out_k))
      synth_source_omics[[k]] <- data.frame(src_exp)
      synth_target_omics[[k]] <- data.frame(tgt_exp)
    }
  }

  list(
    synth_source = synth_source_omics,
    synth_target = synth_target_omics,
    pca_info     = pca_out$pca_info
  )
}
