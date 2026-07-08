## Define simulation scenarios
response_fns <- list(
  \(data) {
    data[, 1] / 4 + data[, 2] / 4 + data[, 2] / 4 + data[, ncol(data)] / 4
  },
  \(data) {
    tanh(data[, 2]) + data[, 1] * data[, ncol(data)] ^ 2
  }
)

## Response scenarios: one row per (response function, response parameters) combo.
## response_fn:         the function mapping features → signal
## response_parameters: named list consumed by response_generator()
##   ncats   – number of classes (NULL = continuous/regression)
##   quantile – cut-point strategy: "quantile" (equal-frequency), "random",
##              or a numeric vector of explicit cut points
response_scenarios <- tibble::tibble(
  response_fn = response_fns,
  response_parameters = list(
    list(ncats = 2L, quantile = "quantile"),   # binary classification, linear signal
    list(ncats = 2L, quantile = "quantile")    # binary classification, nonlinear signal
  )
)

## Define source and target samples and SNR
num_samples_source <- c(50, 100, 200, 500)
snr_source <- c(0.1, 0.5, 2)

num_samples_target <- c(5, 10, 25, 50, 100)
snr_target <- c(0.1, 0.5, 2)

n_features <- c(250)