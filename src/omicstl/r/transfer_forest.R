source(here::here("src", "omicstl", "r", "requirements.R"))
# source("/workspaces/timed-hpc/src/omicstl/r/requirements.R")

# x <- readRDS("/workspaces/timed-hpc/x.rds")
# y <- readRDS("/workspaces/timed-hpc/y.rds")
# rf_source<- readRDS("/workspaces/timed-hpc/refmod.rds")

RANDOMFOREST_NTREE <- 500

# Returns per-class sampsize vector for classification (minority class count repeated
# once per level). Returns NULL for regression so randomForest uses its default.
# Used by tryCatch retry blocks to recover from bootstrap sampling failures on
# small datasets where the default sampsize causes sample.int(0, ...) errors.
.rf_sampsize <- function(y) {
  if (is.factor(y)) rep(as.integer(min(table(y))), nlevels(y)) else NULL
}

#' Train an m0 target only model
#'
#' @param y the target response data
#' @param x the target feature data
#' @returns an m0 model
train_m0 <- function(
  y,
  x
) {
  res_0 <- tryCatch(
    randomForest::randomForest(y=y, x=x, ntree=RANDOMFOREST_NTREE),
    error = function(e) {
      randomForest::randomForest(y=y, x=x, ntree=RANDOMFOREST_NTREE,
                                 sampsize=.rf_sampsize(y))
    }
  )

  return(res_0)
}

#' Train an m1 variable importance transfer learning model
#'
#' @param y the target response data
#' @param x the target feature data
#' @param rf_source the random forests model trained on the source data (source model)
#' @returns an m1 model
train_m1 <- function(
  y,
  x,
  rf_source
) {
  var_importance_source <- importancenew(rf_source, 2)

  res_1 <- tryCatch(
    viRandomForests::viRandomForests(y=y, x=x, fprob=var_importance_source,
                                     ntree=RANDOMFOREST_NTREE, keep.forest=TRUE, importance=TRUE),
    error = function(e) {
      viRandomForests::viRandomForests(y=y, x=x, fprob=var_importance_source,
                                       ntree=RANDOMFOREST_NTREE, keep.forest=TRUE, importance=TRUE,
                                       sampsize=.rf_sampsize(y))
    }
  )

  return(res_1)
}

#' Train m2 model(s) to predict errors in the source model
#'
#' @param y the target response data
#' @param x the target feature data
#' @param rf_source the random forests model trained on the source data (source model)
#' @returns a single m2 model for regression problems, or a vector of m2 models, one for
#'   each class, for classification problems
train_m2 <- function(
  y,
  x,
  rf_source
) {
  pred_type <- ifelse(rf_source$type == "classification", "prob", "response")

  if (pred_type == "prob") {
    # Get probabilities per class
    y_res <- predict(rf_source, newdata = x, type=pred_type)

    res_deltas <- list()
    for (i in seq_len(dim(y_res)[2])) {
      # Calculate deltas between correct class probability and actual probability
      y_cat <- sapply(y, \(x) ifelse(x == i, 1, 0))
      y_delta <- y_cat - y_res[,i]

      # Train models to predict the probability delta
      res_deltas[[i]] <- randomForest::randomForest(
        y = y_delta,
        x = x,
        ntree = RANDOMFOREST_NTREE
      )
    }

    return(res_deltas)
  } else {
    y_delta <- y - predict(rf_source, newdata = x, type=pred_type)

    res_2 <- randomForest::randomForest(
      y = y_delta,
      x = x,
      ntree = RANDOMFOREST_NTREE
    )

    return(res_2)
  }
}

#' Train an m3 model to predict on a combination of source prediction results and
#' new data.
#'
#' @param y the target response data
#' @param x the target feature data
#' @param rf_source the random forests model trained on the source data (source model)
train_m3 <- function(
  y,
  x,
  rf_source
) {
  feature_names <- rownames(rf_source$importance)
  num_features <- length(feature_names)
  pred_type <- ifelse(rf_source$type == "classification", "prob", "response")

  y_source_hat <- predict(rf_source, newdata = x, type = pred_type)

  if(rf_source$type == "classification"){

    res_3 <- tryCatch(
      viRandomForests::viRandomForests(y=y, x=cbind(x, y_source_hat=y_source_hat),
                                       fprob=c(rep(1, num_features), rep(2, ncol(y_source_hat))),
                                       ntree=500, keep.forest=TRUE, importance=TRUE),
      error = function(e) {
        viRandomForests::viRandomForests(y=y, x=cbind(x, y_source_hat=y_source_hat),
                                         fprob=c(rep(1, num_features), rep(2, ncol(y_source_hat))),
                                         ntree=500, keep.forest=TRUE, importance=TRUE,
                                         sampsize=.rf_sampsize(y))
      }
    )

  } else{

    res_3 <- viRandomForests::viRandomForests(
    y = y,
    x = cbind(x, y_source_hat = y_source_hat),
    fprob= c(rep(1, num_features), rep(2, 1)),
    ntree = 500,
    keep.forest = TRUE,
    importance = TRUE)

  }


  return(res_3)
}

#' Train transfer models using an existing random forest model trained on source data
#'
#' @param y target response data
#' @param x target source data
#' @param rf_source the random forest to use for transfer learning
#' @returns a list of trained models including m_source, m_0, m_1, m_2, and m_3
train_trans_rf <- function(
    y,
    x,
    rf_source) {

  out <- vector("list", length = 5)
  names(out) <- c("m0", "m1", "m2", "m3", "m_source")

  # Target only model
  out[["m0"]] <- train_m0(y, x)

  # Variable importance transfer learning model
  out[["m1"]] <- train_m1(y, x, rf_source)

  # Error-based transfer learning model
  out[["m2"]] <- train_m2(y, x, rf_source)

  # Prediction-based transfer learning model
  out[["m3"]] <- train_m3(y, x, rf_source)

  # Source only model
  out[["m_source"]] <- rf_source

  return(out)
}

#' Predict data on only the source model
#'
#' @param models list of models returned by train_trans_rf
#' @param newdata a dataframe containing the new data to predict on
#' @param pred_type "prob" for categorical response or "response" for continuous response
predict_source <- function(
  models,
  newdata,
  pred_type
) {
  return(predict(models[['m_source']], newdata = newdata, type = pred_type))
}

#' Predict data on a model trained on the only the target data
#'
#' @param models list of models returned by train_trans_rf
#' @param newdata a dataframe containing the new data to predict on
#' @param pred_type "prob" for categorical response or "response" for continuous response
predict_m0 <- function(
  models,
  newdata,
  pred_type
) {
  pred_0 <- predict(models[['m0']], newdata = newdata, type = pred_type)
  return(pred_0)
}

#' Predict data using the transfer learning model
#'
#' @param models list of models returned by train_trans_rf
#' @param newdata a dataframe containing the new data to predict on
#' @param pred_type "prob" for categorical response or "response" for continuous response
predict_m1 <- function(
  models,
  newdata,
  pred_type
) {
  pred_1 <- predict(models[['m1']], newdata = newdata, type = pred_type)
  return(pred_1)
}

#' Predict data using error-based transfer learning
#'
#' @param models list of models returned by train_trans_rf
#' @param newdata a dataframe containing the new data to predict on
#' @param pred_source the predictions on the source model
#' @param pred_type "prob" for categorical response or "response" for continuous response
#'
#' @details This method of transfer learning uses a RandomForest model trained on the
#'   source data and then a RandomForest model trained on error deltas of the predictions
#'   of the source model on the target data. When using categorical prediction, a
#'   RandomForest model is trained on each class.
predict_m2 <- function(
  models,
  newdata,
  pred_source,
  pred_type
) {
  if (pred_type == "prob") {
    # Predict error deltas for each class
    num_classes <- length(models[['m_source']]$classes)
    num_samples <- dim(newdata)[1]

    pred_2_unnorm <- list()
    for (i in seq_len(num_classes)) {
      pred_2_deltas <- predict(models[['m2']][[i]], newdata = newdata, pred_type = pred_type)

      # Add deltas to original prediction probabilities
      pred_2_unnorm[[i]] <- pred_source[,i] + pred_2_deltas
    }

    # Normalize probabilities to between 0 and 1
    pred_2 <- NULL
    for (i in seq_len(num_samples)) {
      row <- unname(sapply(seq_len(num_classes), \(x) pred_2_unnorm[[x]][i]))
      if (min(row) < 0) {
        row <- row - min(row)
      }
      norm_class_probs <- row/sum(row)
      pred_2 <- c(pred_2, norm_class_probs)
    }

    return(matrix(pred_2, ncol=num_classes, byrow=TRUE))
  } else {
    # Add the error correction to the predictions
    pred_2_error <- predict(models[['m2']], newdata = newdata, pred_type = pred_type)
    pred_2 <- pred_source + pred_2_error
    return(pred_2)
  }
}

#' Predict data using source prediction injected transfer learning
#'
#' @param models list of models returned by train_trans_rf
#' @param newdata a dataframe containing the new data to predict on
#' @param pred_type "prob" for categorical response or "response" for continuous response
#'
#' @details This method of transfer learning uses a RandomForest model trained on the
#'   source data, predicts values on the source model, then combines it with the new data
#'   before predicting on that combined dataframe using a transfer learning model.
predict_m3 <- function(
  models,
  newdata,
  pred_type
) {
  newdata <- cbind(newdata, y_source_hat = predict(models[['m_source']], newdata = newdata, type = pred_type))
  pred_3 <- predict(models[["m3"]], newdata = newdata, type = pred_type)
  return(pred_3)
}

#' Generate prediction weights for the ensemble weighting method of m1 through m3
#'
#' @param models list of models returned by train_trans_rf
#' @param y_ensemble the ensemble response data used for training the ensemble model
#' @param x_ensemble the ensemble feature data used for training the ensemble model
#' @param pred_type "prob" for categorical response or "response" for continuous response
#'
#' @details This method combines the three TL models (m1, m2, m3) by training
#'   a RandomForest model on a separate ensemble training sample set, extracting the
#'   feature weights, and taking the weighted geometric mean model prediction.
#'   m0 (target-only) is excluded so the ensemble represents pure transfer learning.
predict_ens_weights <- function(
  models,
  y_ensemble,
  x_ensemble,
  pred_type
) {
  classes <- models[['m_source']]$classes
  num_classes <- length(classes)

  pred_ens_source_val <- predict_source(models, x_ensemble, pred_type)
  pred_ens_1_val <- predict_m1(models, x_ensemble, pred_type)
  pred_ens_2_val <- predict_m2(models, x_ensemble, pred_ens_source_val, pred_type)
  pred_ens_3_val <- predict_m3(models, x_ensemble, pred_type)

  if (pred_type == "prob") {
    pred_combine_ensemble = cbind(pred_ens_1_val, pred_ens_2_val, pred_ens_3_val)

    # Create a model to predict the class using the different model results
    m_ensemble <- randomForest::randomForest(
      y = factor(y_ensemble, levels = classes),
      x = pred_combine_ensemble
    )

    # Extract the importance values
    ensemble_importance <- importance(m_ensemble)[,1]
    num_classes <- length(ensemble_importance) / 3
    class_model_weights <- data.frame(
      matrix(
        sapply(
          seq_len(num_classes),
          \(x) sapply(
            1:3,
            \(y) ensemble_importance[((x - 1) * 3) + y]
          )
        ),
        ncol=3,
        byrow=TRUE
      )
    )
    colnames(class_model_weights) <- c("m1", "m2", "m3")

    return(class_model_weights)
  } else {
    m_ensemble <- randomForest::randomForest(
      y = y_ensemble,
      x = data.frame(
        m1 = pred_ens_1_val,
        m2 = pred_ens_2_val,
        m3 = pred_ens_3_val
      )
    )

    ensemble_importance <- importance(m_ensemble)[,1]
    ensemble_importance <- ensemble_importance / sum(ensemble_importance)

    return(ensemble_importance)
  }
}

#' Predict data using an ensemble weighting method of m1 through m3
#'
#' @param prev_preds a list of predictions from m1 through m3 (the three TL models)
#' @param ensemble_weights a list (for continuous response) or data.frame (for categorical
#'   response) of weights to use for ensemble prediction. These should be generated with
#'   \code{predict_ens_weights}
#' @param pred_type "prob" for categorical response or "response" for continuous response
#' @param num_classes the number of classes to predict (categorial only)
#'
#' @details see \code{predict_ens_weights} for details about the ensemble method
predict_ens <- function(
  prev_preds,
  ensemble_weights,
  pred_type,
  num_classes = NULL
) {
  if (pred_type == "prob") {
    # Get the previous predictions in a prob format
    pred_1 <- matrix(prev_preds[[1]], ncol = num_classes)
    pred_2 <- matrix(prev_preds[[2]], ncol = num_classes)
    pred_3 <- matrix(prev_preds[[3]], ncol = num_classes)

    pred_combine <- cbind(pred_1, pred_2, pred_3)

    num_samples <- dim(pred_combine)[1]

    class_scores_all <- c()
    for (sample in seq_len(num_samples)) {
      class_scores <- rep(1, num_classes)
      for (class in seq_len(num_classes)) {
        for (model in 1:3) {
          class_scores[class] <- class_scores[class] * pred_combine[sample, (model - 1) * num_classes + class] ^ ensemble_weights[class, model]
        }
      }
      class_scores <- class_scores / sum(class_scores)
      class_scores_all <- c(class_scores_all, class_scores)
    }

    pred_ensemble <- matrix(class_scores_all, ncol=num_classes, byrow=TRUE)

    return(pred_ensemble)
  } else {
    pred_ensemble <- prev_preds[[1]] * ensemble_weights[1] +
                        prev_preds[[2]] * ensemble_weights[2] +
                        prev_preds[[3]] * ensemble_weights[3]

    return(pred_ensemble)
  }
}

randomize_equal_response = function(responses) {
  max_resp_val <- max(responses)
  max_inds <- which(responses == max_resp_val)
  return(sample(max_inds, 1))
}

#' Predict data
#'
#' @param trans_rf_res the random forest trained using transfer learning
#' @param newdata new data
#' @param y_val (optional) target validation response data
#' @param x_val (optional) target validation source data
#' @param y_ensemble (optional) target ensemble response data, required if x_ensemble is present
#' @param x_ensemble (optional) target ensemble source data, required if y_ensemble is present
#' @param ensemble_weights (optional) ensemble weights from previous prediction. Do not use
#'   simultaneously with x_ensemble and y_ensemble.
predict_trans_rf = function(
    trans_rf_res,
    newdata,
    y_val = NULL,
    x_val = NULL,
    y_ensemble = NULL,
    x_ensemble = NULL,
    ensemble_weights = NULL
){
  if ((!is.null(x_ensemble) && is.null(y_ensemble)) ||
      (!is.null(y_ensemble) && is.null(x_ensemble))) {
    stop("y_ensemble and x_ensemble must both be provided if one is provided.")
  }

  if (!is.null(ensemble_weights) && (!is.null(x_ensemble) || !is.null(y_ensemble))) {
    ensemble_weights <- NULL
    warning(paste0(
      "y_ensemble/x_ensemble, and ensemble_weights were both provided. Ensemble weights will be ",
      "calculated using y_ensemble/x_ensemble and ensemble_weights will be ignored."
    ))
  }

  if (trans_rf_res[["m_source"]]$type == "classification") {
    pred_type <- "prob"
    classes <- levels(predict(trans_rf_res[["m0"]], type = 'response'))
  } else {
    pred_type <- "response"
  }

  #print("pred_source")
  pred_source <- predict_source(trans_rf_res, newdata, pred_type)
  if (!is.null(x_val)) {
    pred_source_val <- predict_source(trans_rf_res, x_val, pred_type)
  } else {
    pred_source_val <- NULL
  }

  #print("pred_0")
  pred_0 <- predict_m0(trans_rf_res, newdata, pred_type)
  if (!is.null(x_val)) {
    pred_0_val <- predict_m0(trans_rf_res, x_val, pred_type)
  } else {
    pred_0_val <- NULL
  }

  #print("pred_1")
  pred_1 = predict_m1(trans_rf_res, newdata, pred_type)
  if (!is.null(x_val)) {
    pred_1_val <- predict_m1(trans_rf_res, x_val, pred_type)
  } else {
    pred_1_val <- NULL
  }

  #print("pred_2")
  pred_2 <- predict_m2(trans_rf_res, newdata, pred_source, pred_type)
  if (!is.null(x_val)) {
    pred_2_val <- predict_m2(trans_rf_res, x_val, pred_source_val, pred_type)
  } else {
    pred_2_val <- NULL
  }

  # Need to specify colnames here since predict_m2() does not
  # automatically assign them.
  colnames(pred_2) <- colnames(pred_1)

  if (!is.null(x_val)) {
    colnames(pred_2_val) <- colnames(pred_1_val)
  }

  #print("pred_3")
  pred_3 <- predict_m3(trans_rf_res, newdata, pred_type)
  if (!is.null(x_val)) {
    pred_3_val <- predict_m3(trans_rf_res, x_val, pred_type)
  } else {
    pred_3_val <- NULL
  }

  #print("pred_ens")
  pred_ensemble <- NULL

  # Generate weights if x_ensemble and y_ensemble are provided
  if (!is.null(x_ensemble) && !is.null(y_ensemble)) {
    ensemble_weights <- predict_ens_weights(trans_rf_res, y_ensemble, x_ensemble, pred_type)
  }

  if (pred_type == "prob") {
    # for categorical response
    if (!is.null(ensemble_weights)) {
      pred_ensemble <- predict_ens(list(pred_1, pred_2, pred_3), ensemble_weights, pred_type, length(classes))

      if (!is.null(x_val)) {
        pred_ensemble_val <- predict_ens(list(pred_1_val, pred_2_val, pred_3_val), ensemble_weights, pred_type, length(classes))
      } else {
        pred_ensemble_val <- NULL
      }

      # Need to specify colnames here since predict_ens() does not
      # automatically assign them.
      colnames(pred_ensemble) <- colnames(pred_1)
      if (!is.null(x_val)) {
        colnames(pred_ensemble_val) <- colnames(pred_1_val)
      }
    } else {
      # If ensemble weights aren't provided, use equal weighting across m1, m2, m3
      pred_ensemble <- Reduce("*", list(pred_1, pred_2, pred_3))
      pred_ensemble <- apply(pred_ensemble, 1, function(x){x/sum(x)})
      pred_ensemble <- t(pred_ensemble)

      if (!is.null(x_val)) {
        pred_ensemble_val <- Reduce("*", list(pred_1_val, pred_2_val, pred_3_val))
        pred_ensemble_val <- apply(pred_ensemble_val, 1, function(x){x/sum(x)})
        pred_ensemble_val <- t(pred_ensemble_val)
      } else {
        pred_ensemble_val <- NULL
      }

      ensemble_weights <- data.frame(matrix(rep(1/3, length(classes) * 3), ncol=3))
      colnames(ensemble_weights) <- c("m1", "m2", "m3")
    }
  } else {
    # For continuous response
    if (!is.null(ensemble_weights)) {
      pred_ensemble <- predict_ens(list(pred_1, pred_2, pred_3), ensemble_weights, pred_type)

      if (!is.null(x_val)) {
        pred_ensemble_val <- predict_ens(list(pred_1_val, pred_2_val, pred_3_val), ensemble_weights, pred_type)
      } else {
        pred_ensemble_val <- NULL
      }
    } else {
      # If ensemble samples aren't provided, use equal weighting across m1, m2, m3
      pred_ensemble <- (pred_1 + pred_2 + pred_3) / 3

      if (!is.null(x_val)) {
        pred_ensemble_val <- (pred_1_val + pred_2_val + pred_3_val) / 3
      } else {
        pred_ensemble_val <- NULL
      }

      ensemble_weights <- rep(1/3, 3)
    }

  }

  out <- NULL
  if(pred_type == "prob") {
    pred_source_pred_class <- colnames(pred_source)[apply(pred_source, 1, randomize_equal_response)]
    pred_source_max_prob <- pred_source
    colnames(pred_source_max_prob) <- paste0("pred_source_prob_", colnames(pred_source_max_prob))
    pred_0_pred_class <- colnames(pred_0)[apply(pred_0, 1, randomize_equal_response)]
    pred_0_max_prob <- pred_0
    colnames(pred_0_max_prob) <- paste0("pred_0_prob_", colnames(pred_0_max_prob))
    pred_1_pred_class <- colnames(pred_1)[apply(pred_1, 1, randomize_equal_response)]
    pred_1_max_prob <- pred_1
    colnames(pred_1_max_prob) <- paste0("pred_1_prob_", colnames(pred_1_max_prob))
    pred_2_pred_class <- colnames(pred_2)[apply(pred_2, 1, randomize_equal_response)]
    pred_2_max_prob <- pred_2
    colnames(pred_2_max_prob) <- paste0("pred_2_prob_", colnames(pred_2_max_prob))
    pred_3_pred_class <- colnames(pred_3)[apply(pred_3, 1, randomize_equal_response)]
    pred_3_max_prob <- pred_3
    colnames(pred_3_max_prob) <- paste0("pred_3_prob_", colnames(pred_3_max_prob))
    pred_ensemble_pred_class <- colnames(pred_ensemble)[unlist(apply(pred_ensemble, 1, randomize_equal_response))]
    pred_ensemble_max_prob <- pred_ensemble
    colnames(pred_ensemble_max_prob) <- paste0("pred_ensemble_prob_", colnames(pred_ensemble_max_prob))

    if (!is.null(x_val)) {
      pred_source_val_pred_class <- colnames(pred_source_val)[apply(pred_source_val, 1, randomize_equal_response)]
      pred_source_val_max_prob <- pred_source_val
      colnames(pred_source_val_max_prob) <- paste0("pred_source_val_prob_", colnames(pred_source_val_max_prob))
      pred_0_val_pred_class <- colnames(pred_0_val)[apply(pred_0_val, 1, randomize_equal_response)]
      pred_0_val_max_prob <- pred_0_val
      colnames(pred_0_val_max_prob) <- paste0("pred_0_val_prob_", colnames(pred_0_val_max_prob))
      pred_1_val_pred_class <- colnames(pred_1_val)[apply(pred_1_val, 1, randomize_equal_response)]
      pred_1_val_max_prob <- pred_1_val
      colnames(pred_1_val_max_prob) <- paste0("pred_1_val_prob_", colnames(pred_1_val_max_prob))
      pred_2_val_pred_class <- colnames(pred_2_val)[apply(pred_2_val, 1, randomize_equal_response)]
      pred_2_val_max_prob <- pred_2_val
      colnames(pred_2_val_max_prob) <- paste0("pred_2_val_prob_", colnames(pred_2_val_max_prob))
      pred_3_val_pred_class <- colnames(pred_3_val)[apply(pred_3_val, 1, randomize_equal_response)]
      pred_3_val_max_prob <- pred_3_val
      colnames(pred_3_val_max_prob) <- paste0("pred_3_val_prob_", colnames(pred_3_val_max_prob))
      pred_ensemble_val_pred_class <- colnames(pred_ensemble_val)[unlist(apply(pred_ensemble_val, 1, randomize_equal_response))]
      pred_ensemble_val_max_prob <- pred_ensemble_val
      colnames(pred_ensemble_val_max_prob) <- paste0("pred_ensemble_val_prob_", colnames(pred_ensemble_val_max_prob))

      out <- data.frame(
        truth = y_val,
        pred_source = pred_source_pred_class,
        pred_source_max_prob,
        pred_0 = pred_0_pred_class,
        pred_0_max_prob,
        pred_1 = pred_1_pred_class,
        pred_1_max_prob,
        pred_2 = pred_2_pred_class,
        pred_2_max_prob,
        pred_3 = pred_3_pred_class,
        pred_3_max_prob,
        pred_ensemble = pred_ensemble_pred_class,
        pred_ensemble_max_prob,
        pred_source_val = pred_source_val_pred_class,
        pred_source_val_max_prob,
        pred_0_val = pred_0_val_pred_class,
        pred_0_val_max_prob,
        pred_1_val = pred_1_val_pred_class,
        pred_1_val_max_prob,
        pred_2_val = pred_2_val_pred_class,
        pred_2_val_max_prob,
        pred_3_val = pred_3_val_pred_class,
        pred_3_val_max_prob,
        pred_ensemble_val = pred_ensemble_val_pred_class,
        pred_ensemble_val_max_prob
      )
    } else {
      out <- data.frame(
        pred_source = pred_source_pred_class,
        pred_source_max_prob,
        pred_0 = pred_0_pred_class,
        pred_0_max_prob,
        pred_1 = pred_1_pred_class,
        pred_1_max_prob,
        pred_2 = pred_2_pred_class,
        pred_2_max_prob,
        pred_3 = pred_3_pred_class,
        pred_3_max_prob,
        pred_ensemble = pred_ensemble_pred_class,
        pred_ensemble_max_prob
      )
    }

    #pred_ensemble_max_prob[pred_ensemble_max_prob == 1.0] <- 0.9999
  } else {
    if (!is.null(x_val)) {
      out <- data.frame(
        pred_source = pred_source,
        pred_0 = pred_0,
        pred_1 = pred_1,
        pred_2 = pred_2,
        pred_3 = pred_3,
        pred_ensemble = pred_ensemble,
        pred_source_val = pred_source_val,
        pred_0_val = pred_0_val,
        pred_1_val = pred_1_val,
        pred_2_val = pred_2_val,
        pred_3_val = pred_3_val,
        pred_ensemble_val = pred_ensemble_val
      )
    } else {
      out <- data.frame(
        pred_source = pred_source,
        pred_0 = pred_0,
        pred_1 = pred_1,
        pred_2 = pred_2,
        pred_3 = pred_3,
        pred_ensemble = pred_ensemble
      )
    }
  }

  attr(out, "ensemble_weights") <- ensemble_weights

  # print(out)
  return(out)
}