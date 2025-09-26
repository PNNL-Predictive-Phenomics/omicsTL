source(here::here("src", "omicstl", "r", "requirements.R"))

#' Train a transfer model using an existing random forest
#'
#' @param y target data
#' @param x source data
#' @param rf_source the random forest to use for transfer learning
#' @param var_importance_source the variable importance scores associated with
#'        the rf_source
train_trans_rf <- function(
    y,
    x,
    rf_source,
    var_importance_source) {

  if (rf_source$type == "classification") {
    pred_type = "prob"
  } else {
    pred_type = "response"
  }

  out <- vector("list", length = 5)
  names(out) <- c("m0", "m1", "m2", "m3", "m_source")

  feature_names <- rownames(rf_source$importance)
  num_features <- length(feature_names)
  model_type <- rf_source$type

  res_0 <- randomForest::randomForest(
    y = y,
    x = x,
    ntree = 500
  )
  #target only model
  out[["m0"]] <- res_0

  res_1 <- viRandomForests::viRandomForests(
    y = y,
    x = x,
    fprob = var_importance_source,
    ntree = 500,
    keep.forest = TRUE,
    importance = TRUE
  )
  out[["m1"]] <- res_1

  if (model_type == "classification") {
    # Get probabilities per class
    y_res <- predict(rf_source, newdata = x, type="prob")

    res_deltas <- list()
    for (i in seq_len(dim(y_res)[2])) {
      # Calculate deltas between correct class probability and actual probability
      y_cat <- sapply(y, \(x) ifelse(x == i, 1, 0))
      y_delta <- y_cat - y_res[,i]

      # Train models to predict the probability delta
      res_deltas[[i]] <- randomForest::randomForest(
        y = y_delta,
        x = x,
        ntree = 500
      )
    }

    out[["m2"]] <- res_deltas
  } else {
    y_delta <- y - predict(rf_source, newdata = x)
    
    res_2 <- randomForest::randomForest(
      y = y_delta,
      x = x,
      ntree = 500
    )
    out[["m2"]] <- res_2
  }


  y_source_hat <- predict(rf_source, newdata = x, type = pred_type)

  res_3 <- viRandomForests::viRandomForests(
    y = y,
    x = cbind(x, y_source_hat = y_source_hat),
    fprob = c(rep(1, num_features), 2),
    ntree = 500,
    keep.forest = TRUE,
    importance = TRUE
  )
  out[["m3"]] <- res_3

  out[["m_source"]] <- rf_source
  return(out)
}

#' Predict data
#'
#' @param trans_rf_res the random forest trained using transfer learning
#' @param newdata new data
#' @param y target data
#' @param x source data
predict_trans_rf = function(
    trans_rf_res,
    newdata,
    y_val,
    x_val,
    x_ensemble = NULL,
    y_ensemble = NULL
){
  if (trans_rf_res[["m_source"]]$type == "classification") {
    pred_type = "prob"
    classes <- levels(predict(trans_rf_res[["m0"]], type = 'response'))
  } else {
    pred_type = "response"
  }

  pred_source = predict(trans_rf_res[['m_source']], newdata = newdata, type = pred_type)
  pred_source_val = predict(trans_rf_res[['m_source']], newdata = x_val, type = pred_type)

  pred_0 = predict(trans_rf_res[['m0']], newdata = newdata, type = pred_type)
  pred_0_val = predict(trans_rf_res[['m0']], newdata = x_val, type = pred_type)

  pred_1 = predict(trans_rf_res[['m1']], newdata = newdata, type = pred_type)
  pred_1_val = predict(trans_rf_res[['m1']], newdata = x_val, type = pred_type)

  if (pred_type == "prob") {
    # Predict on the source model
    pred_2_error <- predict(trans_rf_res[['m_source']], newdata = newdata, type = pred_type)
    pred_2_error_val <- predict(trans_rf_res[['m_source']], newdata = x_val, type = pred_type)

    # Predict error deltas for each class
    num_classes <- length(trans_rf_res[['m_source']]$classes)
    num_samples <- dim(newdata)[1]
    num_samples_val <- dim(x_val)[1]
    
    pred_2_unnorm <- list()
    pred_2_unnorm_val <- list()
    for (i in seq_len(num_classes)) {
      model <- trans_rf_res[['m2']][[i]]
      pred_2_deltas <- predict(model, newdata = newdata)
      pred_2_deltas_val <- predict(model, newdata = x_val)
      
      # Add deltas to original prediction probabilities
      pred_2_unnorm[[i]] <- pred_2_error[,i] + pred_2_deltas
      pred_2_unnorm_val[[i]] <- pred_2_error_val[,i] + pred_2_deltas_val
    }

    # Normalize probabilities to between 0 and 1
    pred_2 <- NULL
    for (i in seq_len(num_samples)) {
      row <- unname(sapply(seq_len(num_classes), \(x) pred_2_unnorm[[x]][i]))
      if (min(row) < 0) {
        row <- row + min(row)
      }
      norm_class_probs <- row/sum(row)
      pred_2 <- c(pred_2, norm_class_probs)
    }
    pred_2 <- matrix(pred_2, ncol=num_classes, byrow=TRUE)

    pred_2_val <- NULL
    for (i in seq_len(num_samples)) {
      row <- unname(sapply(seq_len(num_classes), \(x) pred_2_unnorm_val[[x]][i]))
      if (min(row) < 0) {
        row <- row + min(row)
      }
      norm_class_probs <- row/sum(row)
      pred_2_val <- c(pred_2_val, norm_class_probs)
    }
    pred_2_val <- matrix(pred_2_val, ncol=num_classes, byrow=TRUE)
    #print(pred_2)
  } else {
    pred_2_error = predict(trans_rf_res[['m2']], newdata = newdata)
    pred_2_error_val = predict(trans_rf_res[['m2']], newdata = x_val)
    pred_2 = pred_source + pred_2_error
    pred_2_val = pred_source_val + pred_2_error_val
  }

  newdata = cbind(newdata, y_source_hat = predict(trans_rf_res[['m_source']], newdata = newdata, type = pred_type))
  x_val = cbind(x_val, y_source_hat = predict(trans_rf_res[['m_source']], newdata = x_val, type = pred_type))
  pred_3 = predict(trans_rf_res[["m3"]], newdata = newdata, type = pred_type)
  pred_3_val = predict(trans_rf_res[["m3"]], newdata = x_val, type = pred_type)

  if(pred_type == "prob"){
    pred_0_val <- matrix(pred_0_val, ncol = length(classes))
    pred_1_val <- matrix(pred_1_val, ncol = length(classes))
    pred_2_val <- matrix(pred_2_val, ncol = length(classes))
    pred_3_val <- matrix(pred_3_val, ncol = length(classes))
    pred_0 <- matrix(pred_0, ncol = length(classes))
    pred_1 <- matrix(pred_1, ncol = length(classes))
    pred_2 <- matrix(pred_2, ncol = length(classes))
    pred_3 <- matrix(pred_3, ncol = length(classes))

    pred_combine <- cbind(pred_0, pred_1, pred_2, pred_3)
    pred_combine_val <- cbind(pred_0_val, pred_1_val, pred_2_val, pred_3_val)

    # Create a model to predict the class using the different model results
    m_ensemble <- randomForest::randomForest(
      y = factor(y_val, levels = classes),
      x = pred_combine_val
    )

    # Extract the importance values
    ensemble_importance <- importance(m_ensemble)[,1]
    num_classes <- length(ensemble_importance) / 4
    num_samples <- dim(pred_combine)[1]
    class_model_weights <- data.frame(
      matrix(
        sapply(
          seq_len(num_classes),
          \(x) sapply(
            1:4,
            \(y) ensemble_importance[((x - 1) * 4) + y]
          )
        ),
        ncol=4,
        byrow=TRUE
      )
    )
    colnames(class_model_weights) <- c("m0", "m1", "m2", "m3")

    class_scores_all <- c()
    class_scores_all_val <- c()
    for (sample in seq_len(num_samples)) {
      class_scores <- rep(1, num_classes)
      class_scores_val <- rep(1, num_classes)
      for (class in seq_len(num_classes)) {
        for (model in 1:4) {
          #  print(paste("Class", class, "of model", model, "is", pred_combine[sample, (model - 1) * num_classes + class], "*", class_model_weights[class, model]))
          class_scores[class] <- class_scores[class] * pred_combine[sample, (model - 1) * num_classes + class] ^ class_model_weights[class, model]
          class_scores_val[class] <- class_scores_val[class] * pred_combine_val[sample, (model - 1) * num_classes + class] ^ class_model_weights[class, model]
        }
      }
      class_scores <- class_scores / sum(class_scores)
      class_scores_val <- class_scores_val / sum(class_scores_val)
      class_scores_all <- c(class_scores_all, class_scores)
      class_scores_all_val <- c(class_scores_all_val, class_scores_val)
    }

    pred_ensemble <- matrix(class_scores_all, ncol=num_classes, byrow=TRUE)
    pred_ensemble_val <- matrix(class_scores_all_val, ncol=num_classes, byrow=TRUE)

    #metric0 = mean(ifelse(pred_0_val[,2] > 0.5, classes[2], classes[1]) == y_val)
    #metric1 = mean(ifelse(pred_1_val[,2] > 0.5, classes[2], classes[1]) == y_val)
    #metric2 = mean(ifelse(pred_2_val[,2] > 0.5, classes[2], classes[1]) == y_val)
    #metric3 = mean(ifelse(pred_3_val[,2] > 0.5, classes[2], classes[1]) == y_val)
    #weight_ensemble = c(metric0, metric1, metric2, metric3) / sum(metric0, metric1, metric2, metric3)

    #pred_ensemble = c(cbind(
    #  pred_0[,2],
    #  pred_1[,2],
    #  pred_2[,2],
    #  pred_3[,2]
    #) %*% weight_ensemble)

    pred_source_pred_class <- apply(pred_source, 1, \(x) which.max(x) - 1)
    pred_source_max_prob <- apply(pred_source, 1, max)
    pred_0_pred_class <- apply(pred_0, 1, \(x) which.max(x) - 1)
    pred_0_max_prob <- apply(pred_0, 1, max)
    pred_1_pred_class <- apply(pred_1, 1, \(x) which.max(x) - 1)
    pred_1_max_prob <- apply(pred_1, 1, max)
    pred_2_pred_class <- apply(pred_2, 1, \(x) which.max(x) - 1)
    pred_2_max_prob <- apply(pred_2, 1, max)
    pred_3_pred_class <- apply(pred_3, 1, \(x) which.max(x) - 1)
    pred_3_max_prob <- apply(pred_3, 1, max)
    pred_ensemble_pred_class <- apply(pred_ensemble, 1, \(x) which.max(x) - 1)
    pred_ensemble_max_prob <- apply(pred_ensemble, 1, max)
    
    pred_source_val_pred_class <- apply(pred_source_val, 1, \(x) which.max(x) - 1)
    pred_source_val_max_prob <- apply(pred_source_val, 1, max)
    pred_0_val_pred_class <- apply(pred_0_val, 1, \(x) which.max(x) - 1)
    pred_0_val_max_prob <- apply(pred_0_val, 1, max)
    pred_1_val_pred_class <- apply(pred_1_val, 1, \(x) which.max(x) - 1)
    pred_1_val_max_prob <- apply(pred_1_val, 1, max)
    pred_2_val_pred_class <- apply(pred_2_val, 1, \(x) which.max(x) - 1)
    pred_2_val_max_prob <- apply(pred_2_val, 1, max)
    pred_3_val_pred_class <- apply(pred_3_val, 1, \(x) which.max(x) - 1)
    pred_3_val_max_prob <- apply(pred_3_val, 1, max)
    pred_ensemble_val_pred_class <- apply(pred_ensemble_val, 1, \(x) which.max(x) - 1)
    pred_ensemble_val_max_prob <- apply(pred_ensemble_val, 1, max)
    
    #pred_ensemble_max_prob[pred_ensemble_max_prob == 1.0] <- 0.9999

    out = data.frame(
      truth = y_val,
      pred_source = pred_source_pred_class,
      pred_source_prob = pred_source_max_prob,
      pred_0 = pred_0_pred_class,
      pred_0_prob = pred_0_max_prob,
      pred_1 = pred_1_pred_class,
      pred_1_prob = pred_1_max_prob,
      pred_2 = pred_2_pred_class,
      pred_2_prob = pred_2_max_prob,
      pred_3 = pred_3_pred_class,
      pred_3_prob = pred_3_max_prob,
      pred_ensemble = pred_ensemble_pred_class,
      pred_ensemble_prob = pred_ensemble_max_prob
    )
    # print(out)
  } else {

    # Equal weight, so average all the models
    pred_ensemble <- (pred_0 + pred_1 + pred_2 + pred_3) / 4

    if (!is.null(x_ensemble) && !is.null(y_ensemble)) {

      pred_ens_0_val = predict(trans_rf_res[["m0"]], newdata = x_ensemble, type = pred_type)

      pred_ens_1_val = predict(trans_rf_res[["m1"]], newdata = x_ensemble, type = pred_type)

      pred_ens_source_val = predict(trans_rf_res[['m_source']], newdata = x_ensemble, type = pred_type)
      pred_ens_2_error_val = predict(trans_rf_res[['m2']], newdata = x_ensemble, type = pred_type)
      pred_ens_2_val = pred_ens_source_val + pred_ens_2_error_val

      x_ensemble_val = cbind(x_ensemble, y_source_hat = predict(trans_rf_res[['m_source']], newdata = x_ensemble))
      pred_ens_3_val = predict(trans_rf_res[["m3"]], newdata = x_ensemble_val, type = pred_type)

      meta_model <- randomForest::randomForest(
        y = y_ensemble,
        x = data.frame(
          m0 = pred_ens_0_val,
          m1 = pred_ens_1_val,
          m2 = pred_ens_2_val,
          m3 = pred_ens_3_val
        )
      )

      ensemble_importance <- importance(meta_model)[,1]
      ensemble_importance <- ensemble_importance / sum(ensemble_importance)

      pred_ensemble <- pred_0 * ensemble_importance[1] + 
                       pred_1 * ensemble_importance[2] + 
                       pred_2 * ensemble_importance[3] + 
                       pred_3 * ensemble_importance[4]
    }

    out = data.frame(
      pred_source = pred_source,
      pred_0 = pred_0,
      pred_1 = pred_1,
      pred_2 = pred_2,
      pred_3 = pred_3,
      pred_ensemble = pred_ensemble
    )
  }
  # print(out)
  return(out)
}
