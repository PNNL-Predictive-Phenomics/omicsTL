source(here::here("src", "omicsTL", "r", "requirements.R"))

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

  #need to do something special for categorical
  #This only works for binary classification at the moment
  if (model_type == "classification") {
    y_delta_bool <- y != predict(rf_source, newdata = x)
    y_delta <- factor(y_delta_bool, levels = c(TRUE, FALSE))
    skip <- ifelse(mean(y_delta_bool) %in% c(0, 1), TRUE, FALSE)
  } else {
    y_delta <- y - predict(rf_source, newdata = x)
    skip <- FALSE
  }

  if (!skip) {
    res_2 <- viRandomForests::viRandomForests(
      y = y_delta,
      x = x,
      fprob = var_importance_source,
      ntree = 500,
      keep.forest = TRUE,
      importance = TRUE
    )
    out[["m2"]] <- res_2
  } else {
    #if we are predicting everything correctly using the source model, just use that
    #TODO: exception/warning for if we are predicting everything incorrectly
    out[["m2"]] <- rf_source
  }

  y_source_hat <- predict(rf_source, newdata = x)

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
    x_val
){
  if (trans_rf_res[["m_source"]]$type == "classification") {
    pred_type = "prob"
    classes <- levels(predict(trans_rf_res[["m0"]], type = 'response'))
  } else {
    pred_type = "response"
  }

  pred_0 = predict(trans_rf_res[['m0']], newdata = newdata, type = pred_type)
  pred_0_val = predict(trans_rf_res[['m0']], newdata = x_val, type = pred_type)

  pred_1 = predict(trans_rf_res[['m1']], newdata = newdata, type = pred_type)
  pred_1_val = predict(trans_rf_res[['m1']], newdata = x_val, type = pred_type)

  pred_2_error = predict(trans_rf_res[['m2']], newdata = newdata, type = pred_type)
  pred_2_error_val = predict(trans_rf_res[['m2']], newdata = x_val, type = pred_type)
  if(pred_type == "prob") pred_2 = apply(pred_0, 2, \(v) ifelse(pred_2_error[,2] > 0.5, v, 1 - v))
  if(pred_type != "prob") pred_2 = pred_0 + pred_2_error
  if(pred_type == "prob") pred_2_val = apply(pred_0_val, 2, \(v) ifelse(pred_2_error_val[,2] > 0.5, v, 1 - v))
  if(pred_type != "prob") pred_2_val = pred_0_val + pred_2_error_val

  newdata = cbind(newdata, y_source_hat = predict(trans_rf_res[['m_source']], newdata = newdata))
  x_val = cbind(x_val, y_source_hat = predict(trans_rf_res[['m_source']], newdata = x_val))
  pred_3 = predict(trans_rf_res[["m3"]], newdata = newdata, type = pred_type)
  pred_3_val = predict(trans_rf_res[["m3"]], newdata = x_val, type = pred_type)

  if(pred_type == "prob"){
    # super spaghetti mode
    pred_0_val <- matrix(pred_0_val, ncol = 2)
    pred_1_val <- matrix(pred_1_val, ncol = 2)
    pred_2_val <- matrix(pred_2_val, ncol = 2)
    pred_3_val <- matrix(pred_3_val, ncol = 2)
    pred_0 <- matrix(pred_0, ncol = 2)
    pred_1 <- matrix(pred_1, ncol = 2)
    pred_2 <- matrix(pred_2, ncol = 2)
    pred_3 <- matrix(pred_3, ncol = 2)

    tryCatch({
      meta_model <- randomForest::randomForest(
        y = factor(y_val, levels = classes),
        x = tibble::tibble(
          m0 = pred_0_val[,2],
          m1 = pred_1_val[,2],
          m2 = pred_2_val[,2],
          m3 = pred_3_val[,2]
        )
      )
      pred_ensemble <- predict(
        meta_model,
        newdata = tibble::tibble(
          m0 = pred_0[,2],
          m1 = pred_1[,2],
          m2 = pred_2[,2],
          m3 = pred_3[,2]
        ),
        type = pred_type
      )[,2]

    }, error = function(e) {
      message("An error occurred: ", e$message)
      pred_ensemble <<- NA  # Assign NULL to pred_ensemble in case of failure
    })


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

    out = data.frame(
      pred_0 = pred_0[,2],
      pred_1 = pred_1[,2],
      pred_2 = pred_2[,2],
      pred_3 = pred_3[,2],
      pred_ensemble = pred_ensemble
    )
  } else {

    meta_model <- randomForest::randomForest(
        y = y_val,
        x = tibble::tibble(
          m0 = pred_0_val,
          m1 = pred_1_val,
          m2 = pred_2_val,
          m3 = pred_3_val
        )
      )
    pred_ensemble <- predict(
        meta_model,
        newdata = tibble::tibble(
          m0 = pred_0,
          m1 = pred_1,
          m2 = pred_2,
          m3 = pred_3
        ),
        type = pred_type
      )

    out = data.frame(
      pred_0 = pred_0,
      pred_1 = pred_1,
      pred_2 = pred_2,
      pred_3 = pred_3,
      pred_ensemble = pred_ensemble
    )
  }
  return(out)
}
