library(kernlab)
library(foreach)
library(GPfit)
use_virtualenv("ml")
# py_run_string("import GPy")
# source_python("gp_sample.py")

opt_one_round_sequential <- function(DT_bounds, DT_history, acqs){
  params_names <- DT_bounds$Parameter
  # Normalize the params and scores
  Par_Mat <- Min_Max_Scale_Mat(
    as.matrix(DT_history[, eval(params_names)]),
    lower = as.numeric(DT_bounds[, Lower]),
    upper = as.numeric(DT_bounds[, Upper])
  )
  # Get the ids of the unique sets of params
  unique_rounds_id = setdiff(1:nrow(DT_history), which(duplicated(Par_Mat) == TRUE))
  value_vec = DT_history[, length(params_names)+1]
  # Try each acquisition function
  next_params <- data.frame(matrix(nrow = 0, ncol = length(params_names)))
  for(acq in acqs){
    
    if(acq %in% c("ei","ucb","poi")){
      # GP regression using GPfit package
      GP <- GP_fit(Par_Mat[unique_rounds_id,], value_vec[unique_rounds_id], corr = list(type="exponential", power = 2))
      
      # Get next parameter 
      y_max <- max(value_vec)
      next_one <- Utility_Max(DT_bounds, 
                              GP, 
                              acq = acq,
                              y_max ,
                              kappa = 2.576,
                              eps = 0.0)
      print(next_one)
      next_params <- rbind(next_params,    Min_Max_Inverse_Scale_Vec(next_one, lower = DT_bounds[,Lower], upper = DT_bounds[,Upper])
      )
      
    }else if (acq == "TS"){
      print("TS sequential")
    }
  }
  colnames(next_params) <- params_names
  
  # Check if any parameters need to be rounded to integers
  int_vars_ind <- DT_bounds[,Type] == "integer"
  int_vars <- DT_bounds$Parameter[int_vars_ind]
  
  if(length(int_vars) > 0){
    next_params <- next_params %>% mutate_at(int_vars, list(~ round(., 0)))
    
  }
  return(next_params)
}

opt_one_round_batch_multi_acq <- function(DT_bounds,DT_history,q,k,acqs){

  params_names <- DT_bounds$Parameter
  # Normalize the params and scores
  Par_Mat <- Min_Max_Scale_Mat(
    as.matrix(DT_history[, eval(params_names)]),
    lower = as.numeric(DT_bounds[, Lower]),
    upper = as.numeric(DT_bounds[, Upper])
  )
  # Get the ids of the unique sets of params
  unique_rounds_id = setdiff(1:nrow(DT_history), which(duplicated(Par_Mat) == TRUE))
  value_vec = as.numeric(DT_history[, length(params_names)+1])
  # Try each acquisition function
  next_params <- data.frame(matrix(nrow = 0, ncol = length(params_names)))
  for(acq in acqs){
    if(acq %in% c("ei","ucb","poi")){
      # GP regression using kernlab package
      GP <- kernlab::gausspr(x = Par_Mat[unique_rounds_id, ],
                             y = value_vec[unique_rounds_id],
                             type = "regression",
                             scaled = FALSE,
                             kernel="rbfdot",
                             variance.model = TRUE)
      
      # Get next batch 
      y_max <- max(value_vec)
      next_params <- rbind(next_params,Utility_max_q(DT_bounds,
                                                     GP,
                                                     q,
                                                     k, #Euclidean dist
                                                     acq = acq,
                                                     y_max ,
                                                     kappa = 2.576,
                                                     eps = 0.0))
      
    }else if(acq=="TS"){
      next_params <- rbind(next_params,TS_get_batch_new(DT_bounds,Par_Mat, value_vec, q))
    }
  }
  acq_vec <- rep(acqs,each = q)
  next_params <- cbind(next_params,Acquisition = acq_vec)
  return(next_params)
}



opt_one_round_batch <- function(DT_bounds,DT_history,q,k,acq){
  params_names <- DT_bounds$Parameter
  # Normalize the params and scores
  Par_Mat <- Min_Max_Scale_Mat(
    as.matrix(DT_history[, eval(params_names)]),
    lower = as.numeric(DT_bounds[, Lower]),
    upper = as.numeric(DT_bounds[, Upper])
  )
  # Get the ids of the unique sets of params
  unique_rounds_id = setdiff(1:nrow(DT_history), which(duplicated(Par_Mat) == TRUE))
  value_vec = DT_history[, length(params_names)]
  
  if(acq %in% c("ei","ucb","poi")){
    # GP regression using kernlab package
    GP <- kernlab::gausspr(x = Par_Mat[unique_rounds_id, ],
                           y = value_vec[unique_rounds_id],
                           type = "regression",
                           scaled = FALSE,
                           kernel="rbfdot",
                           variance.model = TRUE)
    # Get next batch 
    y_max <- max(value_vec)
    next_params <- Utility_max_q(DT_bounds, 
                                 GP, 
                                 q,
                                 k, #Euclidean dist
                                 acq = acq,
                                 y_max ,
                                 kappa = 2.576,
                                 eps = 0.0)
  }else if(acq=="TS"){
    next_params <- TS_get_batch_new(DT_bounds,Par_Mat, value_vec, q)
  }

  return(next_params)
}

# Sample q points according to the utility
# Adapted from Utility_Max function in RBayesianOptimization package
Utility_max_q <- function(DT_bounds, GP, q=5, k=0.5, acq="ucb", y_max, kappa = 2.576,eps = 0.0){
  # set.seed(33)
  Mat_tries <- Matrix_runif(100,
                            lower = rep(0, length(DT_bounds[, Lower])),
                            upper = rep(1, length(DT_bounds[, Upper])))
  # Negative Utility Function Minimization
  # tic()
  # Mat_optim <- foreach(i = 1:nrow(Mat_tries), .combine = "rbind") %do% {
  Mat_optim <- foreach(i = 1:(q*3), .combine = "rbind") %do% {
    optim_result <- optim(par = Mat_tries[i,],
                          fn = Utility_my,
                          GP = GP, acq = acq, y_max = y_max, kappa = kappa, eps = eps,
                          # method = "L-BFGS-B",
                          # method = "BFGS",
                          # lower = rep(0, length(DT_bounds[, Lower])),
                          # upper = rep(1, length(DT_bounds[, Upper])),
                          control = list(maxit = 100,
                                         factr = 5e11))
    c(optim_result$par, optim_result$value)
  } %>%
    as.data.frame(.) %>%
    setnames(., old = names(.), new = c(DT_bounds[, Parameter], "Negetive_Utility"))
  # toc()
  # Return Best q sets of Parameters
  next_args <- sample_q_points(Mat_optim, ncol(Mat_optim), FALSE, q, as.integer(q/2), DT_bounds, k)
  # argmax <- as.numeric(Mat_optim[which.min(Negetive_Utility), DT_bounds[, Parameter], with = FALSE])
  
  return(next_args)
}

# Sample q optimal points from the sampled points according to the descending of the j-th column, expecting m 
# randomly sampled points
sample_q_points <- function(Mat_optim, j, descending, q, m, DT_bounds, k = 0.5){
  params_names <- DT_bounds$Parameter
  
  # Get no. of optimal and random points to sample
  no_of_optim <- q - m
  #Order the column to be maximized/minimized
  descending_order <- order(Mat_optim[,j], decreasing = descending)
  next_args <- matrix(nrow = 0, ncol = length(Mat_optim)-1)
  next_args <- rbind(next_args,as.numeric(Mat_optim[descending_order[1], params_names]))
  # Sample optimal pointed according to utility
  optim_count <- 1
  curr_order_ind <- 2
  for(curr_order_ind in c(2:nrow(Mat_optim))){
    if(optim_count >= q){
      break
    }else{
      if(enough_distance(Mat_optim, curr_order_ind, TRUE,j, k, params_names)==FALSE){
        # print(curr_order_ind)
        next
      }
      next_args <- rbind(next_args, as.numeric(Mat_optim[descending_order[curr_order_ind], DT_bounds[, Parameter]]))
      # print(curr_order_ind)
      curr_order_ind <- curr_order_ind + 1
      optim_count <- optim_count + 1
    }
    
  }
  
  # Check if any parameters need to be rounded to integers
  int_vars_ind <- DT_bounds[,Type] == "integer"
  int_vars <- DT_bounds$Parameter[int_vars_ind]
  
  # Convert to original values
  next_args_coverted <- t(apply(next_args, 1, FUN = function(x){Min_Max_Inverse_Scale_Vec(x,lower = DT_bounds[, Lower], upper = DT_bounds[, Upper])})) %>%
    as.data.frame() %>%
    magrittr::set_names(params_names)
  
  if(length(int_vars) > 0){
    next_args_coverted <- next_args_coverted %>% mutate_at(int_vars, list(~ round(., 0)))
    
  }
  
  # Check if random samples are needed
  if(nrow(next_args_coverted) < q){
    #Random Sampling
    for(i in c((optim_count+1):q)){
      next_args_coverted <- rbind(next_args_coverted, extract_one_set_of_params(DT_bounds))
    }
  }
  return(next_args_coverted)
}

# Modified utility calculation function using kernlab for GP regression
Utility_my <- function(x_vec, GP, acq = "ucb", y_max, kappa, eps) {
  # GP prediction
  GP_Mean <- kernlab::predict(GP, matrix(x_vec, nrow = 1))
  GP_sd <- kernlab::predict(GP, matrix(x_vec, nrow = 1), type="sdeviation")
  
  
  # Utility Function Type
  if (acq == "ucb") {
    # Utility <- GP_Mean + kappa * sqrt(GP_MSE)
    Utility <- GP_Mean + kappa * GP_sd
  } else if (acq == "ei") {
    # z <- if(sqrt(GP_MSE) ==0) 0 else (GP_Mean - y_max - eps) / sqrt(GP_MSE)
    # z <- if(GP_sd ==0) 0 else ((GP_Mean - y_max - eps) / GP_sd)
    z <- ifelse(GP_sd == 0, 0, (GP_Mean - y_max - eps)/GP_sd)
    # Utility <- (GP_Mean - y_max - eps) * pnorm(z) + sqrt(GP_MSE) * dnorm(z)
    Utility <- (GP_Mean - y_max - eps) * pnorm(z) + GP_sd * dnorm(z)
  } else if (acq == "poi") {
    # z <- (GP_Mean - y_max - eps) / sqrt(GP_MSE)
    z <- (GP_Mean - y_max - eps) / GP_sd
    Utility <- pnorm(z)
  }
  
  return(-Utility)
}


# Check if the distance between the l-th and (l-1)-th points is greater than k
enough_distance <-  function(Mat_optim, l, descending, j, k = 0.5, params_names){
  descending_order <- order(Mat_optim[,j],decreasing = descending)
  if(l >= 2){
    # corr <- cor(as.numeric(Mat_optim[neg_utility_order[j],params_names]),as.numeric(Mat_optim[neg_utility_order[j-1],params_names]))
    dist <- dist(Mat_optim[descending_order[(l-1):l],params_names], method = "euclidean")
    
    return(dist > k)
  }
}

# Extract one random set of parameters from the given search bound

extract_one_set_of_params <- function(DT_bounds) {
  params_names <- DT_bounds$Parameter
  # Create a list containing all legit param values
  params <- c()
  
  for (i in 1:length(params_names)) { # For each variable
    ub <- DT_bounds[i,Upper]
    lb <- DT_bounds[i,Lower]
    
    if(DT_bounds[i,"Type"] == "integer"){
      params <- c(params,sample(as.integer(lb):as.integer(ub),1))
    }else if (DT_bounds[i,"Type"] == "numeric"){
      params <- c(params,runif(1, as.numeric(lb), as.numeric(ub)))
    }
  }
  return(params)
}

# --------------------- Sampling next batch by TS (GPy)--------------------
TS_get_batch_new <- function(DT_bounds, Par_Mat, value_vec, q=5){
  GPy <- import('GPy')
  source_python("gp_sample.py")
  
  no_params <- nrow(DT_bounds)
  next_args <- matrix(ncol = no_params, nrow = 0)
  kernel <- GPy$kern$RBF(input_dim=no_params, variance=1., lengthscale=1.)
  model <- GPy$models$GPRegression(Par_Mat,matrix(value_vec,ncol = 1),kernel)
  
  while (nrow(next_args) < q) {
    #Sample 1500 points
    testX <- Matrix_runif(1500,
                          lower = rep(0, no_params),
                          upper = rep(1, no_params))
    post <- gp_sample_pred(testX, model)
    max_ind <- which.max(as.vector(post))
    next_args <- rbind(next_args, testX[max_ind,])
  }
  
  # Check if any parameters need to be rounded to integers
  int_vars_ind <- DT_bounds[,Type] == "integer"
  int_vars <- DT_bounds$Parameter[int_vars_ind]
  
  # Convert to original values
  next_args_coverted <- t(apply(next_args, 1, FUN = function(x){Min_Max_Inverse_Scale_Vec(x,lower = DT_bounds[, Lower], upper = DT_bounds[, Upper])})) %>%
    as.data.frame() %>%
    magrittr::set_names(DT_bounds$Parameter)
  
  if(length(int_vars) > 0){
    # next_args_coverted <- next_args_coverted %>% mutate_at(int_vars, funs(round(., 0)))
    next_args_coverted <- next_args_coverted %>% mutate_at(int_vars, round)
  }
  
  return(next_args_coverted)
}

#----------- Encode ordinal params
encode_ord_param <- function(column, order){
  for(i in c(1:length(order))){
    column[column == order[i]] <- i
  }
  as.integer(column)
}

# ---------- Decode ordinal param
decode_ord_param <- function(column, order){
  for(i in c(1:length(order))){
    column[column == i] <- order[i]
  }
  column
}
