options(repos = BiocManager::repositories())
library(PROcess)
library(pracma)

calculate_quality <- function(spectrum){
  #Calculate noise by move a sliding window with the size of 5
  noise_5 <- local_noise(spectrum, span = 5)
  # noise_3 <- noise(spectrum, span = 3)
  noise_envlp <- 3*PROcess::sigma(noise_5, span = 251)
  
  intensity_f <- approxfun(spectrum[,1],spectrum[,2])
  area0 <- quad(intensity_f,min(spectrum[,1]),max(spectrum[,1]))
  
  subtracted <- spectrum[,2]-noise_envlp
  non_na_ind <- !is.na(subtracted)
  sub_f <- approxfun(spectrum[non_na_ind,1],subtracted[non_na_ind])
  area1 <- quad(sub_f,min(spectrum[non_na_ind,1]),max(spectrum[non_na_ind,1]))
  return(area1/area0)
}

local_noise <- function(spectrum, span){
  n <- nrow(spectrum)
  s <- span%/%2
  intensities <- c(rep(NA,s),spectrum[,2],rep(NA,s))
  local_avgs <-rollapply(intensities, width = span, by = 1, FUN = mean, na.rm = TRUE, align = "left")
  local_noise <- spectrum[,2]-local_avgs
  return(local_noise)
}

decode_params <- function(encoded_params){
  Acc_raw <- if(encoded_params[1]==1) 20000 else if(encoded_params[1]==2) 23000 else 25000
  Grid_Pct_raw <- if(encoded_params[2] == 1) 90 else 95
  Delay_raw <- as.integer(encoded_params[3])*100
  Shots_per_spectrum_raw <- if(encoded_params[4]==1) 300 else 500
  res <- c(Acc_raw, Grid_Pct_raw, Delay_raw, Shots_per_spectrum_raw)
  return(res)
}

encode_params <- function(raw_params){
  Acc <- if(raw_params[1]==20000) 1 else if(raw_params[1]==23000) 2 else 3
  Grid <- if(raw_params[2] == 90) 1 else 2
  Delay <- as.integer(raw_params[3])/100
  Shots_per_spectrum <- if(raw_params[4]==300) 1 else 2
  res <- c(Acc, Grid_Pct, Delay, Shots_per_spectrum)
  return(res)
}

#---------- Search bound for parameters -----------
search_bound <- list(Acc = c(1L,3L), Grid_Pct = c(1L,2L),
                     Delay = c(3L,12L), Shots_per_spectrum = c(1L,2L))


quality_func <- function(Acc, Grid_Pct, Delay, Shots_per_spectrum,met_ind=1){
  # Acc_raw <- if(Acc==1) 20000 else if(Acc==2) 23000 else 25000
  # Grid_Pct_raw <- if(Grid_Pct == 1) 90 else 95
  # Delay_raw <- Delay*100
  # Shots_per_spectrum_raw <- if(Shots_per_spectrum==1) 300 else 500
  match_ind <- row.match(c(Acc_raw, Grid_Pct_raw, Delay_raw, Shots_per_spectrum_raw), init_parameters)
  return(list(Score = init_quality_scores[match_ind], Pred = 0))
}

scale_params <- function(mtx, lower, upper){
  t((t(mtx) - lower) / (upper - lower))
}
