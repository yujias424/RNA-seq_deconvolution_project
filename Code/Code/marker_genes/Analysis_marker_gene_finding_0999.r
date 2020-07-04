library(data.table)
library(dtangle)
library(doParallel)

get_marker_list <- function(value){
  typ <- typeof(value[[1]])
  if (typ == "list")
    return(value$L)
  return(value)
}

multi_core_function <- function(samples_no){
  library(data.table)
  library(dtangle)
  input_file <- paste0('~/IndependentStudy/Data/SignatureSimulation/', as.character(samples_no), '_signature.tsv')
  reference_samples_py <-  fread(input_file, data.table = FALSE)

  row.names(reference_samples_py) <- reference_samples_py$`Unnamed: 0`
  pure_samples <- reference_samples_py$`Unnamed: 0`
  reference_samples_py$`Unnamed: 0` <- NULL
  reference_samples_py$V1 <- NULL
  # reference_samples_py <- t(reference_samples_py)

  # reference_samples_py <- log2(reference_samples_py)

  n_markers <- 0.1
  K <- 48
  markers <- find_markers(Y=reference_samples_py, pure_samples = c(1:48), marker_method = 'ratio')

  Y <- reference_samples_py
  n_markers <- sapply(floor(0.999 * lengths(markers$L)), min, ncol(Y)/K)
  # n_markers <- sapply(floor(0.1 * lengths(markers$L)), min, 600/K)
  markers <- get_marker_list(markers)

  if (length(n_markers) == 1) 
    n_markers <- rep(n_markers, K)

  wq_markers <- which(n_markers < 1)

  n_markers[wq_markers] <- floor(n_markers[wq_markers] * lengths(markers)[wq_markers])

  mrkrs <- lapply(1:K, function(i) {
    markers[[i]][1:n_markers[i]]
  })
  names(mrkrs) <- names(pure_samples)

  mrkers_list <- c()
  for(i in mrkrs){
    mrkers_list <- c(mrkers_list, i)
  }

  result_mrkers <- data.frame(mrkers_list)
  result_mrkers$mrkers_list <- result_mrkers$mrkers_list-1
  output_file <- paste0('~/IndependentStudy/Data/dtangle_0999/', as.character(samples_no), '_signature_filter.tsv')
#   print(output_file)
  write.csv(x = result_mrkers, file = output_file)
  # write.csv(x = reference_samples_py, file = output_file)
}

cl <- makeCluster(35)
registerDoParallel(cl)
foreach(samples_no = 0:99) %dopar% multi_core_function(samples_no)