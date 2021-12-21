#############################################
# > sessionInfo()
# R version 4.0.3 (2020-10-10)
# Platform: x86_64-w64-mingw32/x64 (64-bit)
# Running under: Windows 10 x64 (build 19042)
# 
# Matrix products: default
# 
# locale:
#   [1] LC_COLLATE=German_Germany.1252  LC_CTYPE=German_Germany.1252    LC_MONETARY=German_Germany.1252
# [4] LC_NUMERIC=C                    LC_TIME=German_Germany.1252    
# 
# attached base packages:
#   [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# other attached packages:
#   [1] ggplot2_3.3.5       tidyr_1.1.3         xPDPy_0.1-0         DALEXtra_2.0        DALEX_2.3.0         mlr3hyperband_0.1.2
# [7] mlr3tuning_0.7.0    paradox_0.7.1.9000  mlr3learners_0.5.0  mlr3_0.12.0         mlbench_2.1-3      
# 
# loaded via a namespace (and not attached):
#   [1] doParallel_1.0.16    RColorBrewer_1.1-2   bbotk_0.3.1          tools_4.0.3          backports_1.3.0      utf8_1.2.2          
# [7] R6_2.5.1             rpart_4.1-15         Hmisc_4.5-0          rgeos_0.5-5          DBI_1.1.0            mlr3measures_0.4.0  
# [13] colorspace_2.0-2     nnet_7.3-14          withr_2.4.2          mlr3misc_0.9.4       sp_1.4-4             tidyselect_1.1.1    
# [19] gridExtra_2.3        compiler_4.0.3       cli_3.1.0            htmlTable_2.2.1      lgr_0.4.3            labeling_0.4.2      
# [25] scales_1.1.1         checkmate_2.0.0      GISTools_0.7-4       randomForest_4.6-14  rappdirs_0.3.3       palmerpenguins_0.1.0
# [31] stringr_1.4.0        digest_0.6.28        foreign_0.8-80       ingredients_2.2.0    colorRamps_2.3       base64enc_0.1-3     
# [37] jpeg_0.1-9           pkgconfig_2.0.3      htmltools_0.5.2      parallelly_1.28.1    fastmap_1.1.0        htmlwidgets_1.5.4   
# [43] rlang_0.4.12         rstudioapi_0.13      generics_0.1.0       farver_2.1.0         jsonlite_1.7.2       dplyr_1.0.7         
# [49] magrittr_2.0.1       Formula_1.2-4        Matrix_1.2-18        Rcpp_1.0.7           munsell_0.5.0        fansi_0.5.0         
# [55] reticulate_1.18      lifecycle_1.0.1      stringi_1.7.5        yaml_2.2.1           MASS_7.3-53          grid_4.0.3          
# [61] maptools_1.0-2       parallel_4.0.3       listenv_0.8.0        crayon_1.4.2         lattice_0.20-41      splines_4.0.3       
# [67] knitr_1.33           pillar_1.6.4         ranger_0.13.1        uuid_1.0-3           codetools_0.2-16     glue_1.5.0          
# [73] latticeExtra_0.6-29  data.table_1.14.2    remotes_2.2.0        png_0.1-7            vctrs_0.3.8          foreach_1.5.1       
# [79] gtable_0.3.0         purrr_0.3.4          future_1.23.0        assertthat_0.2.1     xfun_0.25            survival_3.2-7      
# [85] tibble_3.1.6         iterators_1.0.13     cluster_2.1.0        globals_0.14.0       ellipsis_0.3.2 
#
#########################################################################################################

# library(mlbench)
# library(mlr3)
# library(mlr3learners)
# library(mlr3tuning)
# library(paradox)
# library(mlr3hyperband)
# library(DALEX)
# library(DALEXtra)
# # remotes::install_github("https://github.com/g-rho/xPDPy")
# library(xPDPy)


library(mlbench)
data("Glass")
str(Glass)
table(Glass$Type)


#########################################################
### 1) Performance Validation and Benchmark of Algorithms

library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(mlr3hyperband)

task <- TaskClassif$new("Glass", Glass, target = "Type")
task

# definer learners
mn  = lrn("classif.multinom", predict_type = "response", predict_sets = c("train", "test"))

# autotune learner for rf
rf = lrn("classif.ranger", predict_sets = c("train", "test"))
resampling = rsmp("cv", folds = 9)
measure = msr("classif.acc")
search_space = ps(
  num.trees = p_dbl(6, 11, trafo = function(x) round(2^x)),
  mtry = p_int(1, 9, tags = "budget")
)
terminator = trm("evals", n_evals = 200)
tuner = tnr("hyperband")#, eta = 2)
atrf = AutoTuner$new(rf, resampling, measure, terminator, tuner, search_space)

# autotune learner for nn
nn = lrn("classif.nnet", predict_sets = c("train", "test"))
resampling = rsmp("cv", folds = 9)
measure = msr("classif.acc")

# http://mlrhyperopt.jakob-r.de/parconfigs #13
search_space = ps(
  decay = p_dbl(-5, 1, trafo = function(x) 10^x),
  size = p_int(1, 20, tags = "budget")
)
terminator = trm("evals", n_evals = 200)
tuner = tnr("hyperband")#, eta = 2)
atnn = AutoTuner$new(nn, resampling, measure, terminator, tuner, search_space)

# benchmark experiment
set.seed(42)
design = benchmark_grid(
  tasks = task,
  learners = list(atnn, atrf, mn),
  resamplings = rsmps("cv", folds = 10)
)

### NOT RUN: ###
# bmr = benchmark(design)
### END NOT RUN ###
load(file = "211215_bmr.Robj")

measures = list(
  msr("classif.acc", predict_sets = "train", id = "acc_train"),
  msr("classif.acc", id = "acc_test")
)

bmr$aggregate(measures)



########################################
### 2) Parameter Tuning (forest and nnet)

### Grid search in order to visualize performance vs. hyperparameters...

# rf
rf_tuned    = lrn("classif.ranger", id = "rf_tuned", predict_type = "response", predict_sets = c("train", "test"))
rf_tuned$param_set

search_space = ps(
  num.trees = p_dbl(6, 11, trafo = function(x) round(2^x)),
  mtry = p_int(1, 9)#, tags = "budget")
)

instance_rf = TuningInstanceSingleCrit$new(
  task = task,
  learner = rf_tuned,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.acc"),
  search_space = search_space,
  terminator = trm("evals", n_evals = 1000)
)

tuner_rf = tnr("grid_search", resolution = 10)
### NOT RUN: ###
# tuner_rf$optimize(instance_rf)
### END NOT RUN ###


####
# nn
nn_tuned    = lrn("classif.nnet", id = "nn_tuned", predict_type = "response") #, predict_sets = c("train", "test")
nn_tuned$param_set

search_space = ps(
  decay = p_dbl(-5, 1, trafo = function(x) 10^x),
  size = p_int(1, 20)#, tags = "budget")
)

instance_nn = TuningInstanceSingleCrit$new(
  task = task,
  learner = nn_tuned,
  resampling = rsmp("cv", folds = 10),
  measure = msr("classif.acc"),
  search_space = search_space,
  terminator = trm("evals", n_evals = 1000)
)

tuner_nn = tnr("grid_search", resolution = 10)
### NOT RUN: ###
# tuner_nn$optimize(instance_nn)
### END NOT RUN ###

load(file = "210603_tuned_params.Robj")

instance_rf$result_y
instance_rf$result_x_search_space
instance_rf$result_x_domain
instance_rf$result_learner_param_vals
as.data.table(instance_rf$archive)

instance_nn$result_y
instance_nn$result_x_search_space
instance_nn$result_x_domain
instance_nn$result_learner_param_vals
as.data.table(instance_nn$archive)



################################
### 3) Final model 
load(file = "210603_tuned_params.Robj")

instance_rf$result_learner_param_vals
rf_final    = lrn("classif.ranger", id = "rf_tuned", predict_type = "prob", predict_sets = c("train", "test"))
rf_final$param_set$values = instance_rf$result_learner_param_vals
rf_final$train(task)

instance_nn$result_learner_param_vals
nn_final    = lrn("classif.nnet", id = "nn_tuned", predict_type = "prob") #, predict_sets = c("train", "test")
nn_final$param_set$values = instance_nn$result_learner_param_vals
nn_final$train(task)


################################
### 4) Partial dependence plots

library(DALEX)
library(DALEXtra)

load(file = "210603_final_tuned_models.Robj")

set.seed(42)
rfx <- explain_mlr3(rf_final,
                    data     = Glass[,-10],
                    y        = Glass$Type,
                    label    = "random forest"
                    )

nnx <- explain_mlr3(nn_final,
                    data     = Glass[,-10],
                    y        = Glass$Type,
                    label    = "neural net"
)

pdp <- model_profile(rfx, variables = names(Glass[,-10]), type = "partial") # $agr_profiles
plot(pdp)

pdp <- model_profile(nnx, variables = names(Glass[,-10]), type = "partial") # $agr_profiles
plot(pdp)




#####################
### 4) Explainability

# remotes::install_github("https://github.com/g-rho/xPDPy")
library(xPDPy)

load(file = "210603_final_tuned_models.Robj")

pfunc = function(model, x){
  res <- as.data.table(model$predict_newdata(x))
  as.data.frame(res)[,cl+3]
  } 


### 1D upsilon
 # xpy(rf_final, Glass, vnames = "RI", parallel = FALSE, pfunction = pfunc)
xpys_1D <- NULL
for(cl in 1:6){
  xpys <- NULL 
  for(vn in names(Glass)[-10]){
    xpys <- c(xpys, xpy(rf_final, Glass, vnames = vn, parallel = FALSE, pfunction = pfunc))
  }
  names(xpys) <- names(Glass)[-10]
  xpys_1D <- cbind(xpys_1D, xpys)
}

colnames(xpys_1D) <- substr(names(as.data.table(rf_final$predict_newdata(Glass)))[4:9],6,6)
xpys_1D


cl = 1
xpy(rf_final, Glass, vnames = "Al", parallel = FALSE, pfunction = pfunc)



### ...upsilon-based variable selection for each class

### NOT RUN: ###
# model     <- rf_final
# x         <- Glass
# target    <- "Type"
# pfunction <- pfunc
# 
# vs_results <- list() 
# for(cl in 1:6){
# #cl        <- 1  
#   n <- 1
#   cat("Step", n, "\n")
#   sel <- NULL
#   trace <- NULL
#   nms <- nms.full <- names(x)[-which(names(x) == target)]
#   xpys <- rep(NA, length(nms))
#   names(xpys) <- nms
#   for (v in nms) xpys[which(names(xpys) == v)] <- xpy(model, x, v, viz = F, parallel = F, pfunction = pfunc)
#   sel <- c(sel, which.max(xpys))
#   trace <- c(trace, max(xpys, na.rm = T))
#   print(xpys)
#   cat("\n", nms.full[sel], max(xpys, na.rm = T), "\n\n")
#   while (length(nms) > 1) {
#     n <- n + 1
#     cat("Step", n, "\n")
#     nms <- nms.full[-sel]
#     xpys <- cbind(xpys, NA)
#     for (v in nms) xpys[which(rownames(xpys) == v), ncol(xpys)] <- xpy(model, 
#                                                                        x, c(names(sel), v), viz = F, parallel = F, pfunction = pfunc)
#     sel <- c(sel, which.max(xpys[, ncol(xpys)]))
#     colnames(xpys) <- c(paste("Step", 1:n))
#     trace <- c(trace, max(xpys, na.rm = T))
#     print(xpys)
#     cat("\n", nms.full[sel], max(xpys[, ncol(xpys)], 
#                                  na.rm = T), "\n\n")
#   }
#   res <- list(selection.order = sel, explainability = trace, 
#               details = xpys)
#   vs_results[[length(vs_results)+1]] <- res
# }
# names(vs_results) <- substr(names(as.data.table(rf_final$predict_newdata(Glass)))[4:9],6,6)
### END NOT RUN: ###

load(file = "210604_vs_results.Robj")
vs_results



#######################################
### multiclass extension of upsilon ###  

# 1) draw PDP for all classes
x <- Glass
vnames <- "RI"
model <- rf_final 

pfunc = function(model, x){
  res <- as.data.table(model$predict_newdata(x))
  as.data.frame(res)[,-(1:3)]  # specific for mlr3 multiclass
} 
pfunction <- pfunc


xs <- x[, names(x) %in% vnames, drop = FALSE]
xs <- data.frame(ID = 1:nrow(xs), xs)
xrest <- x[, !names(x) %in% vnames, drop = FALSE]
xx <- merge(xs, xrest, by.x = NULL, by.y = NULL)
ID <- xx[, 1]

# has to be computed only once for all classes
yhats  <- pfunction(model, xx[, -1])   #, ...)
 #cnames <- colnames(yhats)
pdps   <- aggregate(yhats, list(ID), mean)
 #ID    <- pdps[, 1]
pdps   <- pdps[,-1] 

# for the plot...
ord <- order(xs[[vnames]])

# rng <- range(pdps)
# plot(xs[[vnames]][ord], pdps[ord,1], type = "l", ylim = rng, xlab = vnames, ylab = "PDP")
# for(cl in 2:ncol(yhats)){
#   lines(xs[[vnames]][ord], pdps[ord, cl], type = "l", col = cl)
# }

library(tidyr)
pdps_long        <- data.frame(pdps[ord,], xs[[vnames]][ord])
names(pdps_long) <- c(colnames(pdps), "x") 
pdps_long <- pivot_longer(pdps_long, cols = -x, names_to = "class", values_to = "PDP")

library(ggplot2)
p <- ggplot(pdps_long, aes(x=x, y=PDP, group=class)) +
     geom_line(aes(color=class))+
     geom_point(aes(color=class)) # hier die echten punkte anstatt der PDPs!

p <- p + scale_color_brewer(palette="Paired") + 
     theme_minimal()
p <- p + labs(x = vnames)
p


# 2) compute \upsilon^{MC}

load(file = "210603_final_tuned_models.Robj")
# x <- Glass
# model <- rf_final 
# target <- "Type"

pfunc = function(model, x){
  res <- as.data.table(model$predict_newdata(x))
  as.data.frame(res)[,-(1:3)]  # specific for mlr3 multiclass
} 
pfunction <- pfunc


xpchi <- function(model, x, vnames, pfunction){
  xs <- x[, names(x) %in% vnames, drop = FALSE]
  xs <- data.frame(ID = 1:nrow(xs), xs)
  xrest <- x[, !names(x) %in% vnames, drop = FALSE]
  xx <- merge(xs, xrest, by.x = NULL, by.y = NULL)
  ID <- xx[, 1]
  
  # nur 1x für alle klassen
  yhats  <- pfunction(model, xx[, -1])   #, ...)
  #cnames <- colnames(yhats)
  pdps   <- aggregate(yhats, list(ID), mean)
  pdps   <- pdps[,-1] 
  
  fhats <- pfunction(model, x)
  
  # la place correction in order to prevent posteriors of 0
  fhats_lpc <- fhats * nrow(x) 
  priors <- apply(fhats, 2, mean)
  for(j in 1:length(priors))  fhats_lpc[,j] <- fhats_lpc[,j] + priors[j]
  fhats_lpc <- fhats_lpc / (nrow(x) + 1)
  
  XEs <- apply((pdps - fhats)^2/fhats_lpc, 1, sum) 
  
  # AXE  <- mean(XEs)
  # upsilonM <- 1-pchisq(AXE, df = ncol(fhats)-1)
  # upsilonM
  
  upsilonM <- 1-pchisq(XEs, df = ncol(fhats)-1)
  upsilonM <- mean(upsilonM)
  upsilonM
  
  # XEs <- apply((pdps - fhats)^2/fhats_lpc, 1, sum) 
  # SXE  <- sum(XEs)
  # upsilonM <- 1-pchisq(SXE, df = nrow(fhats)*(ncol(fhats)-1))
  # upsilonM
  }

xpchi(model = rf_final, x = Glass, vnames = c("Mg","Al"), pfunction = pfunc)


### variable selection:

n <- 1
cat("Step", n, "\n")
sel <- NULL
trace <- NULL
nms <- nms.full <- names(x)[-which(names(x) == target)]
xpys <- rep(NA, length(nms))
names(xpys) <- nms
for (v in nms) xpys[which(names(xpys) == v)] <- xpchi(model, x, v, pfunction)
sel <- c(sel, which.max(xpys))
trace <- c(trace, max(xpys, na.rm = T))
print(xpys)
cat("\n", nms.full[sel], max(xpys, na.rm = T), "\n\n")
while (length(nms) > 1) {
  n <- n + 1
  cat("Step", n, "\n")
  nms <- nms.full[-sel]
  xpys <- cbind(xpys, NA)
  for (v in nms) xpys[which(rownames(xpys) == v), ncol(xpys)] <- xpchi(model, x, c(names(sel), v), pfunction)
  sel <- c(sel, which.max(xpys[, ncol(xpys)]))
  colnames(xpys) <- c(paste("Step", 1:n))
  trace <- c(trace, max(xpys, na.rm = T))
  print(xpys)
  cat("\n", nms.full[sel], max(xpys[, ncol(xpys)],
                               na.rm = T), "\n\n")
}

res <- list(selection.order = sel, explainability = trace,
             details = xpys)

names(res$explainability) <- names(res$selection.order)
res$explainability
barplot(res$explainability)


