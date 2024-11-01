# Code that launched machine learning models on three labeling strategies
# FULL_DATA.rds is not available cause it is too large, and can be recreated by folowing the data retrieval described in the original paper

```{r load data and some functions}
library(tidyverse)
library(caret)
source("../utils.R")
DB_drug = readRDS("../DB_drug.rds")

FULL_DATA = readRDS("../Reseau/FULL_DATA.rds")
FULL_DATA_class = as.factor(FULL_DATA$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive)))


FULL_DATA_small = FULL_DATA %>% filter(Drug %in% paste0("Drug_",unlist(DB_drug[-2])))
FULL_DATA_small_class = as.factor(FULL_DATA_small$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive)))


# Create 5 randomly split test sets for both datasets
set.seed(123)
CV_indexes = caret::createDataPartition(y = FULL_DATA_small_class, times = 5, p = 0.80)
CV_indexes = lapply(CV_indexes, function(x) setdiff(1:nbr(FULL_DATA_small_class), x))


TEST_SETS = lapply(CV_indexes, function(x) FULL_DATA_small %>% filter(Drug %in% FULL_DATA_small$Drug[x]))
# saveRDS(TEST_SETS, "TEST_SETS.rds")

Drug_in_test_sets = lapply(TEST_SETS, function(x) x$Drug)

FULL_DATA_sets = lapply(Drug_in_test_sets, function(x) FULL_DATA %>% filter(!Drug %in% x))
FULL_DATA_small_sets = lapply(Drug_in_test_sets, function(x) FULL_DATA_small %>% filter(!Drug %in% x))

# saveRDS(FULL_DATA_sets, "FULL_DATA_sets.rds")
# saveRDS(FULL_DATA_small_sets, "FULL_DATA_small_sets.rds")

TEST_classes = lapply(1:5, function(i) TEST_SETS[[i]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive)))
# saveRDS(TEST_classes, "TEST_classes.rds")


# Save their classes
FULL_DATA_small_classes = lapply(FULL_DATA_small_sets, function(x) x[[1]] %in% paste0("Drug_",unlist(DB_drug$DB_positive)))
# saveRDS(FULL_DATA_small_classes, "FULL_DATA_small_classes.rds")

FULL_DATA_classes = lapply(FULL_DATA_sets, function(x) x[[1]] %in% paste0("Drug_",unlist(DB_drug$DB_positive)))
# saveRDS(FULL_DATA_classes, "FULL_DATA_classes.rds")

```

```{r feature selection function}
select_features = function(df, class, best_n_infgain = 1000, cor_threshold = 0.95, nbr_of_boot = 30, seed = NULL, nCores = 1){
  df_inf_gain = InformationGain_Bootstrap(df = df*10e9, class = as.factor(class), nbr_of_boot = nbr_of_boot, seed = seed, nCores = nCores)
  df_inf_gain = df_inf_gain %>% 
    dplyr::filter(stringr::str_detect(feature, "^SideEffect_|^Category_|^Interaction_|^Disease_", negate = TRUE))

  df = df[, df_inf_gain$feature[1:best_n_infgain]]
  cor_data = abs(cor(df))
  diag(cor_data) = NA
  feature_to_remove = list()
  for(i in 1:ncol(df)){ # For each features
    # Find correlated features and remove them if their information gain is lower (done with unique because the infgain_df is sorted)
    add(feature_to_remove, setdiff(which(cor_data[, df_inf_gain$feature[i]] >= cor_threshold), unique(1:i)))
  }
  df = df[, -unique(unlist(feature_to_remove))]
  cat(paste("Kept the best",ncol(df),"features by Information Gain."))
  return(list(new_df = df, infgain = df_inf_gain))
}

select_hyperparameters = function(df, class, model = c("svm","knn","nb","rf","xgboost"), k = 5, repetition = 5, Ncores = 1){
  results = list()
  if("svm" %in% model){
    parameters_svm = list(cost = round(5^(0:6)), gamma = sort(c(0,signif(2^(0:-15), 3)), decreasing = TRUE), 
                          class.weights = list(c("TRUE" = .5, "FALSE" = .5), c("TRUE" = .6, "FALSE" = .4), c("TRUE" = .2, "FALSE" = .8)), N_of_features = c(50,150,ncol(df)))
    SVM_tuning_res = CV_tuning_SVM(X = df, Y = class, k = k, repetition = repetition, parameters = parameters_svm, randomSearch = 500, Ncores = Ncores)
    SVM_tuning_res_parsed = parse_results(results = SVM_tuning_res, Ncores = Ncores)
    add(results, list(SVM_tuning_res,SVM_tuning_res_parsed))
  }
  if("knn" %in% model){
    parameters_knn = list(kernel = c("gaussian","rectangular","biweight","optimal"),ks = c(2,3,5,7,13,19,31), distance = c(1,1.2,1.5,1.8,2), N_of_features = c(50,150,ncol(df)))
    KNN_tuning_res = CV_tuning_KNN(X = df, Y = class, k = k, repetition = repetition, parameters = parameters_knn, Ncores = 30)
    KNN_tuning_res_parsed = parse_results(results = KNN_tuning_res, Ncores = Ncores)
    add(results, list(KNN_tuning_res, KNN_tuning_res_parsed))
  }
  if("nb" %in% model){
    parameters_bn = list(kernel = c(NA,"gaussian","rectangular","triangular","biweight"), N_of_features = c(50,150,ncol(df)))
    NB_tuning_res =  CV_tuning_NB (X = df, Y = class, k = k, repetition = repetition, parameters = parameters_bn, Ncores = Ncores)
    NB_tuning_res_parsed = parse_results(results = NB_tuning_res, Ncores = Ncores)
    add(results, list(NB_tuning_res,NB_tuning_res_parsed))
  }
  if("rf" %in% model){
    parameters_rf = list(nodesize = c(1,2,4,6,8), N_of_features = c(50,150,ncol(df)))
    RF_tuning_res =  CV_tuning_RF (X = df, Y = class, k = k, repetition = repetition, parameters = parameters_rf, Ncores = Ncores)
    RF_tuning_res_parsed = parse_results(results = RF_tuning_res, Ncores = Ncores)
    add(results, list(RF_tuning_res,RF_tuning_res_parsed))
  }
  if("xgboost" %in% model){
    parameters_xgb = list(max_depth = c(1,2), min_child_weight = c(1,3,7), eta = c(.01,.1,.3,.5), nround = c(60,90,120), N_of_features = c(50,150,ncol(df)))
    XGB_tuning_res = CV_tuning_XGB(X = df, Y = class, k = k, repetition = repetition, parameters = parameters_xgb, randomSearch = 500, Ncores = 1)
    XGB_tuning_res_parsed = parse_results(results = XGB_tuning_res, Ncores = Ncores)
    add(results, list(XGB_tuning_res, XGB_tuning_res_parsed))
  }
  return(results)
}


select_hyperparameters_full = function(df, class, model = c("svm","knn","nb","rf","xgboost"), k = 5, repetition = 5, Ncores = 1){
  results = list()
  if("svm" %in% model){
    parameters_svm = list(cost = round(5^(0:6)), gamma = sort(c(0,signif(2^(0:-15), 3)), decreasing = TRUE), 
                          class.weights = list(c("TRUE" = .5, "FALSE" = .5), c("TRUE" = .6, "FALSE" = .4), c("TRUE" = .2, "FALSE" = .8)), N_of_features = c(50,150,300,ncol(df)))
    SVM_tuning_res = CV_tuning_SVM(X = df, Y = class, k = k, repetition = repetition, parameters = parameters_svm, randomSearch = 100, Ncores = Ncores)
    SVM_tuning_res_parsed = parse_results(results = SVM_tuning_res, Ncores = Ncores)
    add(results, list(SVM_tuning_res,SVM_tuning_res_parsed))
  }
  if("knn" %in% model){
    parameters_knn = list(kernel = c("gaussian","rectangular","biweight","optimal"),ks = c(2,3,5,7,13,19,31), distance = c(1,1.2,1.5,1.8,2), N_of_features = c(50,150,300,ncol(df)))
    KNN_tuning_res = CV_tuning_KNN(X = df, Y = class, k = k, repetition = repetition, parameters = parameters_knn, randomSearch = 100, Ncores = Ncores)
    KNN_tuning_res_parsed = parse_results(results = KNN_tuning_res, Ncores = Ncores)
    add(results, list(KNN_tuning_res, KNN_tuning_res_parsed))
  }
  if("nb" %in% model){
    parameters_bn = list(kernel = c(NA,"gaussian","rectangular","triangular","biweight"), N_of_features = c(50,150,300,ncol(df)))
    NB_tuning_res =  CV_tuning_NB (X = df, Y = class, k = k, repetition = repetition, parameters = parameters_bn, Ncores = Ncores)
    NB_tuning_res_parsed = parse_results(results = NB_tuning_res, Ncores = Ncores)
    add(results, list(NB_tuning_res,NB_tuning_res_parsed))
  }
  if("rf" %in% model){
    parameters_rf = list(nodesize = c(1,2,4,6,8), N_of_features = c(50,150,300,ncol(df)))
    RF_tuning_res =  CV_tuning_RF (X = df, Y = class, k = k, repetition = repetition, parameters = parameters_rf, Ncores = Ncores)
    RF_tuning_res_parsed = parse_results(results = RF_tuning_res, Ncores = Ncores)
    add(results, list(RF_tuning_res,RF_tuning_res_parsed))
  }
  if("xgboost" %in% model){
    parameters_xgb = list(max_depth = c(1,2), min_child_weight = c(1,3,7), eta = c(.01,.1,.3,.5), nround = c(60,90,120), N_of_features = c(50,150,300,ncol(df)))
    XGB_tuning_res = CV_tuning_XGB(X = df, Y = class, k = k, repetition = repetition, parameters = parameters_xgb, randomSearch = 100, Ncores = Ncores)
    XGB_tuning_res_parsed = parse_results(results = XGB_tuning_res, Ncores = Ncores)
    add(results, list(XGB_tuning_res, XGB_tuning_res_parsed))
  }
  return(results)
}


```

```{r sometimes get the FULL information gain }
infgain_list = list()
for(x in Drug_in_test_sets){
  df = FULL_DATA %>% filter(!Drug %in% x)
  print(dim(df))
  df = df[, stringr::str_detect(colnames(df), "^SideEffect_|^Category_|^Interaction_|^Disease_", negate = TRUE)]
  print(dim(df))
  class1 = (df$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive)))
  print(table(class1))
  infgain = InformationGain_Bootstrap(df = df[, -1]*10e9, class = as.factor(class1), nbr_of_boot = 10, seed = 123, nCores = 12)
  print(infgain[1:2, ])
  add(infgain_list, infgain)
}
saveRDS(infgain_list, "infgain_list.rds")


infgain_list = readRDS("infgain_list.rds")

FULL_DATA_sets_selected0 = list()
for(ii in 1:5){
  df = FULL_DATA %>% filter(!Drug %in% Drug_in_test_sets[[ii]])
  print(dim(df))
  df = df[, infgain_list[[ii]]$feature[1:1000]]
  print(dim(df))
  cor_data = abs(cor(df))
  diag(cor_data) = NA
  feature_to_remove = list()
  for(i in 1:ncol(df)){ # For each features
    # Find correlated features and remove them if their information gain is lower (done with unique because the infgain_df is sorted)
    add(feature_to_remove, setdiff(which(cor_data[, infgain_list[[ii]]$feature[i]] >= 0.95), unique(1:i)))
  }
  df = df[, -unique(unlist(feature_to_remove))]
  add(FULL_DATA_sets_selected0, df)
}
saveRDS(FULL_DATA_sets_selected0, "FULL_DATA_sets_selected0.rds")

FULL_DATA_small_sets_selected = lapply(FULL_DATA_small_sets, function(training_set) select_features(df = training_set[, -1]*1e8, class = (training_set$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))), nbr_of_boot = 10, nCores = 10))
saveRDS(FULL_DATA_small_sets_selected, "FULL_DATA_small_sets_selected.rds")


```

```{r feature selection function}
library(tidyverse)
library(caret)
source("../utils.R")
DB_drug = readRDS("../DB_drug.rds")

FULL_DATA_sets_selected = readRDS("FULL_DATA_sets_selected0.rds")
FULL_DATA_small_sets_selected = readRDS("FULL_DATA_small_sets_selected.rds")

FULL_DATA_small_classes = readRDS("FULL_DATA_small_classes.rds")
FULL_DATA_classes = readRDS("FULL_DATA_classes.rds")

FULL_DATA_small_sets_selected_tuning = lapply(1:5, function(i) result_hyper = select_hyperparameters(df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8,
                                                                                                     class = FULL_DATA_small_classes[[i]], 
                                                                                                     k = 5, repetition = 6, Ncores = 30, model = c("svm", "knn","xgboost")))
saveRDS(FULL_DATA_small_sets_selected_tuning, "FULL_DATA_small_sets_selected_tuning.rds")

FULL_DATA_small_sets_selected_tuning_parsed = lapply(FULL_DATA_small_sets_selected_tuning, function(dataset) lapply(dataset, function(model) model[[2]]))
saveRDS(FULL_DATA_small_sets_selected_tuning_parsed, "FULL_DATA_small_sets_selected_tuning_parsed.rds")


FULL_DATA_sets_selected_tuning = lapply(1:5, function(i) result_hyper = select_hyperparameters(df = FULL_DATA_sets_selected[[i]],
                                                                                               class = FULL_DATA_classes[[i]],
                                                                                               k = 5, repetition = 6, Ncores = 33, model = c("svm", "knn","xgboost")))
saveRDS(FULL_DATA_sets_selected_tuning, "FULL_DATA_sets_selected_tuning.rds")



# 132
FULL_DATA_tuned_1 = select_hyperparameters_full(df = FULL_DATA_sets_selected[[1]],class = FULL_DATA_classes[[1]],k = 5, repetition = 6, Ncores = 33, model = c("svm", "knn","xgboost"))
saveRDS(FULL_DATA_tuned_1, "FULL_DATA_tuned_1.rds")

# 142
FULL_DATA_tuned_2 = select_hyperparameters_full(df = FULL_DATA_sets_selected[[2]],class = FULL_DATA_classes[[2]],k = 5, repetition = 6, Ncores = 33, model = c("svm", "knn","xgboost"))
saveRDS(FULL_DATA_tuned_2, "FULL_DATA_tuned_2.rds")

#144
FULL_DATA_tuned_3 = select_hyperparameters_full(df = FULL_DATA_sets_selected[[3]],class = FULL_DATA_classes[[3]],k = 5, repetition = 6, Ncores = 33, model = c("svm", "knn","xgboost"))
saveRDS(FULL_DATA_tuned_3, "FULL_DATA_tuned_3.rds")

# 142
FULL_DATA_tuned_4 = select_hyperparameters_full(df = FULL_DATA_sets_selected[[4]],class = FULL_DATA_classes[[4]],k = 5, repetition = 6, Ncores = 33, model = c("svm", "knn","xgboost"))
saveRDS(FULL_DATA_tuned_4, "FULL_DATA_tuned_4.rds")

#144
FULL_DATA_tuned_5 = select_hyperparameters_full(df = FULL_DATA_sets_selected[[5]],class = FULL_DATA_classes[[5]],k = 5, repetition = 6, Ncores = 60, model = c("svm", "knn","xgboost"))
saveRDS(FULL_DATA_tuned_5, "FULL_DATA_tuned_5.rds")

# Rwmove all models from the object cause they take to much place, in the future, the ML function will not return the models and just the parameters.

FULL_DATA_tuned_1[[1]][[1]] <- lapply(FULL_DATA_tuned_1[[1]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_1[[2]][[1]] <- lapply(FULL_DATA_tuned_1[[2]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_1[[3]][[1]] <- lapply(FULL_DATA_tuned_1[[3]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
saveRDS(FULL_DATA_tuned_1, "FULL_DATA_tuned_1.rds")

FULL_DATA_tuned_2[[1]][[1]] <- lapply(FULL_DATA_tuned_2[[1]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_2[[2]][[1]] <- lapply(FULL_DATA_tuned_2[[2]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_2[[3]][[1]] <- lapply(FULL_DATA_tuned_2[[3]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
saveRDS(FULL_DATA_tuned_2, "FULL_DATA_tuned_2.rds")

FULL_DATA_tuned_3[[1]][[1]] <- lapply(FULL_DATA_tuned_3[[1]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_3[[2]][[1]] <- lapply(FULL_DATA_tuned_3[[2]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_3[[3]][[1]] <- lapply(FULL_DATA_tuned_3[[3]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
saveRDS(FULL_DATA_tuned_3, "FULL_DATA_tuned_3.rds")

FULL_DATA_tuned_4[[1]][[1]] <- lapply(FULL_DATA_tuned_4[[1]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_4[[2]][[1]] <- lapply(FULL_DATA_tuned_4[[2]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_4[[3]][[1]] <- lapply(FULL_DATA_tuned_4[[3]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
saveRDS(FULL_DATA_tuned_4, "FULL_DATA_tuned_4.rds")

FULL_DATA_tuned_5[[1]][[1]] <- lapply(FULL_DATA_tuned_5[[1]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_5[[2]][[1]] <- lapply(FULL_DATA_tuned_5[[2]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
FULL_DATA_tuned_5[[3]][[1]] <- lapply(FULL_DATA_tuned_5[[3]][[1]], function(fold) {
  fold[[1]] = 1
  fold
})
saveRDS(FULL_DATA_tuned_5, "FULL_DATA_tuned_5.rds")


metrics_for_1 = list(FULL_DATA_tuned_1[[1]][[2]],FULL_DATA_tuned_1[[2]][[2]],FULL_DATA_tuned_1[[3]][[2]])
metrics_for_2 = list(FULL_DATA_tuned_2[[1]][[2]],FULL_DATA_tuned_2[[2]][[2]],FULL_DATA_tuned_2[[3]][[2]])
metrics_for_3 = list(FULL_DATA_tuned_3[[1]][[2]],FULL_DATA_tuned_3[[2]][[2]],FULL_DATA_tuned_3[[3]][[2]])
metrics_for_4 = list(FULL_DATA_tuned_4[[1]][[2]],FULL_DATA_tuned_4[[2]][[2]],FULL_DATA_tuned_4[[3]][[2]])
metrics_for_5 = list(FULL_DATA_tuned_5[[1]][[2]],FULL_DATA_tuned_5[[2]][[2]],FULL_DATA_tuned_5[[3]][[2]])

saveRDS(metrics_for_1, "metrics_for_1.rds")
saveRDS(metrics_for_2, "metrics_for_2.rds")
saveRDS(metrics_for_3, "metrics_for_3.rds")
saveRDS(metrics_for_4, "metrics_for_4.rds")
saveRDS(metrics_for_5, "metrics_for_5.rds")
```

```{r}
TEST_SETS = readRDS("TEST_SETS.rds")

metrics_for_1 = readRDS("metrics_for_1.rds")
metrics_for_2 = readRDS("metrics_for_2.rds")
metrics_for_3 = readRDS("metrics_for_3.rds")
metrics_for_4 = readRDS("metrics_for_4.rds")
metrics_for_5 = readRDS("metrics_for_5.rds")

best_params_full1 = list(svm = get_best_param(metrics_for_1[[1]]),
                         knn = get_best_param(metrics_for_1[[2]]),
                         xgboost = get_best_param(metrics_for_1[[3]]))
best_params_full2 = list(svm = get_best_param(metrics_for_2[[1]]),
                         knn = get_best_param(metrics_for_2[[2]]),
                         xgboost = get_best_param(metrics_for_2[[3]]))
best_params_full3 = list(svm = get_best_param(metrics_for_3[[1]]),
                         knn = get_best_param(metrics_for_3[[2]]),
                         xgboost = get_best_param(metrics_for_3[[3]]))
best_params_full4 = list(svm = get_best_param(metrics_for_4[[1]]),
                         knn = get_best_param(metrics_for_4[[2]]),
                         xgboost = get_best_param(metrics_for_4[[3]]))
best_params_full5 = list(svm = get_best_param(metrics_for_5[[1]]),
                         knn = get_best_param(metrics_for_5[[2]]),
                         xgboost = get_best_param(metrics_for_5[[3]]))


test_full1 = train_model(df = FULL_DATA_sets_selected[[1]], class = FULL_DATA_classes[[1]], best_params = best_params_full1, testdata = TEST_SETS[[1]])
test_full2 = train_model(df = FULL_DATA_sets_selected[[2]], class = FULL_DATA_classes[[2]], best_params = best_params_full2, testdata = TEST_SETS[[2]])
test_full3 = train_model(df = FULL_DATA_sets_selected[[3]], class = FULL_DATA_classes[[3]], best_params = best_params_full3, testdata = TEST_SETS[[3]])
test_full4 = train_model(df = FULL_DATA_sets_selected[[4]], class = FULL_DATA_classes[[4]], best_params = best_params_full4, testdata = TEST_SETS[[4]])
test_full5 = train_model(df = FULL_DATA_sets_selected[[5]], class = FULL_DATA_classes[[5]], best_params = best_params_full5, testdata = TEST_SETS[[5]])

FULL_DATA_METRICS = do.call(rbind, list(get_metrics(predicted = test_full1$svm[[2]]    , actuals = TEST_SETS[[1]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full1$knn[[2]]    , actuals = TEST_SETS[[1]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full1$xgboost[[2]], actuals = TEST_SETS[[1]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full2$svm[[2]]    , actuals = TEST_SETS[[2]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full2$knn[[2]]    , actuals = TEST_SETS[[2]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full2$xgboost[[2]], actuals = TEST_SETS[[2]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full3$svm[[2]]    , actuals = TEST_SETS[[3]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full3$knn[[2]]    , actuals = TEST_SETS[[3]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full3$xgboost[[2]], actuals = TEST_SETS[[3]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full4$svm[[2]]    , actuals = TEST_SETS[[4]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full4$knn[[2]]    , actuals = TEST_SETS[[4]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full4$xgboost[[2]], actuals = TEST_SETS[[4]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full5$svm[[2]]    , actuals = TEST_SETS[[5]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full5$knn[[2]]    , actuals = TEST_SETS[[5]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                        get_metrics(predicted = test_full5$xgboost[[2]], actuals = TEST_SETS[[5]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive)))))

FULL_DATA_METRICS = FULL_DATA_METRICS %>% t %>% as.data.frame
colnames(FULL_DATA_METRICS) = c("svm1","knn1","xgboost1","svm2","knn2","xgboost2","svm3","knn3","xgboost3","svm4","knn4","xgboost4","svm5","knn5","xgboost5")
saveRDS(FULL_DATA_METRICS, "FULL_DATA_METRICS.rds")


FULL_DATA_METRICS = readRDS("FULL_DATA_METRICS.rds")
```



```{r construct single models from small dataset}
TEST_SETS = readRDS("TEST_SETS.rds")
FULL_DATA_small_sets_selected_tuning_parsed = readRDS("FULL_DATA_small_sets_selected_tuning_parsed.rds")

# plot_tile(data = FULL_DATA_small_sets_selected_tuning_parsed[[1]][[1]], x = "cost", y = "gamma", facet_var = "N_of_features", title = "1st dataset, svm")
# plot_tile(data = FULL_DATA_small_sets_selected_tuning_parsed[[1]][[2]], x = "ks", y = "kernel", facet_var = "N_of_features", title = "1st dataset, knn")
# plot_tile(data = FULL_DATA_small_sets_selected_tuning_parsed[[1]][[3]], x = "max_depth", y = "min_child_weight", facet_var = "eta", title = "1st dataset, xgboost")

FULL_DATA_small_sets_selected_tuning_parsed[[2]][[1]] %>% select(1:4, MCC, sd_MCC) %>% arrange(desc(MCC))
FULL_DATA_small_sets_selected_tuning_parsed[[2]][[2]] %>% select(1:4, MCC, sd_MCC) %>% arrange(desc(MCC))
FULL_DATA_small_sets_selected_tuning_parsed[[2]][[3]] %>% select(1:5, MCC, sd_MCC) %>% arrange(desc(MCC))

FULL_DATA_small_sets_selected_tuning = lapply(1:5, function(i) result_hyper = select_hyperparameters(df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8,
                                                                                                     class = FULL_DATA_small_classes[[i]], 
                                                                                                     k = 5, repetition = 6, Ncores = 30, model = c("svm", "knn","xgboost")))


best_params1 = list(svm = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[1]][[1]]),
                    knn = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[1]][[2]]),
                    xgboost = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[1]][[3]]))
best_params2 = list(svm = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[2]][[1]]),
                    knn = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[2]][[2]]),
                    xgboost = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[2]][[3]]))
best_params3 = list(svm = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[3]][[1]]),
                    knn = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[3]][[2]]),
                    xgboost = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[3]][[3]]))
best_params4 = list(svm = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[4]][[1]]),
                    knn = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[4]][[2]]),
                    xgboost = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[4]][[3]]))
best_params5 = list(svm = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[5]][[1]]),
                    knn = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[5]][[2]]),
                    xgboost = get_best_param(FULL_DATA_small_sets_selected_tuning_parsed[[5]][[3]]))


test_small1 = train_model(df = FULL_DATA_small_sets_selected[[1]]$new_df/1e8, class = FULL_DATA_small_classes[[1]], best_params = best_params1, testdata = TEST_SETS[[1]])
test_small2 = train_model(df = FULL_DATA_small_sets_selected[[2]]$new_df/1e8, class = FULL_DATA_small_classes[[2]], best_params = best_params2, testdata = TEST_SETS[[2]])
test_small3 = train_model(df = FULL_DATA_small_sets_selected[[3]]$new_df/1e8, class = FULL_DATA_small_classes[[3]], best_params = best_params3, testdata = TEST_SETS[[3]])
test_small4 = train_model(df = FULL_DATA_small_sets_selected[[4]]$new_df/1e8, class = FULL_DATA_small_classes[[4]], best_params = best_params4, testdata = TEST_SETS[[4]])
test_small5 = train_model(df = FULL_DATA_small_sets_selected[[5]]$new_df/1e8, class = FULL_DATA_small_classes[[5]], best_params = best_params5, testdata = TEST_SETS[[5]])


SMALL_DATA_METRICS = do.call(rbind, list(get_metrics(predicted = test_small1$svm[[2]]    , actuals = TEST_SETS[[1]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small1$knn[[2]]    , actuals = TEST_SETS[[1]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small1$xgboost[[2]], actuals = TEST_SETS[[1]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small2$svm[[2]]    , actuals = TEST_SETS[[2]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small2$knn[[2]]    , actuals = TEST_SETS[[2]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small2$xgboost[[2]], actuals = TEST_SETS[[2]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small3$svm[[2]]    , actuals = TEST_SETS[[3]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small3$knn[[2]]    , actuals = TEST_SETS[[3]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small3$xgboost[[2]], actuals = TEST_SETS[[3]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small4$svm[[2]]    , actuals = TEST_SETS[[4]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small4$knn[[2]]    , actuals = TEST_SETS[[4]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small4$xgboost[[2]], actuals = TEST_SETS[[4]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small5$svm[[2]]    , actuals = TEST_SETS[[5]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small5$knn[[2]]    , actuals = TEST_SETS[[5]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive))),
                                         get_metrics(predicted = test_small5$xgboost[[2]], actuals = TEST_SETS[[5]]$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive)))))

SMALL_DATA_METRICS = SMALL_DATA_METRICS %>% t %>% as.data.frame
colnames(SMALL_DATA_METRICS) = c("svm1","knn1","xgboost1","svm2","knn2","xgboost2","svm3","knn3","xgboost3","svm4","knn4","xgboost4","svm5","knn5","xgboost5")

saveRDS(SMALL_DATA_METRICS, "SMALL_DATA_METRICS.rds")

```

# Launch ML models
```{r comparaison with random forest}
library(tidyverse)
library(caret)
source("../utils.R")
TEST_SETS = readRDS("TEST_SETS.rds")
DB_drug = readRDS("../DB_drug.rds")

FULL_DATA_sets_selected = readRDS("FULL_DATA_sets_selected0.rds")
FULL_DATA_small_sets_selected = readRDS("FULL_DATA_small_sets_selected.rds")

FULL_DATA_small_classes = readRDS("FULL_DATA_small_classes.rds")
FULL_DATA_classes = readRDS("FULL_DATA_classes.rds")

TEST_classes = readRDS("TEST_classes.rds")

quick_rf = function(df, class, df_test, class_test){
  # Get training performances
  train_perf = CV_tuning_RF(X = df, Y = class, parameters = list(nodesize = 1, N_of_features = ncol(df)), k = 5, repetition = 2, Ncores = 12)
  
  # Get test performances
  samps = min(table(class))
  ML = randomForest::randomForest(x = df, y = as.factor(class), 
                                  xtest = df_test[, colnames(df)], ytest = as.factor(class_test),
                                  sampsize=c(samps, samps), ntree=300)
  test_pred = as.logical(predict(ML, newdata = df_test[, colnames(df)]))
  
  return(list(train_perf = train_perf, test_y_pred = test_pred, test_y_actual = class_test))
}



rf_small = lapply(1:5, function(i) quick_rf(df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8, class = FULL_DATA_small_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))
rf_full = lapply(1:5, function(i) quick_rf(df = FULL_DATA_sets_selected[[i]], class = FULL_DATA_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))

rf_down = lapply(1:5, function(i) {
  lapply(1:100, function(j) {
    set.seed(j)
    down = caret::downSample(x = FULL_DATA_sets_selected[[i]], y = as.factor(FULL_DATA_classes[[i]]), list = TRUE)
    quick_rf(df = down$x, class = down$y, df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]])
  })
})

saveRDS(rf_down, "rf_down.rds")
saveRDS(rf_small, "rf_small.rds")
saveRDS(rf_full, "rf_full.rds")
```

```{r comparaison with logistic regression}
library(tidyverse)
library(caret)
source("../utils.R")
TEST_SETS = readRDS("TEST_SETS.rds")
TEST_classes = readRDS("TEST_classes.rds")
DB_drug = readRDS("../DB_drug.rds")

FULL_DATA_sets_selected = readRDS("FULL_DATA_sets_selected0.rds")
FULL_DATA_small_sets_selected = readRDS("FULL_DATA_small_sets_selected.rds")

FULL_DATA_small_classes = readRDS("FULL_DATA_small_classes.rds")
FULL_DATA_classes = readRDS("FULL_DATA_classes.rds")


quick_logreg = function(df, class, df_test, class_test){
  library(glmnet)
  # library(doMC)
  # registerDoMC(cores = 14)
  df = as.matrix(df)
  df_test = as.matrix(df_test[, colnames(df)])
  
  df <- scale(df)
  df_test <- scale(df_test, center = attr(df, "scaled:center"), scale = attr(df, "scaled:scale"))

  class_weights <- ifelse(class == 1, sum(class == 0) / length(class),sum(class == 1) / length(class))

  # cv_model <- cv.glmnet(df, as.factor(class), family = "binomial", alpha = 1, nfolds = 10, weights = class_weights, parallel = TRUE)
  cv_model <- cv.glmnet(df, as.factor(class), family = "binomial", alpha = 1, nfolds = 10, weights = class_weights)
  best_lambda <- cv_model$lambda.min
  
  CV_indexes = caret::createMultiFolds(class, k = 10, times = 1)
  
  # Get training accuracy
  train_y_pred = list()
  train_y_actual = list()
  for(fold in CV_indexes){
    x.multifolds = df[fold, ]
    y.multifolds = class[fold]
    x.fold = df[-fold, ]
    y.fold = class[-fold]
    class_weights <- ifelse(y.multifolds == 1, sum(y.multifolds == 0) / length(y.multifolds),sum(y.multifolds == 1) / length(y.multifolds))
    best_model <- glmnet(x.multifolds, as.factor(y.multifolds), family = "binomial", alpha = 1, lambda = best_lambda, weights = class_weights)
    predictions <- predict(best_model, newx = x.fold, type = "response")
    test_y_pred <- ifelse(predictions >= 0.5, TRUE, FALSE)
    
    # Predict on the training set
    add(train_y_pred, test_y_pred)
    add(train_y_actual, y.fold)
  }
  
  class_weights <- ifelse(class == 1, sum(class == 0) / length(class),sum(class == 1) / length(class))
  best_model <- glmnet(df, as.factor(class), family = "binomial", alpha = 1, lambda = best_lambda, weights = class_weights)
  predictions <- predict(best_model, newx = df_test, type = "response")
  test_y_pred <- ifelse(predictions >= 0.5, TRUE, FALSE)
  
  return(list(train_y_pred = train_y_pred, train_y_actual = train_y_actual, test_y_pred = test_y_pred, test_y_actual = class_test))
}


log_small = lapply(1:5, function(i) quick_logreg(df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8, class = FULL_DATA_small_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))
log_full = lapply(1:5, function(i) quick_logreg(df = FULL_DATA_sets_selected[[i]], class = FULL_DATA_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))

log_down = lapply(1:5, function(i) {
  lapply(1:100, function(j) {
    set.seed(j)
    down = caret::downSample(x = FULL_DATA_sets_selected[[i]], y = as.factor(FULL_DATA_classes[[i]]), list = TRUE)
    quick_logreg(df = down$x, class = as.logical(down$y), df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]])
  })
})

saveRDS(log_down, "log_down.rds")
saveRDS(log_small, "log_small.rds")
saveRDS(log_full, "log_full.rds")

log_small = readRDS("log_small.rds")
log_full = readRDS("log_full.rds")
log_down = readRDS("log_down.rds")

log_small_train = sapply(1:5, function(i) sapply(1:10, function(j) mltools::mcc(preds = lapply(log_small[[i]]$train_y_pred, function(x) x[, 1])[[j]], actuals = log_small[[i]]$train_y_actual[[j]])) %>% mean)
log_small_test = sapply(1:5, function(i) mltools::mcc(preds = log_small[[i]]$test_y_pred[, 1], actuals = log_small[[i]]$test_y_actual))

log_full_train = sapply(1:5, function(i) sapply(1:10, function(j) mltools::mcc(preds = lapply(log_full[[i]]$train_y_pred, function(x) x[, 1])[[j]], actuals = log_full[[i]]$train_y_actual[[j]])) %>% mean)
log_full_test = sapply(1:5, function(i) mltools::mcc(preds = log_full[[i]]$test_y_pred[, 1], actuals = log_full[[i]]$test_y_actual))

log_down_train = sapply(1:5, function(g) sapply(1:100, function(i) sapply(1:10, function(j) mltools::mcc(preds = lapply(log_down[[g]][[i]]$train_y_pred, function(x) x[, 1])[[j]], actuals = log_down[[g]][[i]]$train_y_actual[[j]])) %>% mean) %>% mean)
log_down_test = sapply(1:5, function(k) mltools::mcc(preds = rowMeans(sapply(1:100, function(i) log_down[[k]][[i]]$test_y_pred[, 1])) >= 0.5, actuals = log_down[[k]][[1]]$test_y_actual))


log_small_train %>% mean
log_small_test %>% mean
log_full_train %>% mean
log_full_test %>% mean
log_down_train %>% mean
log_down_test %>% mean



```

```{r comparaison with naive bayes}
library(tidyverse)
library(caret)
library(mltools)
source("../utils.R")
TEST_SETS = readRDS("TEST_SETS.rds")
DB_drug = readRDS("../DB_drug.rds")

FULL_DATA_sets_selected = readRDS("FULL_DATA_sets_selected0.rds")
FULL_DATA_small_sets_selected = readRDS("FULL_DATA_small_sets_selected.rds")

FULL_DATA_small_classes = readRDS("FULL_DATA_small_classes.rds")
FULL_DATA_classes = readRDS("FULL_DATA_classes.rds")

TEST_classes = readRDS("TEST_classes.rds")

quick_nb = function(df, class, df_test, class_test){
  # Get training performances
  train_perf = CV_tuning_NB(X = df, Y = class, parameters = list(kernel="gaussian", N_of_features = ncol(df)), k = 5, repetition = 2, Ncores = 12)
  
  # Get test performances
  ML = naivebayes::naive_bayes(x = df, y = class, usekernel = TRUE, kernel="gaussian")
  test_pred = as.logical(predict(ML, newdata = df_test[, colnames(df)]))
  return(list(train_perf = train_perf, test_y_pred = test_pred, test_y_actual = class_test))
}


nb_small = lapply(1:5, function(i) quick_nb(df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8, class = FULL_DATA_small_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[1]]))
nb_full = lapply(1:5, function(i) quick_nb(df = FULL_DATA_sets_selected[[i]], class = FULL_DATA_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[1]]))

saveRDS(nb_small, "nb_small.rds")
saveRDS(nb_full, "nb_full.rds")

nb_down = lapply(1:5, function(i) {
  lapply(1:100, function(j) {
    set.seed(j)
    down = caret::downSample(x = FULL_DATA_sets_selected[[i]], y = as.factor(FULL_DATA_classes[[i]]), list = TRUE)
    quick_nb(df = down$x, class = down$y, df_test = TEST_SETS[[i]], class_test = TEST_classes[[1]])
  })
})

saveRDS(nb_down, "nb_down.rds")


nb_small = readRDS("nb_small.rds")
nb_full = readRDS("nb_full.rds")
nb_down = readRDS("nb_down.rds")

nb_small_train = sapply(1:5, function(i) sapply(nb_small[[i]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean)
nb_full_train = sapply(1:5, function(i) sapply(nb_full[[i]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean)
nb_down_train = sapply(1:5, function(i) sapply(1:100, function(j) sapply(nb_down[[i]][[j]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean) %>% mean)

nb_small_test = sapply(1:5, function(i) mltools::mcc(preds = nb_small[[i]]$test_y_pred, actuals = TEST_classes[[i]]))
nb_full_test = sapply(1:5, function(i) mltools::mcc(preds = nb_full[[i]]$test_y_pred, actuals = TEST_classes[[i]]))
nb_down_test = sapply(1:5, function(i) sapply(1:100, function(j) mltools::mcc(preds = nb_down[[i]][[j]]$test_y_pred, actuals = TEST_classes[[i]])) %>% mean)

nb_full_train %>% mean
nb_full_test %>% mean
nb_small_train %>% mean
nb_small_test %>% mean
nb_down_train %>% mean
nb_down_test %>% mean

```

```{r comparaison with svm }
library(tidyverse)
library(caret)
library(mltools)
source("../utils.R")
TEST_SETS = readRDS("TEST_SETS.rds")
DB_drug = readRDS("../DB_drug.rds")

FULL_DATA_sets_selected = readRDS("FULL_DATA_sets_selected0.rds")
FULL_DATA_small_sets_selected = readRDS("FULL_DATA_small_sets_selected.rds")

FULL_DATA_small_classes = readRDS("FULL_DATA_small_classes.rds")
FULL_DATA_classes = readRDS("FULL_DATA_classes.rds")

TEST_classes = readRDS("TEST_classes.rds")

quick_svm = function(df, class, df_test, class_test){
  # Get training performances
  train_perf = CV_tuning_SVM(X = df, Y = class, parameters = list(cost = c(1000), 
                                                                  gamma = c(0.01,0.001), 
                                                                  class.weights = list(c("TRUE" = .6, "FALSE" = .4), c("TRUE" = .9, "FALSE" = .1)), 
                                                                  N_of_features = c(ncol(df))),
                             k = 5, repetition = 1, Ncores = 6)
  
  # Get test performances
  
  best_params = get_best_param(train_perf %>% parse_results(Ncores = 6))
  
  ML = e1071::svm(x = df, y = as.factor(class), kernel = "radial", 
                  gamma = best_params$gamma,
                  cost = best_params$cost,
                  class.weights = eval(parse(text = best_params$class.weights)))
  
  test_pred = as.logical(predict(ML, newdata = df_test[, colnames(df)]))
  return(list(train_perf = train_perf, test_y_pred = test_pred, test_y_actual = class_test))
}


svm_small = lapply(1:5, function(i) quick_svm(df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8, class = FULL_DATA_small_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))
svm_full = lapply(1:5, function(i) quick_svm(df = FULL_DATA_sets_selected[[i]], class = FULL_DATA_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))

saveRDS(svm_small, "svm_small.rds")
saveRDS(svm_full, "svm_full.rds")

svm_down = lapply(1:5, function(i) {
  lapply(1:100, function(j) {
    set.seed(j)
    down = caret::downSample(x = FULL_DATA_sets_selected[[i]], y = as.factor(FULL_DATA_classes[[i]]), list = TRUE)
    quick_svm(df = down$x, class = down$y, df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]])
  })
})

saveRDS(svm_down, "svm_down.rds")


svm_small = readRDS("svm_small.rds")
svm_full = readRDS("svm_full.rds")
svm_down = readRDS("svm_down.rds")

svm_small_train = sapply(1:5, function(i) sapply(svm_small[[i]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean)
svm_full_train = sapply(1:5, function(i) sapply(svm_full[[i]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean)
svm_down_train = sapply(1:5, function(i) sapply(1:100, function(j) sapply(svm_down[[i]][[j]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean) %>% mean)

svm_small_test = sapply(1:5, function(i) mltools::mcc(preds = svm_small[[i]]$test_y_pred, actuals = TEST_classes[[i]]))
svm_full_test = sapply(1:5, function(i) mltools::mcc(preds = svm_full[[i]]$test_y_pred, actuals = TEST_classes[[i]]))
svm_down_test %>% mean
sapply(1:5, function(i) mltools::mcc(preds = rowMeans(sapply(1:100, function(j) svm_down[[i]][[j]]$test_y_pred)) >= 0.5, actuals = TEST_classes[[i]]))

svm_full_train %>% mean
svm_full_test %>% mean
svm_small_train %>% mean
svm_small_test %>% mean
svm_down_train %>% mean
svm_down_test %>% mean




```

```{r comparaison with ann }
library(tidyverse)
library(caret)
library(mltools)
source("../utils.R")
TEST_SETS = readRDS("TEST_SETS.rds")
DB_drug = readRDS("../DB_drug.rds")

FULL_DATA_sets_selected = readRDS("FULL_DATA_sets_selected0.rds")
FULL_DATA_small_sets_selected = readRDS("FULL_DATA_small_sets_selected.rds")

FULL_DATA_small_classes = readRDS("FULL_DATA_small_classes.rds")
FULL_DATA_classes = readRDS("FULL_DATA_classes.rds")

TEST_classes = readRDS("TEST_classes.rds")

quick_ann = function(df, class, df_test, class_test){
  # Get training performances
  
  train_perf = CV_tuning_ANN(X = df, Y = class, parameters = list(hidden = list(c(100,100)), hidden_dropout_ratios = 0.5, epochs = 100, mini_batch_size = 1, N_of_features = c(ncol(df))),
                             k = 5, repetition = 1, Ncores = 6)
  
  # Get test performances
  
  
  h2o::h2o.init()
  h2o::h2o.no_progress()
  df_test = h2o::as.h2o(df_test[, colnames(df)])
  df = h2o::as.h2o(data.frame(class =  as.factor(class), df))
  
  ML = h2o::h2o.deeplearning(training_frame = df, y = 1,
                             single_node_mode = TRUE, force_load_balance = FALSE, reproducible = TRUE, verbose = FALSE,
                             activation = "RectifierWithDropout", loss = "CrossEntropy", shuffle_training_data = TRUE,
                             input_dropout_ratio=.1,
                             hidden=c(100,100),
                             hidden_dropout_ratios = c(0.5, 0.5),
                             epochs = 100,
                             mini_batch_size = 1)  
  
  test_pred = as.logical(predict(ML, newdata = df_test)$predict)
  return(list(train_perf = train_perf, test_y_pred = test_pred, test_y_actual = class_test))
}


i=3
df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8; class = FULL_DATA_small_classes[[i]]; df_test = TEST_SETS[[i]]; class_test = TEST_classes[[1]]

ann_small = lapply(1:5, function(i) quick_ann(df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8, class = FULL_DATA_small_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))
saveRDS(ann_small, "ann_small.rds")

ann_full = lapply(1:5, function(i) quick_ann(df = FULL_DATA_sets_selected[[i]], class = FULL_DATA_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))
saveRDS(ann_full, "ann_full.rds")


ann_down = lapply(1:5, function(i) {
  lapply(1:100, function(j) {
    set.seed(j)
    down = caret::downSample(x = FULL_DATA_sets_selected[[i]], y = as.factor(FULL_DATA_classes[[i]]), list = TRUE)
    quick_ann(df = down$x, class = down$y, df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]])
  })
})

saveRDS(ann_down, "ann_down.rds")


ann_small = readRDS("ann_small.rds")
ann_full = readRDS("ann_full.rds")
ann_down = readRDS("ann_down.rds")

ann_small_train = sapply(1:5, function(i) sapply(ann_small[[i]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean)
ann_full_train = sapply(1:5, function(i) sapply(ann_full[[i]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean)
ann_down_train = sapply(1:5, function(i) sapply(1:100, function(j) sapply(ann_down[[i]][[j]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean) %>% mean)


ann_small_test = sapply(1:5, function(i) mltools::mcc(preds = ann_small[[i]]$test_y_pred, actuals = ann_small[[i]]$test_y_actual))
ann_full_test = sapply(1:5, function(i) mltools::mcc(preds = ann_full[[i]]$test_y_pred, actuals = ann_full[[i]]$test_y_actual))
ann_down_test = sapply(1:5, function(i) sapply(1:100, function(j) mltools::mcc(preds = ann_down[[i]][[j]]$test_y_pred, actuals = ann_down[[i]][[j]]$test_y_actual)) %>% mean)

ann_full_train %>% mean
ann_full_test %>% mean
ann_small_train %>% mean
ann_small_test %>% mean
ann_down_train %>% mean
ann_down_test %>% mean

```

```{r comparaison with knn}
library(tidyverse)
library(caret)
library(mltools)
source("../utils.R")
TEST_SETS = readRDS("TEST_SETS.rds")
DB_drug = readRDS("../DB_drug.rds")

FULL_DATA_sets_selected = readRDS("FULL_DATA_sets_selected0.rds")
FULL_DATA_small_sets_selected = readRDS("FULL_DATA_small_sets_selected.rds")

FULL_DATA_small_classes = readRDS("FULL_DATA_small_classes.rds")
FULL_DATA_classes = readRDS("FULL_DATA_classes.rds")

TEST_classes = readRDS("TEST_classes.rds")

quick_knn = function(df, class, df_test, class_test){
  df_test = df_test[, colnames(df)]
  # Get training performances
  
  train_perf = CV_tuning_KNN(X = df, Y = class, parameters = list(kernel = c("optimal"), ks = c(5,7), distance = c(1,2), N_of_features = ncol(df)),
                             k = 5, repetition = 1, Ncores = 6)
  
  # Get test performances
  best_params = get_best_param(train_perf %>% parse_results(Ncores = 6))
  df = data.frame(y = as.factor(class), df)
  ML = kknn::train.kknn(y~., df, kernel = "optimal", ks = as.numeric(best_params$ks), distance = as.numeric(best_params$distance))
  
  test_pred = as.logical(predict(ML, newdata = df_test))
  return(list(train_perf = train_perf, test_y_pred = test_pred, test_y_actual = class_test))
}



knn_small = lapply(1:5, function(i) quick_knn(df = FULL_DATA_small_sets_selected[[i]]$new_df/1e8, class = FULL_DATA_small_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))
saveRDS(knn_small, "knn_small.rds")

knn_full = lapply(1:5, function(i) quick_knn(df = FULL_DATA_sets_selected[[i]], class = FULL_DATA_classes[[i]], df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]]))
saveRDS(knn_full, "knn_full.rds")

knn_down = lapply(1:5, function(i) {
  lapply(1:100, function(j) {
    set.seed(j)
    down = caret::downSample(x = FULL_DATA_sets_selected[[i]], y = as.factor(FULL_DATA_classes[[i]]), list = TRUE)
    quick_knn(df = down$x, class = down$y, df_test = TEST_SETS[[i]], class_test = TEST_classes[[i]])
  })
})

saveRDS(knn_down, "knn_down.rds")


knn_small = readRDS("knn_small.rds")
knn_full = readRDS("knn_full.rds")
knn_down = readRDS("knn_down.rds")

knn_small_train = sapply(1:5, function(i) sapply(knn_small[[i]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean)
knn_full_train = sapply(1:5, function(i) sapply(knn_full[[i]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean)
knn_down_train = sapply(1:5, function(i) sapply(1:100, function(j) sapply(knn_down[[i]][[j]]$train_perf, function(fold) mltools::mcc(preds = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])) %>% mean) %>% mean)

knn_small_test = sapply(1:5, function(i) mltools::mcc(preds = knn_small[[i]]$test_y_pred, actuals = knn_small[[i]]$test_y_actual))
knn_full_test = sapply(1:5, function(i) mltools::mcc(preds = knn_full[[i]]$test_y_pred, actuals = knn_full[[i]]$test_y_actual))
knn_down_test = sapply(1:5, function(i) sapply(1:100, function(j) mltools::mcc(preds = knn_down[[i]][[j]]$test_y_pred, actuals = knn_down[[i]][[j]]$test_y_actual)))

knn_small_train %>% mean
knn_small_test %>% mean

knn_full_train %>% mean
knn_full_test %>% mean

knn_down_train %>% mean
knn_down_test %>% mean
```

# Analyze the results
```{r do everything}
TEST_classes = readRDS("TEST_classes.rds")
# Load results
rf_down = readRDS("rf_down.rds")
rf_small = readRDS("rf_small.rds")
rf_full = readRDS("rf_full.rds")
log_small = readRDS("log_small.rds")
log_full = readRDS("log_full.rds")
log_down = readRDS("log_down.rds")
nb_small = readRDS("nb_small.rds")
nb_full = readRDS("nb_full.rds")
nb_down = readRDS("nb_down.rds")
svm_small = readRDS("svm_small.rds")
svm_full = readRDS("svm_full.rds")
svm_down = readRDS("svm_down.rds")
ann_small = readRDS("ann_small.rds")
ann_full = readRDS("ann_full.rds")
ann_down = readRDS("ann_down.rds")
knn_small = readRDS("knn_small.rds")
knn_full = readRDS("knn_full.rds")
knn_down = readRDS("knn_down.rds")

# Get performances for the test set
rf_small_test_metric = sapply(1:5, function(i) get_metrics(predicted = rf_small[[i]]$test_y_pred, actuals = rf_small[[i]]$test_y_actual)) #5
rf_full_test_metric = sapply(1:5, function(i) get_metrics(predicted = rf_full[[i]]$test_y_pred, actuals = rf_full[[i]]$test_y_actual)) #5
rf_down_test_metric = sapply(1:5, function(i) get_metrics(predicted = rowMeans(sapply(1:100, function(j) rf_down[[i]][[j]]$test_y_pred)) >= 0.5, actuals = TEST_classes[[i]])) #500

log_small_test_metric = sapply(1:5, function(i) get_metrics(predicted = log_small[[i]]$test_y_pred[, 1], actuals = log_small[[i]]$test_y_actual)) #5
log_full_test_metric = sapply(1:5, function(i) get_metrics(predicted = log_full[[i]]$test_y_pred[, 1], actuals = log_full[[i]]$test_y_actual)) #5
log_down_test_metric = sapply(1:5, function(i) get_metrics(predicted = rowMeans(sapply(1:100, function(j) log_down[[i]][[j]]$test_y_pred[, 1]), na.rm = TRUE) >= 0.5, actuals = TEST_classes[[i]])) #500

nb_small_test_metric = sapply(1:5, function(i) get_metrics(predicted = nb_small[[i]]$test_y_pred, actuals = TEST_classes[[i]])) #5
nb_full_test_metric = sapply(1:5, function(i) get_metrics(predicted = nb_full[[i]]$test_y_pred, actuals = TEST_classes[[i]])) #5
nb_down_test_metric = sapply(1:5, function(i) get_metrics(predicted = rowMeans(sapply(1:100, function(j) nb_down[[i]][[j]]$test_y_pred), na.rm = TRUE) >= 0.5, actuals = TEST_classes[[i]])) #500

svm_small_test_metric = sapply(1:5, function(i) get_metrics(predicted = svm_small[[i]]$test_y_pred, actuals = TEST_classes[[i]])) #5
svm_full_test_metric = sapply(1:5, function(i) get_metrics(predicted = svm_full[[i]]$test_y_pred, actuals = TEST_classes[[i]])) #5
svm_down_test_metric = sapply(1:5, function(i) get_metrics(predicted = rowMeans(sapply(1:100, function(j) svm_down[[i]][[j]]$test_y_pred), na.rm = TRUE) >= 0.5, actuals = TEST_classes[[i]])) #500

ann_small_test_metric = sapply(1:5, function(i) get_metrics(predicted = ann_small[[i]]$test_y_pred, actuals = ann_small[[i]]$test_y_actual)) #5
ann_full_test_metric = sapply(1:5, function(i) get_metrics(predicted = ann_full[[i]]$test_y_pred, actuals = ann_full[[i]]$test_y_actual)) #5
ann_down_test_metric = sapply(1:5, function(i) get_metrics(predicted = rowMeans(sapply(1:100, function(j) ann_down[[i]][[j]]$test_y_pred), na.rm = TRUE) >= 0.5, actuals = TEST_classes[[i]])) #500

knn_small_test_metric = sapply(1:5, function(i) get_metrics(predicted = knn_small[[i]]$test_y_pred, actuals = knn_small[[i]]$test_y_actual)) #5
knn_full_test_metric = sapply(1:5, function(i) get_metrics(predicted = knn_full[[i]]$test_y_pred, actuals = knn_full[[i]]$test_y_actual)) #5
knn_down_test_metric = sapply(1:5, function(i) get_metrics(predicted = rowMeans(sapply(1:100, function(j) knn_down[[i]][[j]]$test_y_pred), na.rm = TRUE) >= 0.5, actuals = TEST_classes[[i]])) #500


# Get performances for the training set
rf_small_train_metric = sapply(1:5, function(i) rowMeans(sapply(rf_small[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 50
rf_full_train_metric = sapply(1:5, function(i) rowMeans(sapply(rf_full[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 50
rf_down_train_metric = sapply(1:5, function(i) rowMeans(sapply(1:100, function(j) rowMeans(sapply(rf_down[[i]][[j]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)), na.rm = TRUE)) # 5000

log_small_train_metric = sapply(1:5, function(i) rowMeans(sapply(1:10, function(j) get_metrics(predicted = log_small[[i]]$train_y_pred[[j]][, 1], actuals = log_small[[i]]$train_y_actual[[j]])), na.rm = TRUE)) # 50
log_full_train_metric = sapply(1:5, function(i) rowMeans(sapply(1:10, function(j) get_metrics(predicted = log_full[[i]]$train_y_pred[[j]][, 1], actuals = log_full[[i]]$train_y_actual[[j]])), na.rm = TRUE)) # 50
log_down_train_metric = sapply(1:5, function(i) rowMeans(sapply(1:100, function(j) rowMeans(sapply(1:10, function(k) get_metrics(predicted = log_down[[i]][[j]]$train_y_pred[[k]][, 1], actuals = log_down[[i]][[j]]$train_y_actual[[k]])), na.rm = TRUE)), na.rm = TRUE)) # 5000

nb_small_train_metric = sapply(1:5, function(i) rowMeans(sapply(nb_small[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 50
nb_full_train_metric = sapply(1:5, function(i) rowMeans(sapply(nb_full[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 50
nb_down_train_metric = sapply(1:5, function(i) rowMeans(sapply(1:100, function(j) rowMeans(sapply(nb_down[[i]][[j]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)), na.rm = TRUE)) # 5000

svm_small_train_metric = sapply(1:5, function(i) rowMeans(sapply(svm_small[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 100
svm_full_train_metric = sapply(1:5, function(i) rowMeans(sapply(svm_full[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 100
svm_down_train_metric = sapply(1:5, function(i) rowMeans(sapply(1:100, function(j) rowMeans(sapply(svm_down[[i]][[j]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)), na.rm = TRUE)) # 10 000

ann_small_train_metric = sapply(1:5, function(i) rowMeans(sapply(ann_small[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 25
ann_full_train_metric = sapply(1:5, function(i) rowMeans(sapply(ann_full[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 25
ann_down_train_metric = sapply(1:5, function(i) rowMeans(sapply(1:100, function(j) rowMeans(sapply(ann_down[[i]][[j]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)), na.rm = TRUE)) # 1750

knn_small_train_metric = sapply(1:5, function(i) rowMeans(sapply(knn_small[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 100
knn_full_train_metric = sapply(1:5, function(i) rowMeans(sapply(knn_full[[i]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)) # 100
knn_down_train_metric = sapply(1:5, function(i) rowMeans(sapply(1:100, function(j) rowMeans(sapply(knn_down[[i]][[j]]$train_perf, function(fold) get_metrics(predicted = fold$cv_preds_list[[1]], actuals = fold$cv_actuals_list[[1]])), na.rm = TRUE)), na.rm = TRUE)) # 10 000


save(rf_small_test_metric,
rf_full_test_metric,
rf_down_test_metric,
log_small_test_metric,
log_full_test_metric,
log_down_test_metric,
nb_small_test_metric,
nb_full_test_metric,
nb_down_test_metric,
svm_small_test_metric,
svm_full_test_metric,
svm_down_test_metric,
ann_small_test_metric,
ann_full_test_metric,
ann_down_test_metric, 
knn_small_test_metric,
knn_full_test_metric,
knn_down_test_metric, file = "test_metric.rdata")

save(rf_small_train_metric,
rf_full_train_metric,
rf_down_train_metric,
log_small_train_metric,
log_full_train_metric,
log_down_train_metric,
nb_small_train_metric,
nb_full_train_metric,
nb_down_train_metric,
svm_small_train_metric,
svm_full_train_metric,
svm_down_train_metric,
ann_small_train_metric,
ann_full_train_metric,
ann_down_train_metric, 
knn_small_train_metric,
knn_full_train_metric,
knn_down_train_metric, file = "train_metric.rdata")

```

```{r}
load("train_metric.rdata")
load("test_metric.rdata")
rf_small_test_metric[6, ] %>% mean
rf_full_test_metric[6, ] %>% mean
rf_down_test_metric[6, ] %>% mean
log_small_test_metric[6, ] %>% mean
log_full_test_metric[6, ] %>% mean
log_down_test_metric[6, ] %>% mean
nb_small_test_metric[6, ] %>% mean
nb_full_test_metric[6, ] %>% mean
nb_down_test_metric[6, ] %>% mean
svm_small_test_metric[6, ] %>% mean
svm_full_test_metric[6, ] %>% mean
svm_down_test_metric[6, ] %>% mean
ann_small_test_metric[6, ] %>% mean
ann_full_test_metric[6, ] %>% mean
ann_down_test_metric[6, ] %>% mean
knn_small_test_metric[6, ] %>% mean
knn_full_test_metric[6, ] %>% mean
knn_down_test_metric[6, ] %>% mean

rf_small_train_metric[6, ] %>% mean
rf_full_train_metric[6, ] %>% mean
rf_down_train_metric[6, ] %>% mean
log_small_train_metric[6, ] %>% mean
log_full_train_metric[6, ] %>% mean
log_down_train_metric[6, ] %>% mean
nb_small_train_metric[6, ] %>% mean
nb_full_train_metric[6, ] %>% mean
nb_down_train_metric[6, ] %>% mean
svm_small_train_metric[6, ] %>% mean
svm_full_train_metric[6, ] %>% mean
svm_down_train_metric[6, ] %>% mean
ann_small_train_metric[6, ] %>% mean
ann_full_train_metric[6, ] %>% mean
ann_down_train_metric[6, ] %>% mean
knn_small_train_metric[6, ] %>% mean
knn_full_train_metric[6, ] %>% mean
knn_down_train_metric[6, ] %>% mean


```

```{r do everything}
Result_table_test_set = list(rf_small_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             rf_full_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             rf_down_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             log_small_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             log_full_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             log_down_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             nb_small_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             nb_full_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             nb_down_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             svm_small_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             svm_full_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             svm_down_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             ann_small_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             ann_full_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             ann_down_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             knn_small_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             knn_full_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")")),
                             knn_down_test_metric %>% apply(1, function(x) paste(c(mean(x), sd(x)) %>% round(2), collapse = " (") %>% paste0(")"))) %>% do.call(cbind, .)




write_csv(rownames_to_column(as.data.frame(Result_table_test_set)), "Result_table_test_set.csv")

Result_table_test_set = list(rf_small_test_metric %>% apply(1, function(x) mean(x)),
                             rf_full_test_metric %>% apply(1, function(x) mean(x)),
                             rf_down_test_metric %>% apply(1, function(x) mean(x)),
                             log_small_test_metric %>% apply(1, function(x) mean(x)),
                             log_full_test_metric %>% apply(1, function(x) mean(x)),
                             log_down_test_metric %>% apply(1, function(x) mean(x)),
                             nb_small_test_metric %>% apply(1, function(x) mean(x)),
                             nb_full_test_metric %>% apply(1, function(x) mean(x)),
                             nb_down_test_metric %>% apply(1, function(x) mean(x)),
                             svm_small_test_metric %>% apply(1, function(x) mean(x)),
                             svm_full_test_metric %>% apply(1, function(x) mean(x)),
                             svm_down_test_metric %>% apply(1, function(x) mean(x)),
                             ann_small_test_metric %>% apply(1, function(x) mean(x)),
                             ann_full_test_metric %>% apply(1, function(x) mean(x)),
                             ann_down_test_metric %>% apply(1, function(x) mean(x)),
                             knn_small_test_metric %>% apply(1, function(x) mean(x)),
                             knn_full_test_metric %>% apply(1, function(x) mean(x)),
                             knn_down_test_metric %>% apply(1, function(x) mean(x))) %>% do.call(cbind, .)


Result_table_test_set[6, seq(1,18,3)] %>% mean
Result_table_test_set[6, seq(2,18,3)] %>% mean
Result_table_test_set[6, seq(3,18,3)] %>% mean


```

```{r plot results}
MCC = c(
rf_small_test_metric[6, ],
rf_full_test_metric[6, ],
rf_down_test_metric[6, ],
log_small_test_metric[6, ],
log_full_test_metric[6, ],
log_down_test_metric[6, ],
nb_small_test_metric[6, ],
nb_full_test_metric[6, ],
nb_down_test_metric[6, ],
svm_small_test_metric[6, ],
svm_full_test_metric[6, ],
svm_down_test_metric[6, ],
ann_small_test_metric[6, ],
ann_full_test_metric[6, ],
ann_down_test_metric[6, ],
knn_small_test_metric[6, ],
knn_full_test_metric[6, ],
knn_down_test_metric[6, ],

rf_small_train_metric[6, ],
rf_full_train_metric[6, ],
rf_down_train_metric[6, ],
log_small_train_metric[6, ],
log_full_train_metric[6, ],
log_down_train_metric[6, ],
nb_small_train_metric[6, ],
nb_full_train_metric[6, ],
nb_down_train_metric[6, ],
svm_small_train_metric[6, ],
svm_full_train_metric[6, ],
svm_down_train_metric[6, ],
ann_small_train_metric[6, ],
ann_full_train_metric[6, ],
ann_down_train_metric[6, ],
knn_small_train_metric[6, ],
knn_full_train_metric[6, ],
knn_down_train_metric[6, ])

data %>% filter(Dataset == "Test (x5)") %>% group_by(Labelling, Algorithm) %>% summarise(MCC = mean(MCC)) %>% group_by(Algorithm) %>% 
  summarise(avg_MCC_GPT4 = MCC[Labelling == "GPT-4"], max_MCC_other = max(MCC[Labelling != "GPT-4"]), avg_MCC_other = mean(MCC[Labelling != "GPT-4"]),diff_min = avg_MCC_GPT4 - max_MCC_other, diff_avg = avg_MCC_GPT4 - avg_MCC_other) %>% select(Algorithm, diff_min, diff_avg) %>% {rbind(., c("Average", colMeans(.[, -1])))} %>% 
  mutate(diff_min = round(as.numeric(diff_min), 3),diff_avg = round(as.numeric(diff_avg), 3))

data = data.frame(MCC = MCC,
                  Algorithm = rep(c("RF","LogL1","NB","SVM","ANN","kNN"), each = 15, times = 2),
                  Labelling = rep(c("GPT-4","No sampling","Under sampling"), each = 5, times = 12),
                  Dataset = factor(rep(c("Test (x5)","Training (x5)"), each = 90), levels = c("Training (x5)", "Test (x5)")))


ggplot(data, aes(x = Algorithm, y = MCC, color = Labelling)) +
  stat_summary(
    fun = mean,
    geom = "point",
    size = 4,
    position = position_dodge(width = 0.75)
  ) +
  stat_summary(
    fun.data = function(x) {
      mean_x <- mean(x, na.rm = TRUE)
      sd_x <- sd(x, na.rm = TRUE)
      ymin <- mean_x - sd_x
      ymax <- mean_x + sd_x
      return(c(y = mean_x, ymin = ymin, ymax = ymax))
    },
    geom = "errorbar",
    width = 0.2,
    position = position_dodge(width = 0.75)
  ) +
  facet_wrap(~Dataset, scales = "fixed", nrow = 1) + # Ensures same y-axis scale for both facets
  coord_cartesian(ylim = c(0, 1.1)) + # Adjust y-axis range to display values outside [0, 1]
  theme_minimal() +
  labs(
    title = "",
    x = "Algorithm",
    y = "Matthews Correlation Coefficient (MCC)",
    fill = "Labelling",
    color = "Labelling"
  ) +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    strip.text = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.spacing = unit(1, "lines")
  )


data %>% filter(Algorithm == "LogL1" & Labelling == "GPT-4")

```

```{r plot results figure article}
MCC = c(
rf_small_test_metric[6, ],
rf_full_test_metric[6, ],
rf_down_test_metric[6, ],
log_small_test_metric[6, ],
log_full_test_metric[6, ],
log_down_test_metric[6, ],
nb_small_test_metric[6, ],
nb_full_test_metric[6, ],
nb_down_test_metric[6, ],
svm_small_test_metric[6, ],
svm_full_test_metric[6, ],
svm_down_test_metric[6, ],
ann_small_test_metric[6, ],
ann_full_test_metric[6, ],
ann_down_test_metric[6, ],
knn_small_test_metric[6, ],
knn_full_test_metric[6, ],
knn_down_test_metric[6, ])

Bal_acc = c(
rf_small_train_metric[5, ],
rf_full_train_metric[5, ],
rf_down_train_metric[5, ],
log_small_train_metric[5, ],
log_full_train_metric[5, ],
log_down_train_metric[5, ],
nb_small_train_metric[5, ],
nb_full_train_metric[5, ],
nb_down_train_metric[5, ],
svm_small_train_metric[5, ],
svm_full_train_metric[5, ],
svm_down_train_metric[5, ],
ann_small_train_metric[5, ],
ann_full_train_metric[5, ],
ann_down_train_metric[5, ],
knn_small_train_metric[5, ],
knn_full_train_metric[5, ],
knn_down_train_metric[5, ])

data %>% filter(Dataset == "Test (x5)") %>% group_by(Labelling, Algorithm) %>% summarise(MCC = mean(MCC)) %>% 
  group_by(Algorithm) %>% 
  summarise(avg_MCC_GPT4 = MCC[Labelling == "GPT-4"], 
            max_MCC_other = max(MCC[Labelling != "GPT-4"]), 
            avg_MCC_other = mean(MCC[Labelling != "GPT-4"]),
            diff_min = avg_MCC_GPT4 - max_MCC_other, 
            diff_avg = avg_MCC_GPT4 - avg_MCC_other) %>% 
  select(Algorithm, diff_min, diff_avg) %>% {rbind(., c("Average", colMeans(.[, -1])))} %>% 
  mutate(diff_min = round(as.numeric(diff_min), 3),diff_avg = round(as.numeric(diff_avg), 3))

data = data.frame(MCC = c(MCC, Bal_acc),
                  Algorithm = rep(c("RF","LogL1","NB","SVM","ANN","kNN"), each = 15, times = 2),
                  Labelling = rep(c("GPT-4","No sampling","Under sampling"), each = 5, times = 12),
                  Dataset = factor(rep(c("Test (x5)","Training (x5)"), each = 90), levels = c("Training (x5)", "Test (x5)")),
                  Metric = factor(rep(c("MCC","Ballanced Accuracy"), each = 90), levels = c("Ballanced Accuracy", "MCC")))


# Create a new variable that combines Dataset and Metric for labelling

data$Facet_Label <- factor(ifelse(data$Dataset == "Training (x5)", "Training\nBalanced Accuracy", "Test\nMCC"), levels = c("Training\nBalanced Accuracy", "Test\nMCC"))

data$Facet_Label <- factor(ifelse(data$Dataset == "Training (x5)", expression(paste0("Training\n",scriptstyle("Balanced Accuracy"))), expression(paste0("Test\n",scriptstyle("MCC")))), 
                          levels = c(expression(paste0("Training\n",scriptstyle("Balanced Accuracy"))), expression(paste0("Test\n",scriptstyle("MCC")))))

data$Facet_Label <- factor(ifelse(data$Dataset == "Training (x5)", "**Training**", "**Test**"), levels = c("**Training**", "**Test**"))


data$Facet_Label <- factor(ifelse(data$Dataset == "Training (x5)", 
                                  "Training\n<b>Balanced Accuracy</b>", 
                                  "Test\n<b>MCC</b>"), 
                          levels = c("Training\n<b>Balanced Accuracy</b>", 
                                     "Test\n<b>MCC</b>"))

# Now plot with the custom facet labels
ggplot(data %>% filter((Metric == "Ballanced Accuracy" & Dataset == "Training (x5)") | 
                       (Metric == "MCC" & Dataset == "Test (x5)")),
       aes(x = Algorithm, y = MCC, color = Labelling)) +
  stat_summary(
    fun = mean,
    geom = "point",
    size = 4,
    position = position_dodge(width = 0.75)
  ) +
  geom_richtext() +
  stat_summary(
    fun.data = function(x) {
      mean_x <- mean(x, na.rm = TRUE)
      sd_x <- sd(x, na.rm = TRUE)
      ymin <- mean_x - sd_x
      ymax <- mean_x + sd_x
      return(c(y = mean_x, ymin = ymin, ymax = ymax))
    },
    geom = "errorbar",
    width = 0.2,
    position = position_dodge(width = 0.75)
  ) +
  facet_wrap(~Facet_Label, 
             scales = "free_y", 
             nrow = 1) +
  coord_cartesian(ylim = c(0, 1.1)) +
  theme_minimal() +
  labs(
    title = ,
    x = "Algorithm",
    y = "Metric Value",
    fill = "Labelling",
    color = "Labelling"
  ) +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    strip.text = element_text(size = 14, face = "bold", hjust = 0.5), # Center the facet labels
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.spacing = unit(1, "lines")
  )



```

```{r}
data %>% filter(Algorithm == "ANN" & Labelling == "GPT-4" & Dataset == "Test (x5)") %>% pull(MCC)

wilcox.test(data %>% filter(Algorithm == "LogL1" & Labelling == "GPT-4" & Dataset == "Test (x5)") %>% pull(MCC), 
            data %>% filter(Algorithm == "LogL1" & Labelling == "No sampling" & Dataset == "Test (x5)") %>% pull(MCC), 
            exact = FALSE)
```


```{r construct single models from small dataset}
SMALL_DATA_METRICS = readRDS("SMALL_DATA_METRICS.rds")
FULL_DATA_METRICS = readRDS("FULL_DATA_METRICS.rds")

data <- data.frame(MCC = c(unlist(SMALL_DATA_METRICS["MCC", ]), unlist(FULL_DATA_METRICS["MCC", ])),
                   Model = rep(c("SVM", "KNN", "XGBoost"), times = 10),
                   Labelling = rep(c("GPT-4", "Sampling"), each = 15))

# Plot the original points
plot <- data %>% 
  ggplot(aes(y = MCC, x = Model, color = Labelling)) +
  geom_point()

# Calculate means and add larger points
mean_data <- data %>%
  group_by(Model, Labelling) %>%
  summarize(mean_MCC = mean(MCC), .groups = 'drop')

plot + 
  geom_point(data = mean_data, aes(y = mean_MCC, x = Model, color = Labelling), size = 5, shape = 18)+
  labs(title = "MCC of each model for 5 different test sets (with average)")

```




```{r}
repodb = read_csv("../writting/full.csv")
repodb_PC = repodb %>% filter(tolower(ind_name) %in% c("prostatic neoplasms","metastatic prostate carcinoma","malignant neoplasm of prostate","adenocarcinoma of prostate","prostate cancer recurrent","hormone refractory prostate cancer","hormone-resistant prostate cancer","prostate carcinoma"  ))

repodb_PC %>% filter(status != "Approved")
repodb %>% filter(status != "Approved") %>% pull(DetailedStatus) %>% na.omit %>% unique

```

```{r}
deep_list <- lapply(deep_list, function(fold) {
  fold[[1]] = lapply(fold[[1]], function(num){
    num[["data"]] = NULL
    num
  })
  fold
})


FULL_DATA_tuned_3[[2]][[1]] <- lapply(FULL_DATA_tuned_3[[2]][[1]], function(fold) {
  fold[[1]] = lapply(fold[[1]], function(num){
    num[["data"]] = NULL
    num
  })
  fold
})

`%r%` <- function(x, y) {
  if(length(y) == 1){
    x[[y]]
  } else if (length(y) > 1){
    x[y]
  } else {stop("Error")}
}

`%rr%` <- function(x, y) {
  if(length(y) == 1){
    lapply(x, "[[", y)
  } else {stop("Error")}
}


`%rr%<-` <- function(x, y, value) {
  if(length(y) == 1){
    x = lapply(x, function(i){
      i[[y]] = value
      i
    } )
  } else {stop("Error")}
  return(x)
}

`%rr%<-` <- function(x, y, value) {
  if (length(y) == 1) {
    x <- lapply(x, function(i) {
      i[[y]] <- value
      i
    })
  } else if (length(y) > 1) {
    x <- lapply(x, function(i) {
      i[[y[1]]] <- `[[<-`(i[[y[1]]], y[-1], value)
      i
    })
  } else { 
    stop("Error")
  }
  return(x)
}


deep_list = list(E1 = list(a = (1:3), b = (4:6), c = (7:9)),
                 E2 = list(a = (1:3), b = (4:6), c = (7:9)),
                 E3 = list(a = (1:3), b = (4:6), c = (7:9)))

deep_list %rr% 3 = 0
deep_list %rr% c(3,3)
deep_list %rr% c(3,3) = 0

add
```















