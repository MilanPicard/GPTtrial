
```{r}
library(glmnet)
library(tidyverse)
library(caret)
source("../utils.R")
DB_drug = readRDS("../DB_drug.rds")


FULL_DATA = readRDS("../Reseau/FULL_DATA.rds")

# Training set
FULL_DATA_small = FULL_DATA %>% filter(Drug %in% paste0("Drug_",unlist(DB_drug[-2])))
FULL_DATA_small_class = as.factor(FULL_DATA_small$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive)))

# Test set
FULL_DATA_test = FULL_DATA %>% filter(!Drug %in% paste0("Drug_",unlist(DB_drug[-2])))
FULL_DATA_test_class = as.factor(FULL_DATA_test$Drug %in% paste0("Drug_",unlist(DB_drug$DB_positive)))

```

```{r select features }
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

FULL_DATA_small_selected = select_features(df = FULL_DATA_small[, -1], class = FULL_DATA_small_class, nCores = 30)
saveRDS(FULL_DATA_small_selected, "FULL_DATA_small_selected.rds")

FULL_DATA_test_selected = FULL_DATA_test[, colnames(FULL_DATA_small_selected$new_df)]
FULL_DATA_test_drug = FULL_DATA_test$Drug

saveRDS(FULL_DATA_test_drug, "FULL_DATA_test_drug.rds")
saveRDS(FULL_DATA_test_selected, "FULL_DATA_test_selected.rds")
saveRDS(FULL_DATA_test_class, "FULL_DATA_test_class.rds")


```

# Run multiple models and gather predictions
```{r log reg with l1 more tests}
library("glmnet")
library("tidyverse")
library("caret")
library("mltools")
source("../utils.R")
DB_drug = readRDS("../DB_drug.rds")


FULL_DATA_small_selected = readRDS('FULL_DATA_small_selected.rds')
FULL_DATA_small_class = readRDS('FULL_DATA_small_class.rds')

FULL_DATA_test_selected = readRDS("FULL_DATA_test_selected.rds")
FULL_DATA_test_drug = readRDS("FULL_DATA_test_drug.rds")
```

```{r logistic regression full predict}
x = FULL_DATA_small_selected$new_df %>% as.matrix
x = scale(x)
y = FULL_DATA_small_class
  
set.seed(123)
nfolds <- 5
nrepeats <- 10
lambdas <- seq(0.001, 1, length = 100)
class_weights <- ifelse(y == TRUE, sum(y == FALSE) / length(y),sum(y == TRUE) / length(y))

# Function to perform cross-validation
cv_results <- replicate(nrepeats, {
  # Perform cross-validation using glmnet's built-in cv.glmnet function
  cv.glmnet(x, y, alpha = 1, family = "binomial", nfolds = nfolds, lambda = lambdas, weights = class_weights)
}, simplify = FALSE)

lambda.1se = mean(sapply(cv_results, function(x) x$lambda.1se))

THE_MODEL <- glmnet(x, y, family = "binomial", alpha = 1, lambda = lambda.1se, weights = class_weights)

df_test <- scale(as.matrix(FULL_DATA_test_selected), center = attr(x, "scaled:center"), scale = attr(x, "scaled:scale"))
predictions <- predict(THE_MODEL, newx = df_test, type = "response")
predictions_lasso = setNames(predictions[,1], str_remove(FULL_DATA_test_drug, "Drug_"))
saveRDS(predictions_lasso, "predictions_lasso.rds")

```

```{r svm full predict}
svm_try = CV_tuning_SVM(X = FULL_DATA_small_selected$new_df, Y = FULL_DATA_small_class, k = 5, repetition = 2, parameters = NULL)
svm_try_parse = parse_results(svm_try)
# svm_try_parse %>% arrange(desc(MCC))

ML = e1071::svm(x = FULL_DATA_small_selected$new_df, y = as.factor(FULL_DATA_small_class), kernel = "radial", 
                gamma = 0.000122,
                cost = 125,
                class.weights = c("TRUE" = 0.5, "FALSE" = 0.5), probability = TRUE)

predictions_svm = predict(ML, FULL_DATA_test_selected, probability = TRUE)
predictions_svm = setNames(attr(predictions_svm, "probabilities")[, 1], str_remove(FULL_DATA_test_drug, "Drug_"))
saveRDS(predictions_svm, "predictions_svm.rds")
```

```{r knn full predict}
knn_try = CV_tuning_KNN(X = FULL_DATA_small_selected$new_df, Y = FULL_DATA_small_class, k = 5, repetition = 2, parameters = NULL)
knn_try_parse = parse_results(knn_try)
knn_try_parse %>% arrange(desc(MCC))

df = data.frame(y = as.factor(FULL_DATA_small_class), FULL_DATA_small_selected$new_df)
ML = kknn::train.kknn(y~., df, kernel = "optimal", ks = 7, distance = 1.8)
predictions_knn = predict(ML, FULL_DATA_test_selected, type = "prob")
predictions_knn = setNames(predictions_knn[, 2], str_remove(FULL_DATA_test_drug, "Drug_"))
saveRDS(predictions_knn, "predictions_knn.rds")
```

```{r random forest full predict}
rf_try = CV_tuning_RF(X = FULL_DATA_small_selected$new_df, Y = FULL_DATA_small_class, k = 5, repetition = 2, parameters = NULL)
rf_try_parse = parse_results(rf_try)

ML =  ML = randomForest::randomForest(x = FULL_DATA_small_selected$new_df, y = as.factor(FULL_DATA_small_class),
                                      sampsize=c(25, 25),
                                      ntree=300, nodesize = 8)

predictions_rf = predict(ML, FULL_DATA_test_selected, type = "prob")
predictions_rf = setNames(predictions_rf[, 2], str_remove(FULL_DATA_test_drug, "Drug_"))
saveRDS(predictions_rf, "predictions_rf.rds")
```

```{r combine prediction}
prediction_table = data.frame(LASSO_prob = predictions_lasso,
                              SVM_prob = predictions_svm,
                              KNN_prob = predictions_knn,
                              RF_prob = predictions_rf)

prediction_table = rownames_to_column(prediction_table, "drug")
prediction_table = prediction_table %>% rowwise() %>% mutate(prob = mean(c(LASSO_prob, SVM_prob, KNN_prob, RF_prob)), .after = 1)
saveRDS(prediction_table, "prediction_table.rds")

prediction_table_info = inner_join(DB$drugs$general_information, prediction_table, by = c("primary_key" = "drug")) %>% select(primary_key,name,description,prob, LASSO_prob,SVM_prob,KNN_prob,RF_prob)
saveRDS(prediction_table_info, "prediction_table_info.rds")
```

```{r add biological information}
drug_classification = DB$drugs$drug_classification %>% filter(drugbank_id %in% prediction_table_info$primary_key) %>% select(drugbank_id, direct_parent)
pharmacology = DB$drugs$pharmacology %>% filter(drugbank_id %in% prediction_table_info$primary_key) %>% select(drugbank_id,indication, pharmacodynamics, mechanism_of_action)

target_info = DB$cett$targets$general_information %>% select(id, name, organism, parent_key) %>% setNames(c("target_id", "protein_name", "organism","drug_id"))
protein_info = DB$cett$targets$polypeptides$general_information %>% select(id, name, organism, general_function, gene_name,parent_id) %>% setNames(c("uniprot", "protein_name", "organism","protein_function","gene_name","target_id"))

drug_target = inner_join(target_info, protein_info) %>% filter(organism == "Humans" & drug_id %in% prediction_table_info$primary_key) %>% select(drug_id, uniprot, gene_name, protein_name, protein_function)

prediction_table_info_bio = full_join(full_join(full_join(prediction_table_info,drug_classification, by = c("primary_key"="drugbank_id")),
                                                drug_target, by = c("primary_key"="drug_id")),
                                      pharmacology, by = c("primary_key"="drugbank_id"))


prediction_table_info_bio = prediction_table_info_bio %>% select(primary_key,name,prob, direct_parent, protein_name, gene_name, protein_function, description, indication, pharmacodynamics, mechanism_of_action,uniprot, LASSO_prob, SVM_prob, KNN_prob,RF_prob)
prediction_table_info_bio %>% dim
# [1] 17778    16

```

```{r add more biological information}
# BiocManager::install("disgenet2r")
# BiocManager::install("msigdbr")

devtools::install_gitlab("medbio/disgenet2r")
library(msigdbr)
library(disgenet2r)
prostate_targets <- disgenet2r::disease2gene("C0033578") # Prostate cancer MeSH ID
dis_res <- disease2gene( "UMLS_C0028754", database = "CURATED" , score = c(0,1))

Disgenet = read_tsv("Disgenet/full_search.tsv")
Disgenet_PC = Disgenet %>% filter(tolower(Disease) %in% c("adenocarcinoma of prostate","androgen independent prostate cancer","benign prostatic hyperplasia","high-grade prostatic intraepithelial neoplasia","hormone refractory prostate cancer","hormone sensitive prostate cancer","malignant neoplasm of prostate","metastasis from malignant tumor of prostate","metastatic castration-resistant prostate cancer","metastatic prostate carcinoma","non-metastatic prostate cancer","progression of prostate cancer","prostate cancer recurrent","prostate carcinoma","prostatic hyperplasia","prostatic intraepithelial neoplasias","prostatic neoplasms")) %>% pull(Gene) %>% unique

cancer_list = str_subset(tolower(Disgenet$Disease), "cancer|carcino|neoplas|tumor") %>% unique
Disgenet_Cancer = Disgenet %>% filter(tolower(Disease) %in% cancer_list) %>% pull(Gene) %>% unique


hallmark_genes <- msigdbr(species = "Homo sapiens", category = "H")
hallmark_genes = hallmark_genes %>% select(gs_name, human_gene_symbol, gs_description) %>% mutate(gs_name = tolower(str_remove(gs_name, "HALLMARK_")))
hallmark_genes = hallmark_genes[c("gs_name","human_gene_symbol")] %>% group_by(human_gene_symbol) %>% summarise(hallmarks = paste(unique(gs_name), collapse = " ; "))

prediction_table_info_bio_cancer = left_join(prediction_table_info_bio, hallmark_genes, by = c("gene_name" = "human_gene_symbol"))
prediction_table_info_bio_cancer = prediction_table_info_bio_cancer %>% select(primary_key, name, prob, direct_parent, protein_name, hallmarks, everything())
prediction_table_info_bio_cancer %>% dim
# [1] 17778    17

# CTD_genes_diseases = read_csv("CTD_genes_diseases.csv", skip = 27)
# CTD_genes_diseases_PC = CTD_genes_diseases %>% filter(str_detect(tolower(DiseaseName), "prostat"))
# write_csv(CTD_genes_diseases_PC, "CTD_genes_diseases_PC.csv")
CTD_genes_diseases_PC = read_tsv("CTD_genes_diseases_PC.tsv")
CTD_genes_diseases_PC = CTD_genes_diseases_PC %>% filter(!is.na(DirectEvidence))

pharos_prostate_cancer_target = read_csv("query results.csv")
PC_genes = union(CTD_genes_diseases_PC$`# GeneSymbol`,pharos_prostate_cancer_target$Symbol)
prediction_table_info_bio_cancer = prediction_table_info_bio_cancer %>% mutate(is_target_PC_related = ifelse(gene_name %in% PC_genes, "yes", "no"), .after = hallmarks)

# PC_Signature_uniprot = readRDS("../data/PC_Signature_uniprot.rds")
# prediction_table_info_bio_cancer = prediction_table_info_bio_cancer %>% mutate(is_target_PC_signature = ifelse(uniprot %in% unique(unlist(PC_Signature_uniprot)), "yes", "no"),.after = is_target_PC_related)
# prediction_table_info_bio_cancer$is_target_PC_signature = NULL
prediction_table_info_bio_cancer %>% dim
saveRDS(prediction_table_info_bio_cancer, "prediction_table_info_bio_cancer.rds")
prediction_table_info_bio_cancer = readRDS("prediction_table_info_bio_cancer.rds")
```

```{r add pubcmed text mining}
library("rentrez")
library("glmnet")
library("tidyverse")
library("caret")
library("mltools")
source("../utils.R")
prediction_table_info = readRDS("prediction_table_info.rds")

# Function to search PubMed and retrieve article titles
search_and_get_titles <- function(drug_name, disease) {
  # Search for articles related to the drug and prostate cancer
  query <- paste(drug_name, disease)
  entrez_search(db = "pubmed", term = query, retmax = 30)
}

safe_search_and_get_titles <- function(drug_name, disease) {
  attempt <- 1
  max_attempts <- 3  # You can adjust this as needed
  while (attempt <= max_attempts) {
    tryCatch({
      result <- search_and_get_titles(drug_name = drug_name, disease = disease)
      return(result)
    }, error = function(e) {
      cat("Error encountered for", drug_name, ": ", e$message, "\n")
      if (attempt < max_attempts) {
        cat("Retrying in 5 seconds...\n")
        Sys.sleep(5)
        attempt <- attempt + 1
      } else {
        cat("Failed after", max_attempts, "attempts.\n")
        return(NULL)  # Return NULL or any other appropriate value in case of failure
      }
    })
  }
}

# Get titles for each drug
result_pcs = list()
result_cs = list()
results = list()
k = 0
for(x in prediction_table_info$name) {
  k = k+1
  print(paste("Doing molecule",k,"named",x))
  result_pc <- safe_search_and_get_titles(drug_name = x, disease = "prostate cancer")
  result_c <- safe_search_and_get_titles(drug_name = x, disease = "cancer")
  result <- safe_search_and_get_titles(drug_name = x, disease = "")
  add(result_pcs, result_pc)
  add(result_cs, result_c)
  add(results, result)
}

names(result_pcs) = prediction_table_info$name
names(result_cs) = prediction_table_info$name
names(results) = prediction_table_info$name
Text_mining_predictions = setNames(list(result_pcs,result_cs,results), c("prostate","cancer","global"))
saveRDS(Text_mining_predictions, "Text_mining_predictions.rds")

Text_mining_predictions = readRDS("Text_mining_predictions.rds")
Text_mining_predictions_count = data.frame(prostate = sapply(Text_mining_predictions$prostate, function(x) x$count),cancer = sapply(Text_mining_predictions$cancer, function(x) x$count),global = sapply(Text_mining_predictions$global, function(x) x$count))
Text_mining_predictions_count = rownames_to_column(Text_mining_predictions_count, "name")

prediction_table_info_bio_cancer_mining = full_join(Text_mining_predictions_count, prediction_table_info_bio_cancer) %>% select(primary_key, name, prob, prostate, cancer, global, direct_parent, hallmarks, is_target_PC_related, everything())
saveRDS(prediction_table_info_bio_cancer_mining, "prediction_table_info_bio_cancer_mining.rds")

```

```{r get the results}
prediction_table_info_bio_cancer_mining = readRDS("prediction_table_info_bio_cancer_mining.rds")
prediction_table_info_bio_cancer_mining[is.na(prediction_table_info_bio_cancer_mining)] = ""
prediction_table_info_bio_cancer_mining = prediction_table_info_bio_cancer_mining %>% arrange(desc(prob))
# prediction_table_info_bio_cancer_mining = prediction_table_info_bio_cancer_mining %>% mutate(across(contains("prob"), ~ round(.,4)))

prediction_table_info_bio_cancer_mining = prediction_table_info_bio_cancer_mining %>% mutate(prostate = if_else(global == 0, 0, prostate),
                                                                                             cancer = if_else(global == 0, 0, cancer))

prediction_table_info_bio_cancer_mining = prediction_table_info_bio_cancer_mining %>% mutate(prostate = if_else(global < prostate, 0, prostate),
                                                                                             cancer = if_else(global < prostate, 0, cancer))

prediction_table_info_bio_cancer_mining = prediction_table_info_bio_cancer_mining %>% mutate(prostate = if_else(name %in% known_found_molecule, 0, prostate),
                                                                                             cancer = if_else(name %in% known_found_molecule, 0, cancer),
                                                                                             global = if_else(name %in% known_found_molecule, 0, global))

prediction_table_info_bio_cancer_mining = prediction_table_info_bio_cancer_mining %>% mutate(rank = match(name, unique(name)), .after = primary_key)

write_csv(prediction_table_info_bio_cancer_mining, "prediction_table_info_bio_cancer_mining.csv")
prediction_table_info_bio_cancer_mining = read_csv("prediction_table_info_bio_cancer_mining.csv")
```

```{r}

prediction_table_info_bio_cancer_mining

```

```{r add similarity groups by structure}
library(igraph)
fing0_df = readRDS("../data/fing0_df.rds")
drug_smile = DB$drugs$calculated_properties %>% filter(kind == "SMILES") %>% dplyr::select(parent_key, value)
fing0_df_long = fing0_df %>% `colnames<-`(drug_smile$parent_key) %>% mutate(Drug = drug_smile$parent_key, .before=1) %>% pivot_longer(-1)

network_drug_community = fing0_df_long %>% filter(value >= 0.9) %>% filter(Drug != name) %>% graph_from_data_frame()
components <- components(network_drug_community)
community_list <- lapply(unique(components$membership), function(i) {
  V(network_drug_community)$name[components$membership == i]
})
names(community_list) = paste0("group ", 1:nbr(community_list))
community_list[1095]

group_structural = sapply(prediction_table_info_bio_cancer_mining$primary_key, function(drug) ifelse(drug %in% unlist(community_list), names(community_list)[sapply(community_list, function(x) drug %in% x)], "group 0"))
prediction_table_info_bio_cancer_mining_struct = prediction_table_info_bio_cancer_mining %>% mutate(group_structural = group_structural, .after = global)
# write_csv(prediction_table_info_bio_cancer_mining_struct, "prediction_table_info_bio_cancer_mining_struct.csv")
prediction_table_info_bio_cancer_mining_struct %>% head
```

```{r}
library(clusterProfiler)
library(org.Hs.eg.db) 

# GO enrichment analysis
gene = prediction_table_info_bio_cancer_mining_struct %>% filter(prob >= 0.8) %>% pull(gene_name) %>% unique
iugu = unlist(mapIds(org.Hs.eg.db, gene, "ENTREZID", "SYMBOL"))

go_enrichment <- enrichGO(
  gene         = iugu,
  OrgDb        = org.Hs.eg.db,
  keyType      = "ENTREZID",
  ont          = "BP",        # BP for Biological Process, MF for Molecular Function, CC for Cellular Component
  pAdjustMethod = "BH",       # Benjamini-Hochberg adjustment for multiple testing
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2
)

go_enrichment@result %>% arrange(desc(GeneRatio), pvalue)

prediction_table_info_bio_cancer_mining_struct %>% filter(rank <= 30)
```

```{r}
FULL_DATA_small_selected = readRDS("FULL_DATA_small_selected.rds")
FULL_DATA_test_drug = readRDS("FULL_DATA_test_drug.rds")
FULL_DATA_test_selected = readRDS("FULL_DATA_test_selected.rds")
FULL_DATA_test_class = readRDS("FULL_DATA_test_class.rds")

Prot_list_name = FULL_DATA_test_selected %>% names %>% str_subset("GraphRandomWalkUp_Prot_") %>% str_remove("GraphRandomWalkUp_Prot_")
library(org.Hs.eg.db)
 
intersect(mapIds(org.Hs.eg.db, Prot_list_name, "SYMBOL", "UNIPROT") %>% na.omit() %>% unname %>% as.vector, c("C5","PARP1","PARP","ESR1", "SCN1A", "NR1l2","MC4R","GLP1R", "CALCR", "RAMP1", "RAMP2", "RAMP3","OPRL1","MC1R","KIT", "KDR"))



GO_list_name = FULL_DATA_test_selected %>% names %>% str_subset("GraphRandomWalkUp_GO_") %>% str_remove("GraphRandomWalkUp_GO_")
GO_list_df = do.call(rbind, lapply(GO_list_name, function(x) GOfuncR::get_ids(x)[1, ]))

GO_list_parent = GOfuncR::get_parent_nodes(GO_list_df$go_id)

GO_list_parent %>% filter(parent_go_id %in% GO_list_df$go_id & distance != 0)

str_subset(GO_list_df$node_name, "response")

library("rrvgo")
simMatrix <- calculateSimMatrix(GO_list_df$go_id,
                                orgdb="org.Hs.eg.db",
                                ont="BP",
                                method="Rel")
simMatrix2 <- calculateSimMatrix(GO_list_df$go_id,
                                orgdb="org.Hs.eg.db",
                                ont="MF",
                                method="Rel")

reducedTerms <- reduceSimMatrix(simMatrix,
                                threshold=0.7,
                                orgdb="org.Hs.eg.db")
reducedTerms2 <- reduceSimMatrix(simMatrix2,
                                threshold=0.7,
                                orgdb="org.Hs.eg.db")
treemapPlot(reducedTerms)
treemapPlot(reducedTerms2)

treemapPlot(reduceSimMatrix(simMatrix, threshold=0.8, orgdb="org.Hs.eg.db"))
treemapPlot(reduceSimMatrix(simMatrix, threshold=0.6, orgdb="org.Hs.eg.db"))
ahahha = reduceSimMatrix(simMatrix, threshold=0.6, orgdb="org.Hs.eg.db")
liugk = reduceSimMatrix(simMatrix, threshold=0.7, orgdb="org.Hs.eg.db")
liufkyf = reduceSimMatrix(simMatrix, threshold=0.8, orgdb="org.Hs.eg.db")
treemapPlot(ahahha)
treemapPlot(liugk)
treemapPlot(liufkyf)



GO_list_parent %>% group_by(parent_name) %>% mutate(Repe = n()) %>% filter(Repe < 10 & Repe > 2) %>%  arrange(desc(Repe)) %>% pull(parent_name) %>% table %>% sort(T)
"
cell migration 
chemotaxis                                                
nitrogen compound metabolic process 
response to stress
cell surface receptor signaling pathway
regulation of developmental process
response to lipid
cellular component biogenesis
regulation of cell population proliferation
response to growth factor
defense response
muscle tissue development
"
sum(GO_list_parent$parent_name == "sulfur amino acid transport")
reduceSimMatrix(simMatrix, threshold=0.6, orgdb="org.Hs.eg.db")
GO_list_df$node_name %>% sort %>% cat(sep = "\n")
```

```{r}
Biological_network = readRDS("../data/Biological_network.rds")
potential_drug_candidates = paste0("Drug_",c("DB15636", "DB03509", "DB04930", "DB11700", "DB01278", "DB16013", "DB04931", "DB14027", "DB05575"))
rwr_drug_candidates = DiscoNet::extract_by_rwr(Biological_network, start_nodes = potential_drug_candidates)
rwr_drug_candidates = cbind(Target = c("ColMeans", rwr_drug_candidates[[1]]), rbind(colMeans = colMeans(rwr_drug_candidates[, -1], na.rm = TRUE), rwr_drug_candidates[, -1]))
rwr_drug_candidates = column_to_rownames(rwr_drug_candidates, "Target")

df_ordered <- rwr_drug_candidates[, order(unlist(rwr_drug_candidates[1, ]), decreasing = TRUE)]

intersect(df_ordered[, str_subset(names(df_ordered), "GO")] %>% names %>% .[1:10000], paste0("GO_",make.names(GO_list_parent$parent_name)))


lapply(ego(Biological_network, nodes = potential_drug_candidates, order = 3, mindist = 3), names)
```



```{r known_found_molecule}
known_found_molecule = c(
"Prezatide copper",
"N-1,10-phenanthrolin-5-ylacetamide",
"1h-Benoximidazole-2-Carboxylic Acid",
"4-(2-HYDROXYPHENYLTHIO)-1-BUTENYLPHOSPHONIC ACID",
"3',5'-DIFLUOROBIPHENYL-4-CARBOXYLIC ACID",
"3,4-Dihydro-5-Methyl-Isoquinolinone",
"2',6'-DIFLUOROBIPHENYL-4-CARBOXYLIC ACID",
"4-Methoxy-E-Rhodomycin T",
"Ethyl piperidinoacetylaminobenzoate",
"2-(5-HYDROXY-NAPHTHALEN-1-YL)-1,3-BENZOOXAZOL-6-OL",
"Estradiol valerate",
"3,4-Dihydro-2h-Pyrrolium-5-Carboxylate",
"2-AMINO-4-CHLORO-3-HYDROXYBENZOIC ACID",
"3,7-DIHYDROXYNAPHTHALENE-2-CARBOXYLIC ACID",
"1,8-Di-Hydroxy-4-Nitro-Anthraquinone",
"Acetic Acid Salicyloyl-Amino-Ester",
"6-(2-HYDROXY-CYCLOPENTYL)-7-OXO-HEPTANAMIDINE",
"2-(4-HYDROXY-5-PHENYL-1H-PYRAZOL-3-YL)-1H-BENZOIMIDAZOLE-5-CARBOXAMIDINE",
"N-(PARA-GLUTARAMIDOPHENYL-ETHYL)-PIPERIDINIUM-N-OXIDE",
"Para-Mercury-Benzenesulfonic Acid",
"N-(4-CARBAMIMIDOYL-3-CHORO-PHENYL)-2-HYDROXY-3-IODO-5-METHYL-BENZAMIDE",
"4-TERT-BUTYLBENZENESULFONIC ACID",
"N-(7-CARBAMIMIDOYL-NAPHTHALEN-1-YL)-3-HYDROXY-2-METHYL-BENZAMIDE",
"3,6-dihydroxy-xanthene-9-propionic acid",
"2-(2-Hydroxy-5-Methoxy-Phenyl)-1h-Benzoimidazole-5-Carboxamidine",
"3-hydroxyisoxazole-4-carboxylic acid",
"N-(TRANS-4'-NITRO-4-STILBENYL)-N-METHYL-5-AMINO-PENTANOIC ACID",
"5-phenyl-2-keto-valeric acid",
"2,6-diaminoquinazolin-4-ol",
"1,2,4-Triazole-Carboxamidine",
"2,5-Dimethoxy-4-ethylthioamphetamine",
"4-(3-amino-1H-indazol-5-yl)-N-tert-butylbenzenesulfonamide",
"Dioxaphetyl butyrate",
"[2,4,6-Triisopropyl-Phenylsulfonyl-L-[3-Amidino-Phenylalanine]]-Piperazine-N'-Beta-Alanine",
"1,1'-BIPHENYL-2-SULFINIC ACID",
"1-(O-Carboxy-Phenylamino)-1-Deoxy-D-Ribulose-5-Phosphate",
"6,7-dioxo-5H-8-ribitylaminolumazine",
"2,6-Diamino-8-Propylsulfanylmethyl-3h-Quinazoline-4-One",
"N-(2-Flouro-Benzyl)-4-Sulfamoyl-Benzamide",
"N-(2,6-Diflouro-Benzyl)-4-Sulfamoyl-Benzamide",
"7,8-dihydroxy-4-phenyl-2H-chromen-2-one",
"N-(2,3-DIFLUORO-BENZYL)-4-SULFAMOYL-BENZAMIDE",
"1,8-Di-Hydroxy-4-Nitro-Xanthen-9-One",
"1-Deoxy-1-Thio-Heptaethylene Glycol",
"Adenosine-5'-[Beta, Gamma-Methylene]Triphosphate",
"Adenosine 5'-[gamma-thio]triphosphate",
"2-Amino-3-Hydroxy-3-Phosphonooxy-Propionic Acid",
"4-HYDROXY-5-IODO-3-NITROPHENYLACETYL-EPSILON-AMINOCAPROIC ACID ANION",
"4-HYDROXY-3-NITROPHENYLACETYL-EPSILON-AMINOCAPROIC ACID ANION",
"3,8-DIBROMO-7-HYDROXY-4-METHYL-2H-CHROMEN-2-ONE",
"1-[2-DEOXYRIBOFURANOSYL]-2,4-DIFLUORO-5-METHYL-BENZENE-5'MONOPHOSPHATE",
"Decylsulfonic acid",
"1,4-Dideoxy-5-Dehydro-O2-Sulfo-Glucuronic Acid",
"2',3'-dideoxy-3'-fluoro-urididine-5'-diphosphate",
"1-Hexadecanosulfonic Acid",
"GDP-alpha-D-mannuronic acid",
"4,4'-Biphenyldiboronic Acid",
"3,5-dibromobiphenyl-4-ol",
"2-Aminoethanimidic Acid",
"2,3-diphenyl-1H-indole-7-carboxylic acid",
"N-[1-(4-CARBAMIMIDOYL-BENZYLCARBAMOYL)-3-METHYLSULFANYL-PROPYL]-3-HYDROXY-2-PROPOXYAMINO-BUTYRAMID",
"1,4-dithio-alpha-D-glucopyranose",
"1,4-dithio-beta-D-glucopyranose",
"Ethylaminobenzylmethylcarbonyl Group",
"Norepinephrine",
"N,O6-Disulfo-Glucosamine",
"METHYLAMINO-PHENYLALANYL-LEUCYL-HYDROXAMIC ACID",
"(1-Benzyl-5-methoxy-2-methyl-1H-indol-3-yl)acetic acid",
"2-phenyl-1H-imidazole-4-carboxylic acid",
"Olaparib-bodipy FL",
"1-METHYL-3-PHENYL-1H-PYRAZOL-5-YLSULFAMIC ACID",
"Picosulfuric acid",
"1,6-Di-O-Phosphono-D-Allitol",
"1,6-DI-O-PHOSPHONO-D-MANNITOL",
"2-Amino-3-Oxo-4-Sulfo-Butyric Acid",
"Inhibitor BEA409",
"2,6-dibromo-4-phenoxyphenol",
"Cmp-2-Keto-3-Deoxy-Octulosonic Acid",
"alpha,alpha-Dibromo-D-camphor",
"3,4-Methylenedioxy-N-isopropylamphetamine",
"N-[1-Hydroxycarboxyethyl-Carbonyl]Leucylamino-2-Methyl-Butane",
"Inhibitor BEA388",
"S-(N-hydroxy-N-iodophenylcarbamoyl)glutathione",
"2-Thiomethyl-3-Phenylpropanoic Acid",
"2-Benzyl-3-Iodopropanoic Acid",
"Benzoyl-tyrosine-alanine-fluoro-methyl ketone",
"Cytidine-5'-Monophosphate-5-N-Acetylneuraminic Acid",
"1,3,2-Dioxaborolan-2-Ol",
"Alpha-Amino-2-Indanacetic Acid",
"2-ACETYLAMINO-4-METHYL-PENTANOIC ACID (1-FORMYL-2-PHENYL-ETHYL)-AMIDE",
"3,5-Dihydro-5-Methylidene-4h-Imidazol-4-On",
"2,4-Diamino-5-phenyl-6-ethylpyrimidine",
"METHOXYUNDECYLPHOSPHINIC ACID",
"5,6-Cyclic-Tetrahydropteridine",
"O2-Sulfo-Glucuronic Acid",
"16,17-Androstene-3-Ol",
"5-Chloryl-2,4,6-quinazolinetriamine",
"3',5'-Dinitro-N-Acetyl-L-Thyronine",
"(R)-1-Para-Nitro-Phenyl-2-Azido-Ethanol",
"3-BENZYLOXYCARBONYLAMINO-2-HYDROXY-4-PHENYL-BUTYRIC ACID",
"Inhibitor Bea428",
"N-acetyl-alpha-neuraminic acid",
"S,S-Propylthiocysteine",
"5,10-Methylene-6-Hydrofolic Acid",
"2-Anhydro-3-Fluoro-Quinic Acid",
"1,4-Dideoxy-O2-Sulfo-Glucuronic Acid",
"N(2)-succinyl-L-arginine",
"Deoxyamidinoproclavaminic acid",
"4-phospho-L-threonic acid",
"4-phospho-D-erythronic acid",
"2-Tridecanoyloxy-Pentadecanoic Acid",
"1-Hydroxyamine-2-Isobutylmalonic Acid",
"2,3-Di-O-Sulfo-Alpha-D-Glucopyranose",
"Cyclic adenosine monophosphate",
"5-Amino-3-Methyl-Pyrrolidine-2-Carboxylic Acid",
"5-phospho-D-arabinohydroxamic acid",
"N-Ethyl-5'-Carboxamido Adenosine",
"5,6-dihydroxy-NADP",
"7-methyl-5'-guanylic acid",
"5-Hydroxyamino-3-Methyl-Pyrrolidine-2-Carboxylic Acid",
"5-monophosphate-9-beta-D-ribofuranosyl xanthine",
"2,4-deoxy-4-guanidino-5-N-acetyl-neuraminic acid",
"(R)-2-Hydroxy-3-Sulfopropanoic Acid",
"7-thionicotinamide-adenine-dinucleotide phosphate",
"N-acetyl-beta-neuraminic acid",
"5-Amino-6-cyclohexyl-4-hydroxy-2-isobutyl-hexanoic acid",
"Butedronic acid",
"ethyl 2-amino-4-hydroxypyrimidine-5-carboxylate",
"5-AMINO-6-CYCLOHEXYL-4-HYDROXY-2-ISOPROPYL-HEXANOIC ACID",
"2',4'-Dinitrophenyl-2deoxy-2-Fluro-B-D-Cellobioside",
"11-TRANS-13-TRANS-15-CIS-OCTADECATRIENOIC ACID",
"5-Phosphoarabinonic Acid",
"4-Hydroxy-Aconitate Ion",
"3'-phospho-5'-adenylyl sulfate",
"O6-CYCLOHEXYLMETHOXY-2-(4'-SULPHAMOYLANILINO) PURINE",
"2-propenyl-N-acetyl-neuramic acid",
"2',3'-Dideoxycytidine-5'-Monophosphate",
"3-Trimethylsilylsuccinic Acid",
"(4-Hydroxymaltosephenyl)Glycine",
"N-Hydroxy-N-Isopropyloxamic Acid",
"2-OXOHEPTYLPHOSPHONIC ACID",
"4-(4-fluoro-phenylazo)-5-imino-5H-pyrazol-3-ylamine",
"3,3-Dichloro-2-Phosphonomethyl-Acrylic Acid",
"L-2-Amino-6-Methylene-Pimelic Acid",
"Visometin cation",
"Alpha-Benzyl-Aminobenzyl-Phosphonic Acid",
"alpha-D-glucopyranosyl-2-carboxylic acid amide",
"IMIDAZOPYRIDAZIN 1",
"2-PHENYLAMINO-4-METHYL-5-ACETYL THIAZOLE",
"1,2-Docosahexanoyl-sn-glycero-3-phosphoserine",
"PANTOTHENYL-AMINOETHANOL-11-PIVALIC ACID",
"Trifluoro-thiamin phosphate",
"3-Methyl-5-Sulfo-Pyrrolidine-2-Carboxylic Acid",
"2-Phenylamino-Ethanesulfonic Acid",
"2,5-Dimethylpyrimidin-4-Amine",
"2-Iodobenzylthio Group",
"4-Methyl-Pyrroline-5-Carboxylic Acid",
"Tidiacic arginine",
"Leptazoline B",
"Leptazoline C",
"Leptazoline D",
"Ammonium trichlorotellurate",
"1,2-icosapentoyl-sn-glycero-3-phosphoserine",
"4,6-dideoxy-4-amino-alpha-D-glucose",
"1,2-Di-N-Pentanoyl-Sn-Glycero-3-Dithiophosphocholine",
"Retinamidic acid",
"NB-124 sulfate",
"7,8-Diamino-Nonanoic Acid",
"1,2-Dipalmitoyl-Phosphatidyl-Glycerole",
"2,3-Dideoxyfucose",
"(3-Chloro-4-Propoxy-Phenyl)-Acetic Acid",
"N-Bromoacetyl-Aminoethyl Phosphate",
"N-[4-hydroxymethyl-cyclohexan-6-yl-1,2,3-triol]-4,6-dideoxy-4-aminoglucopyranoside",
"2,6-Anhydro-3-Deoxy-D-Erythro-Hex-2-Enonic Acid",
"Alexitol sodium",
"3,4-Dihydro-2'-deoxyuridine-5'-monophosphate",
"2,6-anhydro-3-deoxy-3-fluoronononic acid",
"4,5-Dihydroxy-Tetrahydro-Pyran-2-Carboxylic Acid",
"threo-3-methyl-L-aspartic acid",
"3,4-Epoxybutyl-Alpha-D-Glucopyranoside",
"4-Deoxy-D-Glucuronic Acid",
"2-Aminopropanedioic Acid",
"Cyclohexanepropanoic acid",
"(Diaminomethyl-Methyl-Amino)-Acetic Acid",
"3-deoxy-D-lyxo-hexonic acid",
"2,3-Dihydroxy-5-Oxo-Hexanedioate",
"5-Mercapto-2-Nitro-Benzoic Acid",
"4,5-Dehydro-D-Glucuronic Acid",
"4-Deoxy-D-Mannuronic Acid",
"2-Amino-3-Ketobutyric Acid",
"4,5-Dehydro-L-Iduronic Acid",
"3-Oxo-Pentadecanoic Acid",
"2-Oxo-3-Pentenoic Acid",
"3-Amino-3-Oxopropanoic Acid",
"Alpha-Methylisocitric Acid",
"2,3-Dihydroxy-Valerianic Acid",
"Valiloxibic acid",
"Acetylamino-Acetic Acid",
"KIT-13",
"Glycyl-L-a-Aminopimelyl-E-(D-2-Aminoethyl)Phosphonate")


```





# Other stuff
```{r get predicted drugs info}

geg = prediction_table_info %>% arrange(desc(prob)) %>% pull(primary_key) %>% .[1:50]
intersect_plot(prediction_table_info %>% arrange(desc(LASSO_prob)) %>% pull(primary_key) %>% .[1:50],
               prediction_table_info %>% arrange(desc(SVM_prob)) %>% pull(primary_key) %>% .[1:50],
               prediction_table_info %>% arrange(desc(KNN_prob)) %>% pull(primary_key) %>% .[1:50],
               prediction_table_info %>% arrange(desc(prob)) %>% pull(primary_key) %>% .[1:50], Names = c(1,2,3,"full"))


prediction_table_info[5:7] %>% cor

drug_info = DB$drugs$general_information %>% filter(primary_key %in% (prediction_table %>% filter(prob > 0.5) %>% pull(drug))) %>% select(primary_key, name)



drug_classification = DB$drugs$drug_classification %>% filter(drugbank_id %in% drug_info$primary_key) %>% select(drugbank_id, direct_parent)
pharmacology = DB$drugs$pharmacology %>% filter(drugbank_id %in% drug_info$primary_key) %>% select(drugbank_id,indication, pharmacodynamics, mechanism_of_action)
targets = DB$cett$targets$general_information %>% filter(parent_key %in% drug_info$primary_key) %>% filter(organism == "Humans") %>% select(parent_key, name) %>% group_by(parent_key) %>% summarise(name = paste(name, collapse = " ; ")) %>% setNames(c("parent_key", "target"))

drug_info = full_join(full_join(full_join(drug_info,drug_classification, by = c("drug"="drugbank_id")),
                                targets, by = c("drug"="parent_key")),
                      pharmacology, by = c("drug"="drugbank_id"))



intersect_plot(drug_info %>% arrange(desc(predicted_drugs)) %>% pull(drug) %>% .[1:150], drug_info_4$`DrugBank ID`, Names = c("svm", "lasso"), Title = "Overlap between predicted drugs", colors = 2)

```


```{r}
library("rentrez")
prediction_table_info = readRDS("prediction_table_info.rds")

# Function to search PubMed and retrieve article titles
search_and_get_titles <- function(drug_name, disease) {
  # Search for articles related to the drug and prostate cancer
  query <- paste(drug_name, disease)
  search_results <- entrez_search(db = "pubmed", term = query, retmax = 30)
  
  # Retrieve details of the articles
  ids <- search_results$ids
  if (length(ids) == 0) {
    return(data.frame(id = character(0), dates = character(0), count = character(0), title = character(0)))
  } else if(length(ids) == 1){
    article_details <- entrez_summary(db = "pubmed", id = ids)
    results <- data.frame(id = ids, dates = str_split(article_details$sortpubdate, " ")[[1]][[1]], title = article_details$title, count = search_results$count, stringsAsFactors = FALSE)
  } else {
    # Fetch article details
    article_details <- entrez_summary(db = "pubmed", id = ids)
    # Extract IDs and titles
    titles <- sapply(article_details, function(x) x$title)
    dates <- sapply(article_details, function(x) sapply(str_split(x$sortpubdate, " "), "[[", 1))
    results <- data.frame(id = ids, dates = dates, title = titles, count = search_results$count, stringsAsFactors = FALSE)
  }
  return(results)
}

safe_search_and_get_titles <- function(drug_name, disease) {
  attempt <- 1
  max_attempts <- 3  # You can adjust this as needed
  while (attempt <= max_attempts) {
    tryCatch({
      result <- search_and_get_titles(drug_name = drug_name, disease = disease)
      return(result)
    }, error = function(e) {
      cat("Error encountered for", drug_name, ": ", e$message, "\n")
      if (attempt < max_attempts) {
        cat("Retrying in 5 seconds...\n")
        Sys.sleep(5)
        attempt <- attempt + 1
      } else {
        cat("Failed after", max_attempts, "attempts.\n")
        return(NULL)  # Return NULL or any other appropriate value in case of failure
      }
    })
  }
}

# Get titles for each drug
result_pcs = list()
result_cs = list()
results = list()
k = 0
for(x in prediction_table_info$name) {
  k = k+1
  print(paste("Doing molecule",k,"named",x))
  result_pc <- safe_search_and_get_titles(drug_name = x, disease = "prostate cancer")
  result_c <- safe_search_and_get_titles(drug_name = x, disease = "cancer")
  result <- safe_search_and_get_titles(drug_name = x, disease = "")
  add(result_pcs, result_pc)
  add(result_cs, result_c)
  add(results, result)
}

names(result_pcs) = drug_info$synonym
names(result_cs) = drug_info$synonym
names(results) = drug_info$synonym
Text_mining_predictions = setNames(list(result_pcs,result_cs,results), c("prostate","cancer","global"))
saveRDS(Text_mining_predictions, "Text_mining_predictions.rds")


```

```{r log reg with l1 more tests}
x = FULL_DATA_small_selected$new_df %>% as.matrix
x = scale(x)
y = FULL_DATA_small_class
  
set.seed(123)
nfolds <- 5
nrepeats <- 10
lambdas <- seq(0.001, 1, length = 100)
class_weights <- ifelse(y == TRUE, sum(y == FALSE) / length(y),sum(y == TRUE) / length(y))

# Function to perform cross-validation
cv_results <- replicate(nrepeats, {
  # Perform cross-validation using glmnet's built-in cv.glmnet function
  cv.glmnet(x, y, alpha = 1, family = "binomial", nfolds = nfolds, lambda = lambdas, weights = class_weights)
}, simplify = FALSE)

lambda.1se = mean(sapply(cv_results, function(x) x$lambda.1se))

THE_MODEL <- glmnet(x, y, family = "binomial", alpha = 1, lambda = lambda.1se, weights = class_weights)

df_test <- scale(as.matrix(FULL_DATA_test_selected), center = attr(x, "scaled:center"), scale = attr(x, "scaled:scale"))
predictions <- predict(THE_MODEL, newx = df_test, type = "response")

predicted_drugs = setNames(predictions[,1], str_remove(FULL_DATA_test_drug, "Drug_"))
saveRDS(predicted_drugs, "predicted_drugs.rds")
```

```{r get drug names}
library("dbparser")
library("glmnet")
library("tidyverse")
library("caret")
library("mltools")
library("igraph")
source("../utils.R")
DB_drug = readRDS("../DB_drug.rds")
predicted_drugs = readRDS("predicted_drugs.rds")
DB = readRDS("../data/DrugBank.rds")

predicted_drugs_info = DB$drugs$general_information %>% filter(primary_key %in% names(predicted_drugs)[predicted_drugs >= 0.5])

all_drug_names = DB$drugs$synonyms %>% filter(`drugbank-id` %in% names(predicted_drugs)[predicted_drugs >= 1]) %>% pull(synonym)

drug_info = full_join(
          DB$drugs$general_information %>% filter(primary_key %in% names(predicted_drugs)[predicted_drugs >= 1]) %>% select(primary_key, name),
          DB$drugs$synonyms %>% filter(`drugbank-id` %in% names(predicted_drugs)[predicted_drugs >= 1]) %>% select(synonym,`drugbank-id`),
          by=c("primary_key"="drugbank-id"))


drug_info = drug_info %>% mutate(synonym = ifelse(is.na(synonym), name, synonym))

```

```{r get pubcmed verif on these 145}
library("rentrez")

# Function to search PubMed and retrieve article titles
search_and_get_titles <- function(drug_name, disease) {
  # Search for articles related to the drug and prostate cancer
  query <- paste(drug_name, disease)
  search_results <- entrez_search(db = "pubmed", term = query, retmax = 30)
  
  # Retrieve details of the articles
  ids <- search_results$ids
  if (length(ids) == 0) {
    return(data.frame(id = character(0), dates = character(0), count = character(0), title = character(0)))
  } else if(length(ids) == 1){
    article_details <- entrez_summary(db = "pubmed", id = ids)
    results <- data.frame(id = ids, dates = str_split(article_details$sortpubdate, " ")[[1]][[1]], title = article_details$title, count = search_results$count, stringsAsFactors = FALSE)
  } else {
    # Fetch article details
    article_details <- entrez_summary(db = "pubmed", id = ids)
    # Extract IDs and titles
    titles <- sapply(article_details, function(x) x$title)
    dates <- sapply(article_details, function(x) sapply(str_split(x$sortpubdate, " "), "[[", 1))
    results <- data.frame(id = ids, dates = dates, title = titles, count = search_results$count, stringsAsFactors = FALSE)
  }
  return(results)
}

safe_search_and_get_titles <- function(drug_name, disease) {
  attempt <- 1
  max_attempts <- 3  # You can adjust this as needed
  while (attempt <= max_attempts) {
    tryCatch({
      result <- search_and_get_titles(drug_name = drug_name, disease = disease)
      return(result)
    }, error = function(e) {
      cat("Error encountered for", drug_name, ": ", e$message, "\n")
      if (attempt < max_attempts) {
        cat("Retrying in 5 seconds...\n")
        Sys.sleep(5)
        attempt <- attempt + 1
      } else {
        cat("Failed after", max_attempts, "attempts.\n")
        return(NULL)  # Return NULL or any other appropriate value in case of failure
      }
    })
  }
}

# Get titles for each drug
result_pcs = list()
result_cs = list()
results = list()
k = 0
for(x in drug_info$synonym) {
  k = k+1
  print(paste("Doing molecule",k,"named",x))
  result_pc <- safe_search_and_get_titles(drug_name = x, disease = "prostate cancer")
  result_c <- safe_search_and_get_titles(drug_name = x, disease = "cancer")
  result <- safe_search_and_get_titles(drug_name = x, disease = "")
  add(result_pcs, result_pc)
  add(result_cs, result_c)
  add(results, result)
}

names(result_pcs) = drug_info$synonym
names(result_cs) = drug_info$synonym
names(results) = drug_info$synonym
results_pubmed = setNames(list(result_pcs,result_cs,results), c("prostate","cancer","global"))

saveRDS(results_pubmed, "results_pubmed.rds")
```


```{r get pubcmed verif}
results_pubmed = readRDS("results_pubmed.rds")

drug_info$ProstateCancer_title = sapply(results_pubmed$prostate[drug_info$synonym], function(x) x$id)
drug_info$Cancer_title = sapply(results_pubmed$cancer[drug_info$synonym], function(x) x$id)
drug_info$All_title = sapply(results_pubmed$global[drug_info$synonym], function(x) x$id)

drug_info$ProstateCancer_title_max = sapply(results_pubmed$prostate, function(x) ifelse(nbr(x$count) == 0, 0, x$count))
drug_info$Cancer_title_max = sapply(results_pubmed$cancer, function(x) ifelse(nbr(x$count) == 0, 0, x$count))
drug_info$All_title_max = sapply(results_pubmed$global, function(x) ifelse(nbr(x$count) == 0, 0, x$count))


# write_csv(drug_info, "drug_info.csv")
drug_info_read = read_csv("drug_info.csv")
drug_info = drug_info %>% filter(synonym %in% (drug_info_read %>% filter(is.na(donotkeep)) %>% pull(synonym)))

drug_info_3 = drug_info %>% 
  group_by(primary_key, name) %>% 
  summarise(ProstateCancer = max(ProstateCancer_title_max),
            Cancer = max(Cancer_title_max),
            Lone = max(All_title_max))


drug_classification = DB$drugs$drug_classification %>% filter(drugbank_id %in% drug_info_3$primary_key) %>% select(drugbank_id, direct_parent)
pharmacology = DB$drugs$pharmacology %>% filter(drugbank_id %in% drug_info_3$primary_key) %>% select(drugbank_id,indication, pharmacodynamics, mechanism_of_action)
targets = DB$cett$targets$general_information %>% filter(parent_key %in% drug_info_3$primary_key) %>% filter(organism == "Humans") %>% select(parent_key, name) %>% group_by(parent_key) %>% summarise(name = paste(name, collapse = " ; ")) %>% setNames(c("parent_key", "target"))

drug_info_4 = full_join(full_join(full_join(drug_info_3,drug_classification, by = c("primary_key"="drugbank_id")),
                    targets, by = c("primary_key"="parent_key")),
          pharmacology, by = c("primary_key"="drugbank_id"))


drug_info_4[drug_info_4$name == "Glycerol Tribenzoate", ][, c(3,4,5)] = 0
drug_info_4[drug_info_4$name == "Glycol salicylate", ][, c(3,4,5)] = 0
drug_info_4 %>% arrange(desc(ProstateCancer))

drug_info[drug_info$name == "Glycol salicylate", ]

write_csv(drug_info_4, "drug_info_4.csv")

drug_info_4 = read_csv("drug_info_4.csv")

drug_info_4 = drug_info_4 %>% separate_rows(target, sep = " ; ")

```

```{r get pubcmed verif}
fing0_df_long_prediction = fing0_df_long %>% filter(Drug %in% names(predicted_drugs)[predicted_drugs >= 1] & name %in% names(predicted_drugs)[predicted_drugs >= 1])
network_drug_community = fing0_df_long_prediction %>% filter(value >= 0.9) %>% filter(Drug != name) %>% graph_from_data_frame()
components <- components(network_drug_community)
community_list <- lapply(unique(components$membership), function(i) {
  V(network_drug_community)$name[components$membership == i]
})

predicted_drugs_group = c(community_list, as.list(setdiff(names(predicted_drugs)[predicted_drugs >= 1] , unlist(community_list))))

predicted_drugs_group

drug_info_2$Simi_group = sapply(drug_info_2$primary_key, function(i) which(sapply(predicted_drugs_group, function(x) i %in% x)))
drug_info_2 = drug_info_2 %>% group_by(Simi_group) %>% summarise(primary_key = paste(primary_key, collapse = "/"),
                                                                 name = paste(name, collapse = "/"),
                                                                 ProstateCancer_title = list(unique(unlist(ProstateCancer_title))),
                                                                 Cancer_title = list(unique(unlist(Cancer_title))))


drug_info_2$ProstateCancer = nbrs(drug_info_2$ProstateCancer_title)
drug_info_2$Cancer = nbrs(drug_info_2$Cancer_title)
# 
# drug_info_2 = drug_info_2 %>% select(name, ProstateCancer, Cancer, everything())
# # write_csv(drug_info_2, "drug_info_2.csv")

```


```{python}
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:22:15 2023

@author: Ellie
"""


from Bio import Entrez

def fetch_pubmed_papers(query, max_results=10):
    Entrez.email = "milanpicard2@gmail.com"  # Replace with your email address
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    papers = []
    for pmid in id_list:
        summary_handle = Entrez.esummary(db="pubmed", id=pmid)
        summary_record = Entrez.read(summary_handle, validate=False)
        paper = {
            "Title": summary_record[0]["Title"],
            "Year": summary_record[0]["PubDate"].split(" ")[0],
            "PMID": summary_record[0]["ArticleIds"]["pubmed"],
        }
        papers.append(paper)
    return papers

if __name__ == "__main__":
    # Initialize an empty list to store all papers
    all_papers = []
    max_results = 120  # You can modify this to get more or fewer results
    queries = '("prostate cancer" OR "prostate tumor") AND "MMP1"'
    # Loop through the list of queries
    for query in queries:
        papers = fetch_pubmed_papers(query, max_results)
        all_papers.append(papers)
        


```

```{r}
script_path <- "Prostatecancer_scrap2.py"
queries = paste0('("prostate cancer" OR "prostate tumor") AND "',pred_target_knn_gene,'"')

# Run the Python script and get the results as a JSON string
results_json <- py_run_string(paste0('import sys\nsys.path.append("', script_path, '")\nfrom Prostatecancer_scrap2 import fetch_pubmed_papers\nqueries = ', toJSON(queries), '\nresults = fetch_pubmed_papers(queries, max_results=200)\nresults'), convert = TRUE)

# Parse the JSON string to an R list
results <- jsonlite::fromJSON(results_json$results)
pred_target_knn_gene[sapply(results, nrow) == 150]
sapply(results, nrow)

```





```{r cluster enrichment}
Cluster_list = readRDS("../data/Cluster_list.rds")
BiocManager::install(c("clusterProfiler")) 
# Load the packages
library(clusterProfiler)
library(org.Hs.eg.db) 

# GO enrichment analysis
iugu = mapIds(org.Hs.eg.db, str_remove(Cluster_list[["Cluster_920"]], "Prot_"), "ENTREZID", "UNIPROT")

go_enrichment <- enrichGO(
  gene         = iugu,
  OrgDb        = org.Hs.eg.db,
  keyType      = "ENTREZID",
  ont          = "BP",        # BP for Biological Process, MF for Molecular Function, CC for Cellular Component
  pAdjustMethod = "BH",       # Benjamini-Hochberg adjustment for multiple testing
  pvalueCutoff = 0.05,
  qvalueCutoff = 0.2
)

# View results
print(go_enrichment)
go_enrichment@result
"DNA repair"
"JAK-STAT"
"hormonal/sexual characteristic"

```





```{r effect secondaire}
"not good, not enough drugs to manyb symptoms"
# meddra_name = data.frame(read_tsv("drug_names.tsv", col_names = F)$X2)
# write_csv(meddra_name, "meddra_name.csv", col_names = F)
# meddra_name_converted = read_tsv("meddra_name_converted.txt", col_names = c("drug_name","drugbank_id"))
meddra_syn = inner_join(meddra_name_converted,read_tsv("drug_names.tsv", col_names = F), by = c("drug_name" = "X2"))
meddra_syn = meddra_syn %>% na.omit()

drug_indication = inner_join(meddra_all_indications, meddra_syn) %>% select(X4,drugbank_id, drug_name) %>% unique
drug_indication

meddra_all_indications = read_tsv("meddra_all_indications.tsv", col_names = F)
meddra_all_se = read_tsv("meddra_all_se.tsv", col_names = F)

```



