#Code used for clinical trial data extraction and labeling


# Dowloaded from https://aact.ctti-clinicaltrials.org/download
# Pipe-delimited Files
# Current Month's Daily Static Copies

# <!-- 20240405_export.zip -->


library(tidyverse)

interventions = read.delim("data/AACT_ClinicalTrials_20240425/interventions.txt", sep = "|")
browse_conditions = read.delim("data/AACT_ClinicalTrials_20240425/browse_conditions.txt", sep = "|")
conditions = read.delim("data/AACT_ClinicalTrials_20240425/conditions.txt", sep = "|")
studies = read.delim("data/AACT_ClinicalTrials_20240425/studies.txt", sep = "|")
outcomes = read.delim("data/AACT_ClinicalTrials_20240425/outcomes.txt", sep = "|")
outcome_analyses = read.delim("data/AACT_ClinicalTrials_20240425/outcome_analyses.txt", sep = "|")
outcome_analysis_groups = read.delim("data/AACT_ClinicalTrials_20240425/outcome_analysis_groups.txt", sep = "|")
outcome_measurements = read.delim("data/AACT_ClinicalTrials_20240425/outcome_measurements.txt", sep = "|")
result_groups = read.delim("data/AACT_ClinicalTrials_20240425/result_groups.txt", sep = "|")
design_groups = read.delim("data/AACT_ClinicalTrials_20240425/design_groups.txt", sep = "|")
brief_summaries = read.delim("data/AACT_ClinicalTrials_20240425/brief_summaries.txt", sep = "|")
baseline_counts = read.delim("data/AACT_ClinicalTrials_20240425/baseline_counts.txt", sep = "|")




CT_4_gpt = function(ID){
  Clinical_trial_text = c()
  
  Study_info = studies %>% filter(nct_id == ID) %>% .[c("brief_title","official_title","baseline_population","overall_status","phase","enrollment","number_of_arms","why_stopped")] %>% setNames(c("Brief title", "Official title", "Baseline population", "Status", "Clinical phase", "Enrollment", "Number of arms", "Why stopped")) %>% unlist
  Condition = conditions %>% filter(nct_id == ID) %>% .[c("name")] %>% setNames("Pathology ") %>% unlist
  Conditions = paste0(Condition, collapse = ", ")
  Intervention = interventions %>% filter(nct_id == ID) %>% .[c("name","description")] %>% setNames(c("Drug tested ", "Drug description ")) %>% unlist
  
  Clinical_trial_text = c(paste(names(Study_info), Study_info, sep = ": "),
                          paste("Conditions", Conditions, sep = ": "),
                          paste(names(Intervention), Intervention, sep = ": "),
                          paste0("Clinical Trial Description: ", brief_summaries %>% filter(nct_id == ID) %>% pull(description), "\n"))
  
  Outcome = outcomes %>% filter(nct_id == ID) %>% .[c("id","title","description","units")] %>% setNames(c("id","Outcome type","Outcome description","Outcome Units"))
  if(nrow(Outcome) == 0){
    Clinical_trial_text = c(Clinical_trial_text, "Outcome: No outcomes or results were given, further analysis cannot be done.")
    Clinical_trial_text %>% unlist %>% cat
    writeLines(as.character(unlist(Clinical_trial_text)), paste0("Trial_",ID,".txt"))
    return(NULL)
  }
  Design_Groups = design_groups %>% filter(nct_id == ID)
  for(dg_i in 1:nrow(Design_Groups)){
    Result_design_groups = Design_Groups %>% .[dg_i, c("group_type", "title", "description")] %>% unlist
    Clinical_trial_text = c(Clinical_trial_text, 
                            paste0("Treatment group ",dg_i,": ", '""', paste(names(Result_design_groups), Result_design_groups, sep = ": ", collapse = ", "), '""'))
  }
  Clinical_trial_text = c(Clinical_trial_text, "\nOutomes results")
  for(Outcome_i in 1:length(Outcome$id)){
    Outcome_Id = Outcome$id[Outcome_i]
    OutCome_Measures = outcome_measurements %>% filter(nct_id == ID, outcome_id == Outcome_Id)
    Clinical_trial_text = c(Clinical_trial_text, paste("\nResult ", Outcome_i))
    Result_outcome = Outcome %>% filter(id == Outcome_Id) %>% .[c("Outcome type","Outcome description","Outcome Units")] %>% unlist
    Result_outcome = paste(names(Result_outcome), Result_outcome, sep = ": ")
    
    # Get pvalue and other outcome analysis
    Results_outcome_analysis = outcome_analyses %>% filter(nct_id == ID, outcome_id == Outcome_Id) %>% .[c("non_inferiority_type", "param_type","param_value","p_value","ci_n_sides")] %>% setNames(c("Test type","Test parameter type", "Parameter value", "P-value", "Ci n sides")) %>% unlist
    outcome_analyses %>% filter(nct_id == ID, outcome_id == Outcome_Id)
    
    Clinical_trial_text = c(Clinical_trial_text, 
                            Result_outcome,
                            paste0("Analysis of the outcome...\n", paste(names(Results_outcome_analysis), Results_outcome_analysis, sep = ": ", collapse = ", ")))
    for(outcome_row in 1:nrow(OutCome_Measures)){
      OutCome_Values =  OutCome_Measures[outcome_row, c("result_group_id","param_type","param_value")] %>% unlist
      OutCome_Values = c(treatment = result_groups %>% filter(id == OutCome_Values["result_group_id"]) %>% pull(title), OutCome_Values[-1])
      Results_outCome_Values = paste(names(OutCome_Values), OutCome_Values, sep = ": ")
      Results_outCome_Values = paste0(Results_outCome_Values, collapse = ", ")
      Clinical_trial_text = c(Clinical_trial_text, Results_outCome_Values)
    }
  }
  Clinical_trial_text %>% unlist %>% cat
  writeLines(as.character(unlist(Clinical_trial_text)), paste0("Trial_",ID,".txt"))
}

read_GPT_answer = function(file){
  answer = read.delim(file, header = FALSE) %>% `colnames<-`("Result")
  answer = answer %>% 
    mutate(Group = cumsum(str_detect(answer$Result, "Trial_"))) %>% 
    {as.data.frame(t(as.data.frame({split(.$Result, .$Group)})))} %>% 
    mutate(nct_id = str_remove(str_remove(V1, "Trial_"), ".txt"),
           Fit_guideline = ifelse(str_detect(V2, "Yes"), TRUE, FALSE),
           Drug = str_remove(V3, "Drug: "),
           Result = trimws(str_remove(V4, "Result: ")),
           Explanation = str_remove(V5, "Explanation: ")) %>% select(-c(V1, V2, V3, V4,V5)) %>% 
    `rownames<-`(NULL)
}

split_and_extract <- function(string, pattern, position) {
  parts <- str_split(string, pattern)
  sapply(parts, function(x) x[[position]])
}


df_studies = studies[c("nct_id","phase","study_first_submitted_date","study_type","overall_status","why_stopped")]
df_studies = df_studies %>% 
  mutate(phase = ifelse(phase %in% c("Early Phase 1","Phase 1","Phase 1/Phase 2"), 1, 
                        ifelse(phase %in% c("Phase 2/Phase 3","Phase 2"), 2,
                               ifelse(phase %in% c("Phase 3"), 3, 
                                      ifelse(phase %in% c("Phase 4"), 4, 0))))) %>% 
  mutate(date = as.numeric(split_and_extract(study_first_submitted_date, "-", 1)), study_first_submitted_date = NULL) %>% 
  mutate(intervention = ifelse(study_type == "Interventional", TRUE, FALSE), study_type = NULL) %>% 
  mutate(completed = ifelse(overall_status == "Completed", TRUE, FALSE)) %>% 
  select(nct_id, phase, date, intervention, completed, everything())

# tablesort(str_subset(conditions$downcase_name, "prostat"))
PC_keyword = c("prostate cancer",
               "prostatic neoplasms",
               "prostate adenocarcinoma",
               "metastatic castration-resistant prostate cancer",
               "adenocarcinoma of the prostate",
               "metastatic prostate cancer",
               "recurrent prostate cancer",
               "prostate cancer metastatic",
               "castration-resistant prostate cancer",
               "recurrent prostate carcinoma",
               "stage ivb prostate cancer ajcc v8",
               "stage iii prostate cancer",
               "stage iii prostate cancer ajcc v8",
               "cancer of prostate",
               "prostatic cancer",
               "hormone refractory prostate cancer",
               "stage iiia prostate cancer ajcc v8",
               "metastatic prostate adenocarcinoma",
               "metastatic castration-resistant prostate cancer (mcrpc)",
               "hormone-refractory prostate cancer",
               "adenocarcinoma of the prostate",
               "stage iv prostate cancer",
               "prostate carcinoma",
               "metastatic prostate carcinoma",
               "prostatic neoplasm",
               "stage iv prostate cancer ajcc v8",
               "hormone-resistant prostate cancer",
               "cancer of the prostate",
               "prostate neoplasm",
               "prostate neoplasms",
               "metastatic castration resistant prostate cancer",
               "prostate cancer recurrent",
               "prostatic neoplasms, castration-resistant",
               "stage iiic prostate cancer ajcc v8",
               "stage iiib prostate cancer ajcc v8",
               "advanced prostate cancer",
               "castrate resistant prostate cancer")

CT_PC = conditions %>% filter(downcase_name %in% PC_keyword) %>% pull(nct_id) %>% unique
CT_PC %>% nbr # [1] 5495

DF_result = data.frame(nct_id = CT_PC)
DF_result = merge(DF_result, df_studies)
DF_result = merge(DF_result, interventions)

DF_result = DF_result %>% 
  filter(intervention & intervention_type == "Drug") %>% 
  select(nct_id,phase,date,intervention,completed,name,overall_status,why_stopped,intervention_type,description)

NCT_ID_to_analyse = DF_result %>% arrange(desc(date)) %>% filter(completed | nct_id %in% outcomes$nct_id) %>% pull(nct_id) %>% unique
NCT_ID_to_analyse = paste0("Trial_",NCT_ID_to_analyse,".txt")
NCT_ID_to_analyse %>% nbr # 1442
# writeLines(NCT_ID_to_analyse, "NCT_ID_to_analyse.txt")


GPT_answers = read_GPT_answer("data/GPT_answers_full.txt")
GPT_answers = GPT_answers %>% 
  mutate(phase = df_studies$phase[match(nct_id, df_studies$nct_id)], .after = 1) %>% 
  mutate(status = df_studies$overall_status[match(nct_id, df_studies$nct_id)], .after = 5) %>% 
  mutate(why_stopped = df_studies$why_stopped[match(nct_id, df_studies$nct_id)], .after = 6)



