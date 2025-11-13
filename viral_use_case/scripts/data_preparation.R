# The filepaths in this script likely will not work within a container environment 
# and thus should be re-specified if ran in such an environment.

options(repos = "https://cloud.r-project.org/")
if(!require("mvdalab", character.only = TRUE)) {
    install.packages("mvdalab")
}
library("mvdalab")

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
options("repos" = BiocManager::repositories())
if(!require("pmartR", character.only = TRUE)) {
    install.packages("pmartR")
}
library("pmartR")
library(magrittr)
library(here)

i_am("scripts/data_preparation.R")


# The intention of this script is to prepare the datasets that will be used as the basis for the viral use-case experiments. 
# "Preparing the datasets" in this context refers to collecting the data that will be used as a base "source" dataset, collecting
# the data that will be used as a base "target" dataset, imputing missing values in each of these data, partitioning the target
# dataset into a transfer and validation subset (where the validation subset is fixed), using the unpartitioned target dataset to
# perform differentialy expression analyses to determine statistically differential proteins.

# Note that these prepared data will be used in two ways. First, they will be used in an "experimental sense" where we consider
# various changes to the source and target datasets in such a way that mimics our simulation experiments. Specifically: 
# 1. We will consider various sizes of the base "source" dataset to use for initial training
# 2. We will consider various sizes of the transer partition for the target dataset to use for transer modeling
# 3. As a proxy for SNR, we will consider various ratios of differentially expressed to non-differentially expressed features in 
#    the source/target-transfer datasets.
# 4. Experiments that vary the above three conditions will be done on datasets compiled using multiple timepoints, but we will
#    perform similar experiments on datasets compiled using only one timepoint (in case the modeling is substantially affected
#    by our ignoring timepoint in modeling efforts)
# The second way that the data will be used is to generate lists of important proteins for Amy to look at. We would generate at
# least two lists, where the first is based off of the best performing transfer model (as assessed through our experiments) and 
# the second is based off of training a model to the target dataset alone. We will then need to compare these lists (i.e. point
# out what they identify as important in common and where they differ) and then share that information with Amy somehow so that
# she is able to gauge whether there was anything biologically interesting among the differences that the transfer learning approach
# had identified. 


### Import/process all datasets -----------------------------------------------------------

# data_files = list.files("/workspaces/timed-hpc/timeddata/data/", full.names = TRUE)
data_files = list.files("C:/Users/flor829/OneDrive - PNNL/Projects/PPI_NV1827/current_codebase/timed-hpc/timeddata/data/", full.names = TRUE)

# Target Data Files (We will use the unicorn data)
# We will also filter down to only 12hr, 24hr, 36hr, and 48hr since those
# timepoints are the only ones that overlap with what we plan to use for the
# source data. 
unicorn_files <- data_files[grepl("unicorn", data_files)]
unicorn_files <- sapply(c("12hr", "24hr", "36hr", "48hr"), function(x){
  unicorn_files[grepl(x, unicorn_files)]
})

# This loop iteratively loads and initially processes each corresponding
# dataset.
unicorn_datasets <- vector("list", length = length(unicorn_files))
kept_prot_names_unicorn <- vector("list", length = length(unicorn_files))
for(i in 1:length(unicorn_files)){
  loop_filename <- unicorn_files[i]
  load(loop_filename)

  datname <- as.character(gsub(".*?/data/", "", loop_filename))
  datname <- gsub(".rda", "", datname)

  tempdat <- get(datname)

  # Data already in pmartR format. 
  class(tempdat)
  
  # Remove peptides with >35% missingness. 
  missing_percs_feat <- apply(tempdat$e_data[,-1], 1, function(x){sum(is.na(x))/length(x)*100})
  missing_feat_rm <- tempdat$e_data$Peptide[which(missing_percs_feat > 35)]
  myfilt <- custom_filter(omicsData = tempdat, e_data_remove = missing_feat_rm)
  tempdat <- applyFilt(myfilt, omicsData = tempdat)
  
  # Impute the remaining data
  imputed_data <- imputeEM(data = tempdat$e_data[,-1], impute.ncomps = 3)$Imputed.DataFrames[[1]]
  
  tempdat$e_data[,-1] <- imputed_data

  # Normalize the data
  tempdat <- normalize_global(
    omicsData = tempdat,
    subset_fn = "all",
    norm_fn = "median",
    apply_norm = TRUE,
    backtransform = FALSE
  )

  # Roll-up to the protein level for analysis
  tempdat_prot <- protein_quant(pepData = tempdat, 
                                method = "rrollup",
                                combine_fn = "median",
                                parallel = FALSE)
  
  # only names of proteins that were kept (assuming some proteins are dropped
  # by virtue of earlier removal of peptides with missingness > 35%)
  kept_prot_names_unicorn[[i]] <- c(tempdat_prot$e_data$Protein)

  # transpose of e_data in preparation for concatenating with other data
  unicorn_datasets[[i]] <- tempdat_prot$e_data %>%
  t(.) %>%
  as.data.frame(.) %>%
  setNames(.[1,]) %>%
  dplyr::slice(-1) %>%
  dplyr::mutate(dplyr::across(dplyr::everything(), as.numeric)) %>%
  dplyr::mutate(SampleID = rownames(.)) %>%
  # merge in response data
  dplyr::left_join(tempdat_prot$f_data %>% 
                    dplyr::select(SampleID, Strain), 
                    by = "SampleID") %>%
  dplyr::mutate(Resp = dplyr::case_when(
    Strain == "MCK" ~ "mock",
    Strain != "MCK" & !is.na(Strain) & Strain != "" ~ "viral",
    TRUE ~ NA
  )) %>%
  dplyr::relocate(SampleID, Strain, Resp) 
  
  rm(list = datname)
  rm(loop_filename, datname, tempdat, missing_percs_feat, 
     missing_feat_rm, myfilt, imputed_data, tempdat_prot)
}

common_prots_unicorn <- Reduce(intersect, kept_prot_names_unicorn)

# Source Data Files (We will use the omics LHV data)
# We will also filter down to only 12hr, 24hr, 36hr, and 48hr since those
# timepoints are the only ones that overlap with what we plan to use for the
# target data. 
lhv_files <- sapply(c("mfb", "mhae", "mmve"), function(x){
  data_files[grepl(x, data_files)]
})
lhv_files <- sapply(c("12hr", "24hr", "36hr", "48hr"), function(x){
  lhv_files[grepl(x, lhv_files)]
})

# This loop iteratively loads and initially processes each corresponding
# dataset.
lhv_datasets <- vector("list", length = length(lhv_files))
kept_prot_names_lhv <- vector("list", length = length(lhv_files))
for(i in 1:length(lhv_files)){
  loop_filename <- lhv_files[i]
  load(loop_filename)
  
  datname <- as.character(gsub(".*?/data/", "", loop_filename))
  datname <- gsub(".rda", "", datname)
  
  tempdat <- get(datname)
  
  # Data already in pmartR format. 
  class(tempdat)
  
  # Remove peptides with >35% missingness. 
  missing_percs_feat <- apply(tempdat$e_data[,-1], 1, function(x){sum(is.na(x))/length(x)*100})
  missing_feat_rm <- tempdat$e_data$Peptide[which(missing_percs_feat > 35)]
  myfilt <- custom_filter(omicsData = tempdat, e_data_remove = missing_feat_rm)
  tempdat <- applyFilt(myfilt, omicsData = tempdat)
  
  # Impute the remaining data
  imputed_data <- imputeEM(data = tempdat$e_data[,-1], impute.ncomps = 3)$Imputed.DataFrames[[1]]
  
  tempdat$e_data[,-1] <- imputed_data
  
  # Normalize the data
  tempdat <- normalize_global(
    omicsData = tempdat,
    subset_fn = "all",
    norm_fn = "median",
    apply_norm = TRUE,
    backtransform = FALSE
  )
  
  # Roll-up to the protein level for analysis
  tempdat_prot <- protein_quant(pepData = tempdat, 
                                method = "rrollup",
                                combine_fn = "median",
                                parallel = FALSE)
  
  # only names of proteins that were kept (assuming some proteins are dropped
  # by virtue of earlier removal of peptides with missingness > 35%)
  kept_prot_names_lhv[[i]] <- c(tempdat_prot$e_data$Reference_Protein)
  
  # transpose of e_data in preparation for concatenating with other data
  lhv_datasets[[i]] <- tempdat_prot$e_data %>%
    t(.) %>%
    as.data.frame(.) %>%
    setNames(.[1,]) %>%
    dplyr::slice(-1) %>%
    dplyr::mutate(dplyr::across(dplyr::everything(), as.numeric)) %>%
    dplyr::mutate(SampleID = rownames(.)) %>%
    # merge in response data
    dplyr::left_join(tempdat_prot$f_data %>% 
                       dplyr::select(SampleID, Virus), 
                     by = "SampleID") %>%
    dplyr::mutate(Resp = dplyr::case_when(
      Virus == "Mock" ~ "mock",
      Virus != "Mock" & !is.na(Virus) & Virus != "" ~ "viral",
      TRUE ~ NA
    )) %>%
    dplyr::relocate(SampleID, Virus, Resp) 
  
  rm(list = datname)
  rm(loop_filename, datname, tempdat, missing_percs_feat, 
     missing_feat_rm, myfilt, imputed_data, tempdat_prot)
}

common_prots_lhv <- Reduce(intersect, kept_prot_names_lhv)


# Find common set of proteins across lhv and unicorn datasets. Requires
# reformatting of unicorn protein names. 
common_prots_unicorn_fmt <- gsub("^[^|]*\\|[^|]*\\|","",common_prots_unicorn)

common_prots_lhvunicorn <- intersect(common_prots_unicorn_fmt, common_prots_lhv)

# For each dataset, downselect to common proteins and only retain common columns
# (i.e. drop the Strain and Virus columns, depending on whether unicorn or lhv, and
#  retain only the Resp column, which has been manually verified as accurately reflecting
#  virus vs mock)
for(i in 1:length(unicorn_datasets)){
  unicorn_datasets[[i]] <- unicorn_datasets[[i]] %>%
    setNames(gsub("^[^|]*\\|[^|]*\\|","",colnames(.))) %>%
    dplyr::select(SampleID, Resp, dplyr::all_of(common_prots_lhvunicorn))
}

for(i in 1:length(lhv_datasets)){
  lhv_datasets[[i]] <- lhv_datasets[[i]] %>%
    dplyr::select(SampleID, Resp, dplyr::all_of(common_prots_lhvunicorn))
}


### Create Source and Target Datasets ---------------------------------------

source_dset <- Reduce(rbind.data.frame, lhv_datasets)
target_dset <- Reduce(rbind.data.frame, unicorn_datasets)


# The transfer set will itself be subsetted/reduced in sample size
# depending on the experiment. However this will serve as the "base"
# version of the transfer set. The "base" version will have as many
# samples as 80% of the samples in target_dset.
set.seed(1)
transfer_idxs <- sample(1:nrow(target_dset), floor(nrow(target_dset)*0.80))
target_transfer <- target_dset[transfer_idxs,]
target_validation <- target_dset[-transfer_idxs,]

write.csv(source_dset, here("data", "source_dset.csv"), row.names = FALSE)
write.csv(target_dset, here("data", "target_dset.csv"), row.names = FALSE)
write.csv(target_transfer, here("data", "target_transfer.csv"), row.names = FALSE)
write.csv(target_validation, here("data", "target_validation.csv"), row.names = FALSE)



### Find Differentially Expressed Proteins -------------------------------------------------------------------------

# We use the entire target dataset to determine which proteins are differentially
# expressed. The target dataset is used as opposed to the source because our goal
# is to use these differentially expressed proteins (or at least the proportion of
# their presence relative to all other features) as a proxy for SNR of the "signal"
# of interest, i.e. the signal in our target dataset. That being said, in order to
# consider whether there is any correspondence between the differentially expressed
# proteins in the target dataset and source dataset, we also separately determine 
# for the source dataset which proteins are differentially expressed. 

de_prots <- data.frame(Protein = names(target_dset)[-c(1,2)],
                       tstat_source = NA,
                       tstat_target = NA,
                       pval_source = NA,
                       pval_target = NA)
for(i in 1:nrow(de_prots)){
  temp_prot <- de_prots$Protein[i]
  
  temp_target <- target_dset %>%
    dplyr::select(Resp, dplyr::all_of(temp_prot)) %>%
    dplyr::mutate(Resp = factor(Resp, levels = c("viral", "mock"))) %>%
    setNames(c("Resp", "protein"))
  
  res <- t.test(protein ~ Resp, data = temp_target)
  de_prots$tstat_target[i] <- res$statistic
  de_prots$pval_target[i] <- res$p.value
  
  temp_source <- source_dset %>%
    dplyr::select(Resp, dplyr::all_of(temp_prot)) %>%
    dplyr::mutate(Resp = factor(Resp, levels = c("viral", "mock"))) %>%
    setNames(c("Resp", "protein"))
  
  res <- t.test(protein ~ Resp, data = temp_source)
  de_prots$tstat_source[i] <- res$statistic
  de_prots$pval_source[i] <- res$p.value
}

de_prots <- de_prots %>%
  dplyr::mutate(flag_source = dplyr::case_when(
    pval_source > 0.05 ~ 0,
    tstat_source >= 0 & pval_source <= 0.05 ~ 1,
    tstat_source < 0 & pval_source <= 0.05 ~ -1,
    TRUE ~ NA
  )) %>%
  dplyr::mutate(flag_target = dplyr::case_when(
    pval_target > 0.05 ~ 0,
    tstat_target >= 0 & pval_target <= 0.05 ~ 1,
    tstat_target < 0 & pval_target <= 0.05 ~ -1,
    TRUE ~ NA
  )) %>%
  dplyr::mutate(source_target_agreement = dplyr::case_when(
    flag_source == flag_target ~ 1, 
    TRUE ~ 0
  ))

# Not a strong agreement in terms of what is significant and the direction
# of the associated fold change. But more agreement for what is significant
# without consideration of the fold change.

# Save the table for later reference. 
write.csv(de_prots, here("data", "de_prots.csv"), row.names = FALSE)

sum(abs(de_prots$flag_target) == 1)

# Total of 68 differentially expressed proteins as per the target dataset alone. 
