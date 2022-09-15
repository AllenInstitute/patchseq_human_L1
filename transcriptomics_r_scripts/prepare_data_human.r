
name = 'human'
ps_dataset = read.csv("/home/tom.chartrand/projects/human_l1/human_l1_dataset_2022_05_03.csv") %>%
              filter(l1_type=="True")

l1_types = c('Inh L1-2 PAX6 CDH12', 'Inh L1-2 PAX6 TNFAIP8L3',
       'Inh L1 LAMP5 NMBR', 'Inh L1-4 LAMP5 LCP2',
       'Inh L1-2 LAMP5 DBP', 'Inh L1 SST CHRNA4',
       'Inh L1-2 ADARB2 MC4R', 'Inh L1-2 SST BAGE2',
       'Inh L1-2 VIP TSPAN12', 'Inh L1-2 VIP PCDH20')
  

## Folder locations
facsFolder   <- "//allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/facs_seq/MTG_paper_rev/"
psFolder = "//allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/patch_seq/star/human/human_patchseq_MTG_current/"

source("prepare_data.r")


