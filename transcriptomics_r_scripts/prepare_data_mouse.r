name = "mouse"
l1_types = c('Lamp5 Krt73', 'Lamp5 Plch2 Dock5', 'Lamp5 Lsp1', 'Lamp5 Fam19a1 Pax6',
             'Lamp5 Ntn1 Npy2r', 'Lamp5 Fam19a1 Tmem182', 'Sncg Vip Nptx2',
             'Vip Col15a1 Pde1a')
ps_dataset = read.csv("/home/tom.chartrand/projects/human_l1/mouse_l1_dataset_2022_05_03.csv") %>% 
  filter(l1_type=="True")

## Folder locations
facsFolder = "//allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/facs_seq/Mm_VISp_AIT2.1_20047_20200224/"
psFolder = "//allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/patch_seq/star/mouse_patchseq_VISp_current/"


source("prepare_data.r")