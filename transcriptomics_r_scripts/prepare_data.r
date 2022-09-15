## Load libraries

library(feather)
library(dplyr)
library(future)
library(scrattch.hicat) # get_cl_medians etc

options(stringsAsFactors=FALSE)

## Prepare for extra memory usage
plan("multicore", workers = 8)
options(future.globals.maxSize = 10000 * 1024^2)

data_folder = file.path('', 'home','tom.chartrand','projects','human_l1')

# patch-seq data

annoPS   <- read_feather(paste(psFolder,"anno.feather",sep="")) 
Expr.dat  <- feather(paste(psFolder,"data.feather",sep=""))   # FPKM
annoPS   <- annoPS[match(Expr.dat$sample_id,annoPS$sample_id),] 
datPS    <- as.matrix(Expr.dat[,names(Expr.dat)!="sample_id"])
rownames(datPS) <- annoPS$sample_id
datPS    <- t(datPS)
# datPS    <- log2(datPS+1)  
gc()

ps_filter = eval(expr(spec_id_label %in% ps_dataset$spec_id), annoPS)
annoPS = annoPS[ps_filter,]
datPS = datPS[,ps_filter]

save(datPS, annoPS, file=file.path(data_folder, paste0("ps_", name, ".RData")))

# FACS data

annoFACS   <- read_feather(paste(facsFolder,"anno.feather",sep=""))
Expr.dat  <- feather(paste(facsFolder,"data.feather",sep=""))
annoFACS   <- annoFACS[match(Expr.dat$sample_id,annoFACS$sample_id),]
datFACS    <- as.matrix(Expr.dat[,names(Expr.dat)!="sample_id"])
rownames(datFACS) <- annoFACS$sample_id
datFACS    <- t(datFACS)
# datFACS    <- log2(datFACS+1)

datFACS_all  <- datFACS
annoFACS_all <- annoFACS
gc()
# save(datFACS_all, annoFACS_all, file=file.path(data_folder, paste0("complete_facs_", name, ".RData")))

# define good gene set
# library(VENcelltypes) # mito_genes and sex_genes
load("mito_genes.rda")
load("sex_genes.rda")
isExclude <- sort(unique(c(sex_genes,mito_genes)))  
excludeGn <- is.element(rownames(datPS),isExclude)

# Second find glial genes and exclude these from consideration.  
clFACS   = factor(annoFACS_all$cluster_label)
names(clFACS) <- colnames(datFACS_all)
medians = get_cl_medians(datFACS_all,clFACS)
isUsed  = unique(annoFACS_all$cluster_label)

if (name=="human") {
  isGlia  = isUsed[!(grepl("Exc",isUsed)|grepl("Inh",isUsed))]
  isUsed  = setdiff(isUsed,isGlia)
} else {
  isUsed  = isUsed[!(grepl("Low",isUsed)|grepl("Doublet",isUsed)|grepl("Batch",isUsed)|grepl("High",isUsed))]
  isGlia  = isUsed[grepl("Astro ",isUsed)|grepl("Endo ",isUsed)|grepl("Microglia ",isUsed)|
                     grepl("Oligo ",isUsed)|grepl("OPC ",isUsed)|grepl("Oligo ",isUsed)|
                     grepl("PVM ",isUsed)|grepl("SMC ",isUsed)|grepl("VLMC ",isUsed)]
  isUsed  = setdiff(isUsed,c(isGlia,"CR Lhx5","Meis2 Adamts19"))
}

maxUsed = apply(medians[,isUsed],1,max)
maxGlia = apply(medians[,isGlia],1,max)
glialGn = maxGlia>maxUsed     # Exclude genes with maximal median expression in glia

good_genes = rownames(datPS)[!(excludeGn|glialGn)]

l1_facs = eval(expr(cluster_label %in% l1_types), annoFACS_all)
annoFACS = annoFACS_all[l1_facs,]
datFACS = datFACS_all[,l1_facs]
save(datFACS, annoFACS, good_genes, file=file.path(data_folder, paste0("facs_", name, ".RData")))
# save(datFACS, annoFACS, file=file.path(data_folder, paste0("facs_", name, ".RData")))

