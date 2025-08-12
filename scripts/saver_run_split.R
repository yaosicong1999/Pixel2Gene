library(SAVER)
library(arrow)
library(Matrix)
library(argparse)

parser <- ArgumentParser(description = "Example argparse in R")
parser$add_argument("--pref", type = "character", help = "prefix of the input files")
parser$add_argument("--split_idx", type = "integer", help = "index of the split for which to run SAVER")
args <- parser$parse_args()
split_idx <- args$split_idx
pref <- args$pref

cnts_split = readRDS(paste0(pref, "saver-split", split_idx, "-cnts.rds"))
zero_row_indices <- which(rowSums(cnts_split) == 0)
if (length(zero_row_indices) == 0) {
  print("the split data was filtered successully")
} else {
  print("the split data was not filtered successfully")
}

# Ensure parquet_data is converted to a matrix before using saver
cnts_split = t(cnts_split)
saver_object <- saver(cnts_split, ncores = 16, estimates.only = TRUE)
saver_object = t(saver_object)
write_parquet(as.data.frame(as.matrix(saver_object)), paste0(pref, "saver-split", split_idx, "-imputed.parquet"))