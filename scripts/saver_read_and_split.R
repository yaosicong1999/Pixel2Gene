library(arrow)
library(Matrix)
library(argparse)

parser <- ArgumentParser(description = "Example argparse in R")
parser$add_argument("--pref", type = "character", help = "prefix of the input files")
parser$add_argument("--num", type = "integer", default = 4, help = "number of splits")
args <- parser$parse_args()
n_split <- args$num
pref <- args$pref

parquet_data <- read_parquet(paste0(pref, "cnts.parquet"))
parquet_data <- parquet_data[, !colnames(parquet_data) %in% "__index_level_0__"]

gene_names <- readLines(paste0(pref, "gene-names.txt"))
matching_indices <- match(gene_names, colnames(parquet_data))
parquet_data <- parquet_data[, matching_indices, drop = FALSE]

# Convert to sparse matrix
parquet_data_sparse <- as(as.matrix(parquet_data), "dgCMatrix")
# Get the row indices where the row sum is 0
zero_row_indices <- which(rowSums(parquet_data_sparse) == 0)
parquet_data_sparse <- parquet_data_sparse[-zero_row_indices, ]

locs <- read.csv(paste0(pref, "locs.tsv"), sep = "\t", header = TRUE, row.names = 1)
locs <- locs[-zero_row_indices, ]
write.table(locs, file = paste0(pref, "saver-raw-locs.tsv"), sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)

# Randomly split into n equal sets and store the splits separately
set.seed(123) # For reproducibility
num_rows <- nrow(parquet_data_sparse)
indices <- sample(1:num_rows) # Shuffle row indices
split_size <- floor(num_rows / n_split)

# Create a list of n splits for cnts
cnts_split_list = list()
for (i in 1:n_split) {
  cnts_split_list[[i]] <- parquet_data_sparse[indices[((i - 1) * split_size + 1):(i * split_size)], ]
  saveRDS(cnts_split_list[[i]], file = paste0(pref, "saver-split", i, "-cnts.rds"))
}

# Create a list of n splits for locs
locs_split_list = list()
for (i in 1:n_split) {
  locs_split_list[[i]] <- locs[indices[((i - 1) * split_size + 1):(i * split_size)], ]
  write.table(locs_split_list[[i]], file = paste0(pref, "saver-split", i, "-locs.tsv"), sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
}