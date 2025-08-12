#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/
device="cuda"  # "cuda" or "cpu"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_WORKSPACE_LIMIT_IN_MB=4096

python harmony.py --embs ../xenium/CRC-P1-embeddings-hipt-raw.pickle ../xenium/CRC-P2-embeddings-hipt-raw.pickle ../xenium/CRC-P5-embeddings-hipt-raw.pickle --output ../xenium/CRC-P1-embeddings-hipt-raw-har.pickle ../xenium/CRC-P2-embeddings-hipt-raw-har.pickle ../xenium/CRC-P5-embeddings-hipt-raw-har.pickle 

python harmony.py --embs ../xenium/CRC-P1-embeddings-hipt-smooth.pickle ../xenium/CRC-P2-embeddings-hipt-smooth.pickle ../xenium/CRC-P5-embeddings-hipt-smooth.pickle --output ../xenium/CRC-P1-embeddings-hipt-smooth-har.pickle ../xenium/CRC-P2-embeddings-hipt-smooth-har.pickle ../xenium/CRC-P5-embeddings-hipt-smooth-har.pickle 
