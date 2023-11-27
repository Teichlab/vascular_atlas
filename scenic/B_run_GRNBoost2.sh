#!/bin/bash

f_loom_path_scenic="outputs/prepare_scenic/filtered_scenic.loom"
f_tfs="TF_list.txt"
cores="32"
outfile="outputs/GRNBoost2/run$1/adj.csv"

set -x


python $(which arboreto_with_multiprocessing.py) \
	${f_loom_path_scenic} \
	${f_tfs} \
	--method grnboost2 \
	--output "$outfile" \
	--num_workers $((cores-5))

