#!/bin/bash

reg_input="outputs/cisTarget/reg.csv"
f_loom_path_scenic="outputs/prepare_scenic/filtered_scenic.loom"
cores="32"
outfile="outputs/AUCell/aucell_scenic.loom"

set -x


pyscenic aucell \
	"${f_loom_path_scenic}" \
	"${reg_input}" \
	--output "${outfile}" \
	--num_workers $((cores-5))
