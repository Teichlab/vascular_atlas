#!/bin/bash

adj_input="outputs/find_modules/adj_mod.dat"
f_db_names=$2
f_motif_path="$3"
cores="32"
outfile="outputs/cisTarget/reg.csv"

set -x


pyscenic ctx \
	"${adj_input}" \
	${f_db_names} \
	--annotations_fname "${f_motif_path}" \
	--output "$outfile" \
	--mask_dropouts \
	--num_workers $((cores-5)) \

