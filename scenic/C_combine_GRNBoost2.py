#!/usr/bin/env python

from collections import defaultdict


input_files = [f"results/GRNBoost2/run{i+1}/adj.csv" for i in range(10)]
output_file = "results/GRNBoost2/combined/adj.csv"
count_frac = 0.5

input_len = len(input_files)

print(f"number of input files: {input_len}")

#####################
#  read from files  #
#####################

def dict_init():
    return {"count": 0, "importance": []}

edge_dict = defaultdict(dict_init)


for in_file in input_files:
    with open(in_file, "r") as f:
        next(f)
        for line in f:
            TF, TG, importance = line.strip().split(",")
            comb = (TF, TG)
            if comb:
                edge_dict[comb]["count"] += 1
                edge_dict[comb]["importance"].append(float(importance))

#################################
#  filter (TF,TG) combinations  #
#################################

edge_dict = {
    k: v
    for k, v in edge_dict.items()
    if v["count"]/input_len >= count_frac
}

########################
#  average importance  #
########################

for _, v in edge_dict.items():
    # average importance across all identical (TF, TG) combinations
    v["importance"] = sum(v["importance"])/len(v["importance"])


###############################
#  sort (TF,TG) combinations  #
###############################

edge_list_filtered = list(edge_dict.items())

edge_list_filtered.sort(
    key = lambda x: x[1]["importance"],
    reverse = True,
)

###################
#  write to file  #
###################

with open(output_file, "w") as f:
    f.write("TF,target,importance\n")
    for k, v in edge_list_filtered:
        TF, TG, importance = k[0], k[1], v["importance"]
        line = f"{TF},{TG},{importance}\n"
        f.write(line)
