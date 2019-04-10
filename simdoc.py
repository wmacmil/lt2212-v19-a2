import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()

path3 = args.vectorfile

def read_in(file_name):
    df_input = pd.read_csv(file_name, index_col=0)
    df_input_split = df_input.to_dict('split')
    returned_input_data1 = df_input_split['data']
    return returned_input_data1

xytc_mat = read_in(path3)

# read_in(path2) == tr(first)

# f is the word vectors for each document in the first folder

# break apart
first = xytc_mat[:582]
second = xytc_mat[582:]

cssf = cosine_similarity(second,first)
csfs = cosine_similarity(first,second)
csff = cosine_similarity(first,first)
csss = cosine_similarity(second,second)

# # intersection
# ifirsts = inters(first,second)
# # s minus the intersection
# smifirsts = minusxs(ifirsts,second)

averages = [np.average(x) for x in [csfs,cssf,csff,csss]]

print("Reading matrix from {}.".format(args.vectorfile))
print(averages)

