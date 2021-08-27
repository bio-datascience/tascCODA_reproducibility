# Only relevant for server execution
import sys
import argparse

sys.path.insert(0, '/home/icb/johannes.ostner/tree_aggregation/')
sys.path.insert(0, '/Users/johannes.ostner/Documents/PhD/tree_aggregation/')

import benchmarking_2.scripts.benchmark_functions as ben

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data_path', nargs="?", type=str)
parser.add_argument('-s', '--save_path', nargs="?", type=str)
parser.add_argument('-n', '--min_id', nargs="?", type=int)
parser.add_argument('-x', '--max_id', nargs="?", type=int)
parser.add_argument('-m', '--model', nargs="?", type=str)
parser.add_argument('-l', '--lbda', nargs="?", type=int)
parser.add_argument('-p', '--phi', nargs="?", type=float)
parser.add_argument('-r', '--reg_method', nargs="?", type=str)
parser.add_argument('-b', '--batch_id', nargs="?", type=int)

args = parser.parse_args()

data_path = args.data_path
save_path = args.save_path
min_id = args.min_id
max_id = args.max_id
model = args.model
lbda = args.lbda
phi = args.phi
reg_method = args.reg_method
batch_id = args.batch_id

ben.benchmark_job(
    data_path=data_path,
    save_path=save_path,
    min_id=min_id,
    max_id=max_id,
    batch_id=batch_id,
    model=model,
    lbda=lbda,
    phi=phi,
    reg_method=reg_method,
    keep_results=True
)
