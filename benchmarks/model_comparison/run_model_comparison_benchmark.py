import sys
sys.path.insert(0, '/Users/johannes.ostner/Documents/PhD/tree_aggregation/')
sys.path.insert(0, '/home/icb/johannes.ostner/tree_aggregation/')
import benchmarking_2.scripts.benchmark_functions as util

data_path = "/home/icb/johannes.ostner/tree_aggregation/benchmarking_2/benchmark_0808/data"
result_path = "/home/icb/johannes.ostner/tree_aggregation/benchmarking_2/benchmark_0808/results"
bash_path = "/home/icb/johannes.ostner/tree_aggregation/benchmarking_2/benchmark_0808/slurm"
benchmark_name = "benchmark_0808"

model = "tree_agg"
model_params = {
    "phi": [-10, -5, -1, 0, 1, 5, 10]
}
reg_method = "new_3"
dpj = 50

util.run_benchmark_one_model(
    data_path,
    result_path,
    bash_path,
    benchmark_name,
    model,
    reg_method,
    model_params,
    datasets_per_job=dpj,
)

model = "sccoda"

util.run_benchmark_one_model(
    data_path,
    result_path,
    bash_path,
    benchmark_name,
    model,
    datasets_per_job=dpj,
)

