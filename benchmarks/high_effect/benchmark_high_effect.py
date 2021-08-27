import sys
sys.path.insert(0, '/Users/johannes.ostner/Documents/PhD/tree_aggregation/')
sys.path.insert(0, '/home/icb/johannes.ostner/tree_aggregation/')
import benchmarking_2.scripts.benchmark_functions as util

data_path = "/home/icb/johannes.ostner/tree_aggregation/benchmarking_2/benchmark_high_effect/data"
result_path = "/home/icb/johannes.ostner/tree_aggregation/benchmarking_2/benchmark_high_effect/results3"
bash_path = "/home/icb/johannes.ostner/tree_aggregation/benchmarking_2/benchmark_high_effect/slurm"

model = "tree_agg"
model_params = {
    "phi": [-10, -5, -1, 0, 1, 5, 10]
}
reg_methods = ["new_3"]
dpj = 10

for r in reg_methods:
    benchmark_name = f"benchmark_high_effect_{r}"

    util.run_benchmark_one_model(
        data_path,
        result_path,
        bash_path,
        benchmark_name,
        model,
        r,
        model_params,
        datasets_per_job=dpj,
    )

