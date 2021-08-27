# Functions for tree aggregation benchmarks that are executed from within a slurm job

import numpy as np
import pandas as pd
import sys
import pickle as pkl
import os
import itertools
import anndata as ad
import toytree as tt
import statsmodels as sm

sys.path.insert(0, '/home/icb/johannes.ostner/tree_aggregation/')
sys.path.insert(0, '/Users/johannes.ostner/Documents/PhD/tree_aggregation/')

import tree_aggregation.tree_data_generation as tgen
import tree_aggregation.tree_ana as ta
import sccoda.util.comp_ana as ca
import benchmarking_2.scripts.bash_gen as ben
# from sccoda.model import other_models as om

# import rpy2.robjects as rp
# from rpy2.robjects import numpy2ri, pandas2ri
# numpy2ri.activate()
# pandas2ri.activate()


def generate_datasets(K, D, cells_per_type, n_rep, save_path, seed=None):

    if seed is not None:
        np.random.seed(seed)

    id_df = pd.DataFrame(columns=[
        "n_otus",
        "total_nodes",
        "n_internal_nodes",
        "tree_depth",
        "effect_nodes",
        "effect_leaves"
    ])

    data_id = 0

    for k in K:
        for d in D:
            for n in range(n_rep):
                print(f"\n Parameters: {(k ,d, n)} \n")

                # generate data
                tgen_data, effect_info = tgen.generate_random_tree_data(K=k, D=d, cells_per_type=cells_per_type)
                tgen_data.uns["effect_info"] = effect_info

                t = tgen_data.uns["phylo_tree"].nnodes
                effect_nodes = effect_info[0]
                if not type(effect_nodes) == list:
                    effect_nodes = [effect_nodes]
                effect_leaves = effect_info[1]

                id_df = id_df.append(
                    dict(zip(id_df.columns, [
                        k,
                        t,
                        t-k,
                        d,
                        effect_nodes,
                        effect_leaves
                    ])),
                    ignore_index=True
                )

                try:
                    os.mkdir(save_path + "/data/datasets")
                except FileExistsError:
                    pass

                tgen_data.uns["phylo_tree_newick"] = tgen_data.uns["phylo_tree"].write()
                tgen_data.uns.pop("phylo_tree")
                tgen_data.write_h5ad(save_path + f"/data/datasets/data_{data_id}")

                data_id += 1

    id_df.to_csv(save_path + "/data/info_df")


def run_benchmark_one_model(data_path, result_path, bash_path, benchmark_name, model, reg_method="None", model_params={}, datasets_per_job=500):

    num_datasets = len(os.listdir(data_path + "/datasets2"))

    batch_id = 0
    min_data = 0
    max_data = min(datasets_per_job, num_datasets)
    datasets_left = True

    param_combinations = [dict(zip(model_params.keys(), c)) for c in list(itertools.product(*model_params.values()))]

    print(f"model: {model}")
    print(f"reg_method: {reg_method}")

    while datasets_left:

        print(f"max data: {max_data}")
        if max_data == num_datasets:
            datasets_left = False

        for p in param_combinations:
            print(p)
            # script_location = "/Users/johannes.ostner/Documents/PhD/tree_aggregation/benchmarking_2/scripts/tree_benchmark_job.py"
            script_location = "/home/icb/johannes.ostner/tree_aggregation/benchmarking_2/scripts/tree_benchmark_job.py"

            bash_name = f"{benchmark_name}_{batch_id}"
            args = {
                "data_path": data_path,
                "save_path": result_path,
                "min_id": min_data,
                "max_id": max_data,
                "model": model,
                "reg_method": reg_method,
                "batch_id": batch_id,
            }

            if "phi" in p:
                args["phi"] = p["phi"]

            if "lambda" in p:
                args["lbda"] = p["lambda"]

            ben.execute_on_server(bash_path + "/", bash_name, script_location, args,
                              python_path="/home/icb/johannes.ostner/anaconda3/envs/tree_aggregation/bin/python"
                              # python_path="/Users/johannes.ostner/opt/anaconda3/envs/scCODA_2/bin/python"
                              )

            batch_id += 1

        min_data = max_data
        max_data = min(max_data + datasets_per_job, num_datasets)


def benchmark_job(data_path, save_path, min_id, max_id, batch_id, model, reg_method="None", lbda=np.nan, phi=np.nan, keep_results=False):

    result_df = pd.DataFrame(columns=[
        "model",
        "reg_method",
        "lambda",
        "phi",
        "result_nodes",
        "result_otus",
        "mean_log_prob",
        "acc_rate",
        "num_samples",
        "effect_size",
        "num_otus",
        "dataset_id",
    ])

    result_dict = {}

    num_results = 20000
    num_burnin = 5000

    for n in range(min_id, max_id, 1):
        print(n)
        dat = ad.read_h5ad(data_path + f"/datasets2/data_{n}")
        dat.uns["phylo_tree"] = tt.tree(newick=dat.uns["tree_newick"])

        res_key = (n, model, reg_method, lbda, phi)

        num_samples = dat.uns["num_samples"]
        effect_size = dat.uns["effect_size"]
        num_otus = dat.X.shape[1]

        if model == "sccoda":

            # scCODA model
            sccoda_model = ca.CompositionalAnalysis(
                data=dat,
                formula="x_0",
                reference_cell_type=str(num_otus-1)
            )
            sccoda_result = sccoda_model.sample_hmc(num_results=num_results, num_burnin=num_burnin)
            # Credible OTUs and nodes (=OTUs)
            res_otus = sccoda_result.effect_df[
                sccoda_result.effect_df["Final Parameter"] != 0].index.droplevel(
                0).astype(int).values.tolist()
            res_nodes = res_otus
            # Mean log_prob
            mlp = np.mean(sccoda_result.sample_stats.target_log_prob.values)
            acc_rate = sccoda_result.sampling_stats["acc_rate"]
            alphas_df, betas_df = sccoda_result.summary_prepare()

            # append betas_df to output list
            if keep_results:
                result_dict[res_key] = (alphas_df, betas_df)

            result_df = result_df.append(
                dict(zip(result_df.columns, [
                    model,
                    reg_method,
                    lbda,
                    phi,
                    res_nodes,
                    res_otus,
                    mlp,
                    acc_rate,
                    num_samples,
                    effect_size,
                    num_otus,
                    n,
                ])),
                ignore_index=True
            )

        elif model == "tree_agg":

            if reg_method == "L0":

                pen_args: dict = {"lambda": lbda}

                # Tree model
                ta_model = ta.CompositionalAnalysisTree(
                    data=dat,
                    formula="x_0",
                    reference_cell_type=str(num_otus-1),
                    pen_args=pen_args,
                    reg=reg_method
                )
                ta_result = ta_model.sample_hmc(num_results=num_results, num_burnin=num_burnin)
                # Credible OTUs and nodes
                res_nodes = ta_result.node_df[ta_result.node_df["Final Parameter"] != 0].index.droplevel(
                    0).values.tolist()
                res_otus = np.where(np.matmul(ta_model.A, ta_result.node_df["Final Parameter"]) != 0)[0].tolist()
                # mean log_prob
                mlp = np.mean(ta_result.sample_stats.target_log_prob.values)
                acc_rate = ta_result.sampling_stats["acc_rate"]
                alphas_df, betas_df, node_df = ta_result.summary_prepare()

                result_df = result_df.append(
                    dict(zip(result_df.columns, [
                        model,
                        reg_method,
                        lbda,
                        phi,
                        res_nodes,
                        res_otus,
                        mlp,
                        acc_rate,
                        num_samples,
                        effect_size,
                        num_otus,
                        n,
                    ])),
                    ignore_index=True
                )

                # append betas_df to output list
                if keep_results:
                    result_dict[res_key] = (alphas_df, betas_df, node_df)

            elif reg_method == "None":

                # Tree model
                ta_model = ta.CompositionalAnalysisTree(
                    data=dat,
                    formula="x_0",
                    reference_cell_type=str(num_otus-1),
                    pen_args={},
                    reg=reg_method
                )
                ta_result = ta_model.sample_hmc(num_results=num_results, num_burnin=num_burnin)
                # Credible OTUs and nodes
                res_nodes = ta_result.node_df[ta_result.node_df["Final Parameter"] != 0].index.droplevel(
                    0).values.tolist()
                res_otus = np.where(np.matmul(ta_model.A, ta_result.node_df["Final Parameter"]) != 0)[
                    0].tolist()
                # mean log_prob
                mlp = np.mean(ta_result.sample_stats.target_log_prob.values)
                acc_rate = ta_result.sampling_stats["acc_rate"]
                alphas_df, betas_df, node_df = ta_result.summary_prepare()

                result_df = result_df.append(
                    dict(zip(result_df.columns, [
                        model,
                        reg_method,
                        lbda,
                        phi,
                        res_nodes,
                        res_otus,
                        mlp,
                        acc_rate,
                        num_samples,
                        effect_size,
                        num_otus,
                        n,
                    ])),
                    ignore_index=True
                )

                # append betas_df to output list
                if keep_results:
                    result_dict[res_key] = (alphas_df, betas_df, node_df)

            elif reg_method == "L0_scaled":

                pen_args: dict = {"lambda": lbda, "phi": phi}

                # Tree model
                ta_model = ta.CompositionalAnalysisTree(
                    data=dat,
                    formula="x_0",
                    reference_cell_type=str(num_otus-1),
                    pen_args=pen_args,
                    reg=reg_method
                )
                ta_result = ta_model.sample_hmc(num_results=num_results, num_burnin=num_burnin)
                # Credible OTUs and nodes
                res_nodes = ta_result.node_df[ta_result.node_df["Final Parameter"] != 0].index.droplevel(
                    0).values.tolist()
                res_otus = np.where(np.matmul(ta_model.A, ta_result.node_df["Final Parameter"]) != 0)[
                    0].tolist()
                # mean log_prob
                mlp = np.mean(ta_result.sample_stats.target_log_prob.values)
                acc_rate = ta_result.sampling_stats["acc_rate"]
                alphas_df, betas_df, node_df = ta_result.summary_prepare()

                result_df = result_df.append(
                    dict(zip(result_df.columns, [
                        model,
                        reg_method,
                        lbda,
                        phi,
                        res_nodes,
                        res_otus,
                        mlp,
                        acc_rate,
                        num_samples,
                        effect_size,
                        num_otus,
                        n,
                    ])),
                    ignore_index=True
                )

                # append betas_df to output list
                if keep_results:
                    result_dict[res_key] = (alphas_df, betas_df, node_df)

            elif reg_method == "new":

                pen_args: dict = {"lambda_0": lbda, "phi": phi, "lambda_1": 1}

                # Tree model
                ta_model = ta.CompositionalAnalysisTree(
                    data=dat,
                    formula="x_0",
                    reference_cell_type=str(num_otus-1),
                    pen_args=pen_args,
                    reg="scaled",
                    model="new"
                )
                ta_result = ta_model.sample_hmc_da(num_results=num_results, num_burnin=num_burnin)
                # Credible OTUs and nodes
                res_nodes = ta_result.node_df[np.abs(ta_result.node_df["Final Parameter"]) > 0.15].index.droplevel(
                    0).values.tolist()
                res_otus = np.where(np.abs(np.matmul(ta_model.A, ta_result.node_df["Final Parameter"])) > 0.15)[
                    0].tolist()
                # mean log_prob
                alphas_df, betas_df, node_df = ta_result.summary_prepare()

                mlp = np.mean(ta_result.sample_stats.target_log_prob.values)
                acc_rate = ta_result.sampling_stats["acc_rate"]

                result_df = result_df.append(
                    dict(zip(result_df.columns, [
                        model,
                        reg_method,
                        lbda,
                        phi,
                        res_nodes,
                        res_otus,
                        mlp,
                        acc_rate,
                        num_samples,
                        effect_size,
                        num_otus,
                        n,
                    ])),
                    ignore_index=True
                )

                # append betas_df to output list
                if keep_results:
                    result_dict[res_key] = (alphas_df, betas_df, node_df)

            elif reg_method == "new_2":

                pen_args: dict = {"lambda_0": 50, "phi": phi, "lambda_1": 1}

                # Tree model
                ta_model = ta.CompositionalAnalysisTree(
                    data=dat,
                    formula="x_0",
                    reference_cell_type=str(num_otus-1),
                    pen_args=pen_args,
                    reg="scaled_2",
                    model="new"
                )
                ta_result = ta_model.sample_hmc_da(num_results=num_results, num_burnin=num_burnin)
                # Credible OTUs and nodes
                res_nodes = ta_result.node_df[np.abs(ta_result.node_df["Final Parameter"]) != 0].index.droplevel(
                    0).values.tolist()
                res_otus = np.where(np.abs(np.matmul(ta_model.A, ta_result.node_df["Final Parameter"])) != 0)[
                    0].tolist()
                # mean log_prob
                alphas_df, betas_df, node_df = ta_result.summary_prepare()

                mlp = np.mean(ta_result.sample_stats.target_log_prob.values)
                acc_rate = ta_result.sampling_stats["acc_rate"]

                result_df = result_df.append(
                    dict(zip(result_df.columns, [
                        model,
                        reg_method,
                        lbda,
                        phi,
                        res_nodes,
                        res_otus,
                        mlp,
                        acc_rate,
                        num_samples,
                        effect_size,
                        num_otus,
                        n,
                    ])),
                    ignore_index=True
                )

                # append betas_df to output list
                if keep_results:
                    result_dict[res_key] = (alphas_df, betas_df, node_df)

            elif reg_method == "new_3":

                pen_args: dict = {"lambda_0": 50, "phi": phi, "lambda_1": 5}

                # Tree model
                ta_model = ta.CompositionalAnalysisTree(
                    data=dat,
                    formula="x_0",
                    reference_cell_type=str(num_otus-1),
                    pen_args=pen_args,
                    reg="scaled_3",
                    model="new"
                )
                ta_result = ta_model.sample_hmc_da(num_results=num_results, num_burnin=num_burnin)
                # Credible OTUs and nodes
                res_nodes = ta_result.node_df[np.abs(ta_result.node_df["Final Parameter"]) != 0].index.droplevel(
                    0).values.tolist()
                res_otus = np.where(np.abs(np.matmul(ta_model.A, ta_result.node_df["Final Parameter"])) != 0)[
                    0].tolist()
                # mean log_prob
                alphas_df, betas_df, node_df = ta_result.summary_prepare()

                mlp = np.mean(ta_result.sample_stats.target_log_prob.values)
                acc_rate = ta_result.sampling_stats["acc_rate"]

                result_df = result_df.append(
                    dict(zip(result_df.columns, [
                        model,
                        reg_method,
                        lbda,
                        phi,
                        res_nodes,
                        res_otus,
                        mlp,
                        acc_rate,
                        num_samples,
                        effect_size,
                        num_otus,
                        n,
                    ])),
                    ignore_index=True
                )

                # append betas_df to output list
                if keep_results:
                    result_dict[res_key] = (alphas_df, betas_df, node_df)

            elif reg_method == "x_1_add":

                pen_args: dict = {"lambda_0": 50, "phi": phi, "lambda_1": 5}

                # Tree model
                ta_model = ta.CompositionalAnalysisTree(
                    data=dat,
                    formula="x_0 + x_1",
                    reference_cell_type=str(num_otus-1),
                    pen_args=pen_args,
                    reg="scaled_3",
                    model="new"
                )
                ta_result = ta_model.sample_hmc_da(num_results=num_results, num_burnin=num_burnin)
                # Credible OTUs and nodes
                res_nodes = ta_result.node_df[np.abs(ta_result.node_df["Final Parameter"]) != 0].index.droplevel(
                    0).values.tolist()
                res_otus = np.where(ta_result.effect_df["Effect"] != 0)[
                    0].tolist()
                # mean log_prob
                alphas_df, betas_df, node_df = ta_result.summary_prepare()

                mlp = np.mean(ta_result.sample_stats.target_log_prob.values)
                acc_rate = ta_result.sampling_stats["acc_rate"]

                result_df = result_df.append(
                    dict(zip(result_df.columns, [
                        model,
                        reg_method,
                        lbda,
                        phi,
                        res_nodes,
                        res_otus,
                        mlp,
                        acc_rate,
                        num_samples,
                        effect_size,
                        num_otus,
                        n,
                    ])),
                    ignore_index=True
                )

                # append betas_df to output list
                if keep_results:
                    result_dict[res_key] = (alphas_df, betas_df, node_df)

            elif reg_method == "x_1_mult":

                pen_args: dict = {"lambda_0": 50, "phi": phi, "lambda_1": 5}

                # Tree model
                ta_model = ta.CompositionalAnalysisTree(
                    data=dat,
                    formula="x_0 * x_1",
                    reference_cell_type=str(num_otus-1),
                    pen_args=pen_args,
                    reg="scaled_3",
                    model="new"
                )
                ta_result = ta_model.sample_hmc_da(num_results=num_results, num_burnin=num_burnin)
                # Credible OTUs and nodes
                res_nodes = ta_result.node_df[np.abs(ta_result.node_df["Final Parameter"]) != 0].index.droplevel(
                    0).values.tolist()
                res_otus = np.where(ta_result.effect_df["Effect"] != 0)[
                    0].tolist()
                # mean log_prob
                alphas_df, betas_df, node_df = ta_result.summary_prepare()

                mlp = np.mean(ta_result.sample_stats.target_log_prob.values)
                acc_rate = ta_result.sampling_stats["acc_rate"]

                result_df = result_df.append(
                    dict(zip(result_df.columns, [
                        model,
                        reg_method,
                        lbda,
                        phi,
                        res_nodes,
                        res_otus,
                        mlp,
                        acc_rate,
                        num_samples,
                        effect_size,
                        num_otus,
                        n,
                    ])),
                    ignore_index=True
                )

                # append betas_df to output list
                if keep_results:
                    result_dict[res_key] = (alphas_df, betas_df, node_df)

            else:
                raise ValueError("Invalid regularization name specified!")

        else:
            raise ValueError("Invalid model name specified!")

    result_df.to_csv(save_path + f"/result_df_{model}_{reg_method}_{batch_id}.csv")

    try:
        os.mkdir(save_path + "/exact_results")
    except FileExistsError:
        pass

    # save stuff
    with open(save_path + f"/exact_results/results_{model}_{reg_method}_{batch_id}.pkl", "wb") as f:
        pkl.dump(result_dict, f)


def other_models_benchmarking(data_dir, result_dir, model):
    print(f"Running {model}...")
    r_home = "/Library/Frameworks/R.framework/Resources"
    r_path = r"/Library/Frameworks/R.framework/Resources/bin"

    os.environ["R_HOME"] = r_home
    os.environ["PATH"] = r_path + ";" + os.environ["PATH"]

    result_df = pd.DataFrame(columns=[
        "model",
        "result_nodes",
        "result_otus",
        "dataset_id",
        "effect_nodes",
        "effect_otus",
        "num_samples",
        "effect_size",
        "num_otus"
    ])

    result_dict = {}

    for id in range(len(os.listdir(data_dir))):
        if id % 100 == 0:
            print(f"{id}/{len(os.listdir(data_dir))}")
        res_key = (id, model)

        data = ad.read_h5ad(data_dir + f"/data_{id}")

        effect_nodes = data.uns["effect_nodes"]
        effect_otus = data.uns["effect_leaves"]
        num_samples = data.uns["num_samples"]
        effect_size = data.uns["effect_size"]
        num_otus = data.X.shape[1]

        result_nodes, result_otus, out = run_on_one_dataset(data, model, r_home, r_path)

        result_df = result_df.append(
            dict(zip(result_df.columns, [
                model,
                result_nodes,
                result_otus,
                id,
                effect_nodes,
                effect_otus,
                num_samples,
                effect_size,
                num_otus
            ])),
            ignore_index=True
        )

        result_dict[res_key] = out

    result_df.to_csv(result_dir + f"/result_df_{model}.csv")

    try:
        os.mkdir(result_dir + "/exact_results")
    except FileExistsError:
        pass

    # save stuff
    with open(result_dir + f"/exact_results/results_{model}.pkl", "wb") as f:
        pkl.dump(result_dict, f)

    return result_df, result_dict


def run_on_one_dataset(data, model, r_home, r_path):

    if model == "adaANCOM":
        tree = tt.tree(newick=data.uns["tree_newick"])
        newick = tree.newick

        ada_out = rp.r(f"""
            library(adaANCOM)
            library(phyloseq)
            library(ape)

            newick <- "{newick}"
            tree <- ape::read.tree(text=newick)
            tree <- collapse.singles(ape::multi2di(tree))
            tree$edge.length <- rep(1, length(tree$edge.length))
            tree$node.label <- (tree$Nnode+1):((tree$Nnode+1)+tree$Nnode-1)+1

            counts <- {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(data.X, columns=data.var.index)).r_repr()}
            colnames(counts) <- rev(tree$tip.label)
            group <- {pandas2ri.py2rpy_pandasdataframe(pd.DataFrame(data.obs["x_0"])).r_repr()}[,1]
            
            invisible(capture.output(fit <- adaANCOM(otu=counts, Y=group, tree=tree, tfun = t.test, smooth=0.5)))
            fit$L$p.adj <- p.adjust(fit$L$p.value, method = 'fdr')
            fit

            """)

        V = pd.DataFrame(ada_out[0])
        L = pd.DataFrame(ada_out[1])

        result_nodes = list(V.loc[V["p.adj"] < 0.05, "taxa"])
        result_otus = list(L.loc[L["p.adj"] < 0.05, "otu"])
        out = (V, L)

    elif model == "ANCOM":

        ancom_mod = om.AncomModel(data, "x_0")
        ancom_mod.fit_model()
        accept = np.array(ancom_mod.ancom_out[0]["Reject null hypothesis"])

        result_nodes = []
        result_otus = list(np.where(accept == True)[0])
        out = ancom_mod.ancom_out[0]

    elif model == "ANCOMBC":

        ancombc_mod = om.ANCOMBCModel(data, "x_0")
        ancombc_mod.fit_model(r_home=r_home, r_path=r_path)
        p_val = np.array(ancombc_mod.p_val)

        result_nodes = []
        result_otus = list(np.where(p_val < 0.05)[0])
        out = pd.DataFrame({
                    "Cell Type": data.var.index.to_list,
                    "p value": ancombc_mod.p_val,
                })

    elif model == "ALDEx2":
        K = data.X.shape[1]

        aldex_mod = om.ALDEx2Model(data, "x_0")
        aldex_mod.fit_model(r_home=r_home, r_path=r_path, denom=[K])
        p_val = np.array(aldex_mod.p_val)

        pval = np.nan_to_num(np.array(p_val), nan=1)
        reject, pvals, _, _ = sm.stats.multitest.multipletests(pval, 0.05, method="fdr_bh")

        result_nodes = []
        result_otus = list(np.where(pvals < 0.05)[0])
        out = pd.DataFrame({
            "Cell Type": data.var.index.to_list,
            "p value": pvals,
        })

    elif model == "alr_ttest":
        K = data.X.shape[1]

        ttest_mod = om.ALRModel_ttest(data, "x_0")
        ttest_mod.fit_model(reference_cell_type=K-1)
        p_val = np.array(ttest_mod.p_val)

        pval = np.nan_to_num(np.array(p_val), nan=1)
        reject, pvals, _, _ = sm.stats.multitest.multipletests(pval, 0.05, method="fdr_bh")

        result_nodes = []
        result_otus = list(np.where(pvals < 0.05)[0])
        out = pd.DataFrame({
            "Cell Type": data.var.index.to_list,
            "p value": pvals,
        })

    elif model == "alr_wilcoxon":
        K = data.X.shape[1]

        wilcox_mod = om.ALRModel_wilcoxon(data, "x_0")
        wilcox_mod.fit_model(reference_cell_type=K-1)
        p_val = np.array(wilcox_mod.p_val)

        pval = np.nan_to_num(np.array(p_val), nan=1)
        reject, pvals, _, _ = sm.stats.multitest.multipletests(pval, 0.05, method="fdr_bh")

        result_nodes = []
        result_otus = list(np.where(pvals < 0.05)[0])
        out = pd.DataFrame({
            "Cell Type": data.var.index.to_list,
            "p value": pvals,
        })

    else:
        raise(ValueError, "This model does not exist!")

    return result_nodes, result_otus, out


