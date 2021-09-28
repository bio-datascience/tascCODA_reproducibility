import os
import pandas as pd
import anndata as ad
import numpy as np
import toytree as tt
import toyplot
import sys

import sccoda.util.comp_ana as mod
import sccoda.model.other_models as om
import tree_aggregation.tree_ana as ta

tax_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]


def agg_ibs_data(author, level):

    # read data
    data_dir = '../../../tascCODA_data/applications/labus_IBS/Labus-2017_data/'
    file_name = data_dir + f"/{author.lower()}_{level.lower()}-agg.csv"
    raw_data = pd.read_csv(file_name, index_col=0)

    # get taxonomic levels in the data
    tl = [x for x in tax_levels[:tax_levels.index(level) + 1]]

    # extract counts
    count_data = raw_data.pivot(index="Sample", columns=tl, values="Abundance")
    # get taxonomic tree (for data.var)
    tax_info = pd.DataFrame(index=count_data.columns).reset_index()
    tax_index = tax_info.apply('*'.join, axis=1)
    tax_info.index = tax_index

    count_data.columns = tax_index

    # get metadata
    metadata_cols = raw_data.columns.drop(["Sample", "Abundance"] + tax_levels, errors="ignore")
    metadata = raw_data.groupby("Sample").agg(dict([(x, "first") for x in metadata_cols]))

    ret = ad.AnnData(X=count_data, obs=metadata, var=tax_info)
    return ret


def run_sccoda(author, level, add=None, fdr_level=0.1):
    references = {
        "Genus": "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales*Rikenellaceae*Alistipes",
        "Family": "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales*Rikenellaceae",
        "Order": "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales",
        "Class": "Bacteria*Bacteroidota*Bacteroidia",
        "Phylum": "Bacteria*Bacteroidota",
    }

    data = agg_ibs_data(author, level)
    if add is not None:
        data = data[data.obs[add[0]] == add[1]]

    model = mod.CompositionalAnalysis(
        data=data,
        formula="C(host_disease, Treatment('Healthy'))",
        reference_cell_type=references[level]
    )
    result = model.sample_hmc()
    _, effect_df = result.summary_prepare(est_fdr = fdr_level)

    return effect_df


def run_tree_agg(author, level, add=None, fdr_level=0.1, reg="None", pen_args={"lambda": 10, "phi": 2}, model="old"):

    references = {
        "Genus": "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales*Rikenellaceae*Alistipes",
        "Family": "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales*Rikenellaceae",
        "Order": "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales",
        "Class": "Bacteria*Bacteroidota*Bacteroidia",
        "Phylum": "Bacteria*Bacteroidota",
    }

    data = agg_ibs_data(author, level)
    if add is not None:
        data = data[data.obs[add[0]] == add[1]]

    n_level = tax_levels.index(level)+1

    for char in data.var.index:
        split = char.split(sep="*")

        for n in range(n_level):
            data.var.loc[char, tax_levels[n]] = "*".join(split[:n+1])

    newick = df2newick(data.var.reset_index(drop=True))

    tree = tt.tree(newick, tree_format=1)
    data.uns["phylo_tree"] = tree

    data = data[:, tree.get_tip_labels()]

    model = ta.CompositionalAnalysisTree(
        data=data,
        formula="C(host_disease, Treatment('Healthy'))",
        reference_cell_type=references[level],
        reg=reg,
        pen_args=pen_args,
        model=model
    )
    result = model.sample_hmc_da(num_results=2000, num_burnin=500)
    result.set_fdr(fdr_level)

    eff_lv = get_phylo_levels(result.effect_df.reset_index(), level)
    result.effect_df.index = pd.MultiIndex.from_frame(eff_lv)

    nd = result.node_df.reset_index()
    node_lv = get_phylo_levels(nd, level, "Node")
    result.node_df.index = pd.MultiIndex.from_frame(node_lv)
    result.node_df["Cell Type"] = np.array(nd.loc[:, "Node"])

    return result


def run_ancom_model(author, level, add=None):
    data = agg_ibs_data(author, level)
    if add is not None:
        data = data[data.obs[add[0]] == add[1]]

    data.X[data.X == 0] = 0.5
    ac = om.AncomModel(data, covariate_column="host_disease")
    ac.fit_model()
    out = ac.ancom_out[0].loc[:, ["W", "Reject null hypothesis"]]

    return out

def run_ancombc_model(author, level, add=None, alpha=0.05):
    data = agg_ibs_data(author, level)
    if add is not None:
        data = data[data.obs[add[0]] == add[1]]

    data.X[data.X == 0] = 0.5
    ac = om.ANCOMBCModel(data, covariate_column="host_disease")
    ac.fit_model(
        alpha=alpha,
        r_home="/Library/Frameworks/R.framework/Resources",
        r_path=r"/Library/Frameworks/R.framework/Resources/bin",
    )
    out = pd.DataFrame({"p_adj": ac.p_val, "is_da": [x < alpha for x in ac.p_val]})
    out.index = data.var.index

    return out


def read_authors_results(authors, data_dir, method, add=None):
    out_dict = {}

    for l in tax_levels[1:]:
        ll = []
        for a in authors:
            for f in os.listdir(data_dir):
                start = f"{a.lower()}_{l.lower()}"
                if add:
                    start += f"_{add}"
                if f.startswith(start) and f.endswith(f"_{method}.csv"):
                    res_ = pd.read_csv(data_dir + "/" + f, index_col=0)
                    if method == "sccoda":
                        res_ = res_.reset_index()
                        res_["Is credible"] = (res_["Final Parameter"] != 0)
                    elif method == "ancom":
                        res_ = res_.reset_index().rename(columns={
                            "index": "Cell Type",
                            "Reject null hypothesis": "Is credible"
                        })
                    elif method == "ANCOMBC":
                        res_ = res_.reset_index().rename(columns={
                            "index": "Cell Type",
                            "is_da": "Is credible"
                        })
                    res_["author"] = a
                    tax = get_phylo_levels(res_, l)
                    res_ = pd.merge(res_, tax, left_index=True, right_index=True)
                    ll.append(res_)
        out_dict[l] = pd.concat(ll)

    return out_dict


def get_significances(out_dict, method):
    res_sigs = {}
    for l in tax_levels[1:]:

        res = out_dict[l]
        res_sig = res[res["Is credible"] == True]

        res_sig_gr = pd.DataFrame(res_sig.groupby("Cell Type").size()).rename(columns={0: "count"})
        res_sig_gr["model"] = method

        res_sigs[l] = res_sig_gr

    return res_sigs


def get_phylo_levels(results, level, col="Cell Type"):

    """
    Get a taxonomy table (columns are "Kingdom", "Phylum", ...) from a DataFrame where the one column contains full taxon names.

    :param results: pandas DataFrame
        One column must be strings of the form "<Kingdom>*<Phylum>*...", e.g. "Bacteria*Bacteroidota*Bacteroidia*Bacteroidales*Rikenellaceae*Alistipes"
    :param level: string
        Lowest taxonomic level (from ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]) that should be included
    :param col: string
        Name of the column with full taxon names
    :return:
    DataFrame with columns ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]

    """

    max_level_id = tax_levels.index(level)+1
    cols = tax_levels

    tax_table = pd.DataFrame(columns=cols, index=np.arange(len(results)))
    for i in range(len(results)):
        char = results.loc[i, col]
        split = char.split(sep="*")
        for j in range(max_level_id):
            try:
                tax_table.iloc[i, j] = split[j]
            except IndexError:
                tax_table.iloc[i, j] = None

    return tax_table


def traverse(df_, a, i, innerl):
    """
    Helper function for df2newick
    :param df_:
    :param a:
    :param i:
    :param innerl:
    :return:
    """
    if i+1 < df_.shape[1]:
        a_inner = pd.unique(df_.loc[np.where(df_.iloc[:, i] == a)].iloc[:, i+1])

        desc = []
        for b in a_inner:
            desc.append(traverse(df_, b, i+1, innerl))
        if innerl:
            il = a
        else:
            il = ""
        out = f"({','.join(desc)}){il}"
    else:
        out = a

    return out


def df2newick(df, inner_label=True):
    """
    Converts a taxonomy DataFrame into a Newick string
    :param df: DataFrame
        Must have columns ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
    :param inner_label: Boolean
        If True, internal nodes in the tree will keep their respective names
    :return:
        Newick string
    """
    df_tax = df.loc[:, [x for x in tax_levels if x in df.columns]]

    alevel = pd.unique(df_tax.iloc[:, 0])
    strs = []
    for a in alevel:
        strs.append(traverse(df_tax, a, 0, inner_label))

    newick = f"({','.join(strs)});"
    return newick


def build_fancy_tree(data):
    """
    Make a toytree object with all kinds of extras, like edge colors, effect sizes, ...

    :param data: Dictionary of DataFrames.
        Contains effect sizes, taxonomy info, etc. for all taxonomic ranks
        Dictionary keys should be ["Phylum", "Class", "Order", "Family", "Genus"]
        Each DataFrame should have the columns "Final Parameter" (= effect size), as well as columns for all taxonomic ranks
    :return:
        toytree object
    """

    # Get genus-level data for building a complete tree
    data_gen = data["Genus"]
    # Combine results from all levels into one df
    data_all = pd.concat(data.values())

    # Get newick string and build toytree with no extras
    newick = df2newick(data_gen)
    tree = tt.tree(newick, tree_format=1)

    # Edge colors.
    # The color dictionary is hard-coded, keys must be the names of all phyla in the data
    palette = toyplot.color.brewer.palette("Set1") + toyplot.color.brewer.palette("Set3")
    edge_color_dict = {
        "Firmicutes": palette[0],
        "Proteobacteria": palette[1],
        "Bacteroidota": palette[2],
        "Actinobacteriota": palette[3],
        "Synergistota": palette[4],
        "Fusobacteriota": palette[5],
        "Verrucomicrobiota": palette[6],
        "Desulfobacterota": palette[7],
        "Campilobacterota": palette[8],
        "Patescibacteria": palette[9],
        "Elusimicrobiota": palette[10],
        "Euryarchaeota": palette[11],
        "Spirochaetota": palette[12],
        "Deinococcota": palette[13],
        "Cyanobacteria": palette[14],
        "Halanaerobiaeota": palette[15],
        "Bdellovibrionota": palette[16],
        "Thermoplasmatota": palette[17],
        "WPS-2": palette[18],
    }

    # marker objects for plotting a legend
    markers = []

    # Height of the colored level (here: 2nd highest)
    max_height = np.max([n.height - 2 for n in tree.treenode.traverse()])

    # Iterate over all tree nodes and assign edge colors
    c = 0
    for n in tree.treenode.traverse():
        # If node corresponds to the colored level (here: Phylum), determine color and assign it as feature "edge_color" to all descendants
        if n.height == max_height:
            col = edge_color_dict[n.name]
            n.add_feature("edge_color", col)
            for n_ in n.get_descendants():
                n_.add_feature("edge_color", col)

            # Also add a marker for the legend
            col2 = '#%02x%02x%02x' % tuple([int(255*x) for x in col.tolist()[:-1]])
            m = toyplot.marker.create(shape="o", size=8, mstyle={"fill": col2})
            markers.append((n.name, m))

            c += 1
        # For all levels above the colored level, assign edge color black
        elif n.height > max_height:
            n.add_feature("edge_color", "black")

    # assign taxonomic levels to nodes (e.g. "Genus" for a node on the lowest level)
    for n in tree.treenode.traverse():
        if n.height == "":
            l = tax_levels[-1]
        elif n.height >= len(tax_levels):
            l = ""
        else:
            l = tax_levels[-(int(n.height) + 1)]
        n.add_feature("tax_level", l)

    # add effects to each node as feature "effect":
    # For all results, add the taxonomic rank (forgot to do that when combining initially)
    data_all["level"] = "Kingdom"
    for l in tax_levels[1:]:
        data_all.loc[pd.notna(data_all[l]), "level"] = l

    # Iterate over all tree nodes
    for n in tree.treenode.traverse():
        # Catch root node
        if n.tax_level == "":
            n.add_feature("effect", 0)
        else:
            # Find row in the effect DataFrame that mathches in taxonomic rank and name
            l = data_all.loc[(data_all["level"] == n.tax_level) & (data_all[n.tax_level] == n.name), :]
            # If there is only one matching row, assign effect size as node feature
            if len(l) == 1:
                n.add_feature("effect", l["Final Parameter"].values[0])
            # If there is no corresponding row, assign effect size 0
            elif len(l) == 0:
                n.add_feature("effect", 0)
            # If there are multiple corresponding rows (might happen if e.g. genera from two different families have the same name),
            # solve by comparing entire taxonomy assignment and assign effect size
            elif len(l) > 1:
                par_names = [n.name] + [m.name for m in n.get_ancestors()][:-1]
                par_names.reverse()
                full_name = '*'.join(par_names)
                ll = l[l["Cell Type"] == full_name]
                n.add_feature("effect", ll["Final Parameter"].values[0])


    # add node colors as feature "color", depending on whether effects are positive (black) or negative (white).
    # Effects of size 0 have no impact, as their markers will have size 0, but are colored in cyan for completeness.
    for n in tree.treenode.traverse():
        if np.sign(n.effect) == 1:
            n.add_feature("color", "black")
        elif np.sign(n.effect) == -1:
            n.add_feature("color", "white")
        else:
            n.add_feature("color", "cyan")

    return tree, markers
