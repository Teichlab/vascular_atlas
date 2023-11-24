import scanpy as sc
import numpy as np
import os
data_type = 'float32'
os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=' + data_type + ',force_device=True'
import cell2location
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from cell2location.models import RegressionModel


def read_and_qc(path, sample_name, hasMT=False):
    adata = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.obs['sample'] = sample_name
    adata.var['SYMBOL'] = adata.var_names
    adata.var.rename(columns={'gene_ids': 'ENSEMBL'}, inplace=True)
    adata_var_geneids = adata.var[['SYMBOL', 'ENSEMBL']].drop_duplicates(subset=['SYMBOL'])['ENSEMBL'].unique()
    adata = adata[:,adata.var['ENSEMBL'].isin(adata_var_geneids)]
    adata.var_names = adata.var['SYMBOL']
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    if hasMT:
        adata.var['mt'] = [gene.startswith('MT-') for gene in adata.var['SYMBOL']]
        adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze() / adata.obs['total_counts']
        adata.obsm['mt'] = adata[:, adata.var['mt'].values].X.toarray()
        adata = adata[:, ~adata.var['mt'].values]
    adata.obs["sample"] = adata.obs["sample"].astype(object)
    adata.obs_names = adata.obs["sample"] + '_' + adata.obs_names
    adata.obs.index.name = 'spot_id'
    return adata


def select_slide(adata, s, s_col='sample'):
    slide = adata[adata.obs[s_col].isin([s]), :]
    s_keys = list(slide.uns['spatial'].keys())
    s_spatial = np.array(s_keys)[[s in k for k in s_keys]][0]
    slide.uns['spatial'] = {s_spatial: slide.uns['spatial'][s_spatial]}
    return slide


def train_reference(region_ref,
                    annotation_colname,
                    batch_key,
                    continuous_covariate_keys,
                    categorical_covariate_keys,
                    write_prefix,
                    cell_number_thres=15):
    region_high_number_celltypes = list(region_ref.obs[annotation_colname].value_counts()[region_ref.obs[annotation_colname].value_counts() > cell_number_thres].index)
    print('Selected the following celltypes for getting signatures', region_high_number_celltypes)
    selected_region_ref = region_ref[region_ref.obs[annotation_colname].isin(region_high_number_celltypes)]
    # Regression model
    RegressionModel.setup_anndata(adata=selected_region_ref,
                                   batch_key=batch_key,
                                   labels_key=annotation_colname,
                                   continuous_covariate_keys=continuous_covariate_keys,
                                   categorical_covariate_keys=categorical_covariate_keys)
    mod = RegressionModel(selected_region_ref)
    print('Construct regression model:')
    print(mod.view_anndata_setup())
    print("Start training...")
    mod.train(max_epochs=250, use_gpu=True)
    selected_region_ref = mod.export_posterior(
        selected_region_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
    )
    mod.save(write_prefix / f"regression_model", overwrite=True)
    print("Model saved in.", write_prefix/'regression_model')
    selected_region_ref.write(write_prefix / f"ref.posterior.h5ad")
    print("Reference data with posterior values saved in ",
          write_prefix/f"ref.posterior.h5ad")
    return selected_region_ref


def train_cell2loc_model(adata_ref,
                         adata_vis,
                         ncells_per_location,
                         detection_alpha,
                         write_prefix,
                         batch_key='sample'):
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                              for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                          for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis,
                                                     batch_key=batch_key,
                                                     continuous_covariate_keys=['total_counts',
                                                                                'n_genes_by_counts',
                                                                                'mt_frac'])
    # create and train the model
    print("cell2location model setup")
    mod = cell2location.models.Cell2location(
        adata_vis,
        cell_state_df=inf_aver,
        N_cells_per_location=ncells_per_location,
        detection_alpha=detection_alpha
    )
    print(mod.view_anndata_setup())
    print("####### Start with Cell2location training model.")
    mod.train(max_epochs=30000,
              batch_size=None,
              train_size=1,
              use_gpu=True)
    # plot ELBO loss history during training, removing first 100 epochs from the plot
    mod.plot_history(1000)
    plt.legend(labels=['full data training'])
    if not os.path.exists(write_prefix / f"ncell{ncells_per_location}_alpha{detection_alpha}"):
        os.mkdir(write_prefix / f"ncell{ncells_per_location}_alpha{detection_alpha}")
    plt.savefig(write_prefix / f'cell2loc_model_ncell{ncells_per_location}_alpha{detection_alpha}.training_history.png')
    # export the estimated cell abundance (summary of the posterior distribution).
    print("Export the estimated cell abundance")
    adata_vis = mod.export_posterior(
        adata_vis, sample_kwargs={'num_samples': 1000,
                                  'batch_size': mod.adata.n_obs,
                                  'use_gpu': True}
    )
    mod.save(write_prefix / f"ncell{ncells_per_location}_alpha{detection_alpha}", overwrite=True)
    print('cell2location model saved in', write_prefix / f'cell2loc_model_ncell7_alpha{detection_alpha}')
    # add 5% quantile, representing confident cell abundance, 'at least this amount is present'
    adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']
    print("computing KNN from estimated cell abundance")
    sc.pp.neighbors(adata_vis, use_rep='q05_cell_abundance_w_sf',
                    n_neighbors=15)
    sc.tl.leiden(adata_vis, resolution=1.1)
    adata_vis.obs["region_cluster"] = adata_vis.obs["leiden"].astype("category")
    sc.tl.umap(adata_vis, min_dist=0.3, spread=1)
    adata_file = write_prefix / f"ncell{ncells_per_location}_alpha{detection_alpha}/sp.h5ad"
    adata_vis.write(adata_file)
    print("Saved visium data with estimated cell abundance, umap, leiden")
    return adata_vis
