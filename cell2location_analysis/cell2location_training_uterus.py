import scanpy as sc
import os
data_type = 'float32'
os.environ["THEANO_FLAGS"] = 'device=cuda,floatX=' + data_type + ',force_device=True'
import cell2location
import numpy as np

uterus_vis_sc = sc.read("vasculature/data/public_visium/uterus/uterus_visium.lognormalized.h5ad")
uterus_ref_sc = sc.read("vasculature/uterus/ref.posterior.h5ad")

if 'means_per_cluster_mu_fg' in uterus_ref_sc.varm.keys():
    inf_aver = uterus_ref_sc.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                          for i in uterus_ref_sc.uns['mod']['factor_names']]].copy()
else:
    inf_aver = uterus_ref_sc.var[[f'means_per_cluster_mu_fg_{i}'
                                      for i in uterus_ref_sc.uns['mod']['factor_names']]].copy()
inf_aver.columns = uterus_ref_sc.uns['mod']['factor_names']
intersect = np.intersect1d(uterus_vis_sc.var_names, inf_aver.index)
uterus_vis_sc = uterus_vis_sc[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()


# create and train the model
print("cell2location data setup")
cell2location.models.Cell2location.setup_anndata(adata=uterus_vis_sc,
                                                 layer='counts',
                                                 batch_key='sample')
print("cell2location model setup")
c2l_mod = cell2location.models.Cell2location(
    uterus_vis_sc,
    cell_state_df=inf_aver,
    N_cells_per_location=8,
    detection_alpha=20
)
print(c2l_mod.view_anndata_setup())
print("####### Start with Cell2location training model.")
c2l_mod.train(max_epochs=400,
          batch_size=1000,
          train_size=1,
          use_gpu=True)

uterus_vis_sc = c2l_mod.export_posterior(
        uterus_vis_sc, sample_kwargs={'num_samples': 1000,
                                  'batch_size': c2l_mod.adata.n_obs,
                                  'use_gpu': True}
    )
os.makedirs("vasculature/uterus/ncells8_alpha200/")
c2l_mod.save("vasculature/uterus/ncells8_alpha200/", overwrite=True)
uterus_vis_sc.obs[uterus_vis_sc.uns['mod']['factor_names']] = uterus_vis_sc.obsm['q05_cell_abundance_w_sf']
print("computing KNN from estimated cell abundance")
sc.pp.neighbors(uterus_vis_sc, use_rep='q05_cell_abundance_w_sf', n_neighbors=15)
sc.tl.leiden(uterus_vis_sc)
uterus_vis_sc.obs["region_cluster"] = uterus_vis_sc.obs["leiden"].astype("category")
sc.tl.umap(uterus_vis_sc, min_dist=0.3, spread=1)
uterus_vis_sc.write("vasculature/uterus/ncells8_alpha200/sp.h5ad")

