import scanpy as sc
from pathlib import Path
from utilities import read_and_qc, train_cell2loc_model
from argparse import ArgumentParser


cleaned_heart_ref = sc.read_h5ad('vasculature/data/heart/cleaned_ref.h5ad')

def load_heart_ref_region(path):
    print('Loading heart reference..')
    cleaned_heart_ref = sc.read(path)
    return cleaned_heart_ref


def load_heart_region_visium(region):
    print('Loading visium data..')
    prefix1 = Path("adult_endo/visium/heart/heart-imperial")
    sample_names = {'ventricle': ['RTR02-HHH4-LV', 'RTR02-HHH2-LV-2-2',
                                  'RTR02-HHH2-LV-2-1', 'RTR02-HHH2-LV-1-2',
                                  'RTR02-HHH2-LV-1-1'],
                    'atrium': ['RTR02-HHH2-RA', 'RTR02-HHH4-LA', 'RTR02-HHH1-LA']}
    region_sample_names = sample_names[region]
    print(f'For heart {region}, these samples will be loaded..', region_sample_names)
    slides = []
    for sample in region_sample_names:
        slide = read_and_qc(prefix1 / sample / 'outs', sample, hasMT=True)
        slides.append(slide)
    if len(slides) > 1:
        adata_vis = slides[0].concatenate(
            slides[1:],
            batch_key="sample",
            uns_merge="unique",
            batch_categories=region_sample_names,
            index_unique=None
        )
        print('Visium data loaded.')
    else:
        adata_vis = slides[0]
    return adata_vis


def args():
    parser = ArgumentParser()
    parser.add_argument('--ref-path', dest='ref_path', type=str)
    parser.add_argument('--vis-region', dest='vis_region', type=str)
    parser.add_argument('--save-prefix', dest='save_prefix', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = args()
    vis_region = arguments.vis_region
    ref_path = arguments.ref_path
    trained_ref = load_heart_ref_region(arguments.ref_path)
    save_prefix = Path(arguments.save_prefix)
    trained_visium = train_cell2loc_model(trained_ref,
                                          load_heart_region_visium(vis_region),
                                          ncells_per_location=7,
                                          detection_alpha=20,
                                          write_prefix=save_prefix,
                                          batch_key='sample')
