import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Cible les alertes de backend
warnings.filterwarnings("ignore")

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import numpy as np
import matplotlib
matplotlib.use("Agg") # Configuration du backend avant plt
import matplotlib.pyplot as plt
import os
from simu import config
from tqdm import tqdm
from itertools import product

workdir = os.path.dirname(os.path.abspath(__file__))
datadir=os.path.join(workdir, "data/skumanich")


def skumanich():
    for i in tqdm(range(5), desc="skumanich"):
        cfg=config(datadir=datadir, term='long', tfin=1e5)
        data=cfg.run(save=False)
        cfg.nu=0.05
        cfg.write_config_file()
        minimas=cfg.stat_analysis(data)
        cfg.write_stat_file(minimas)
        cfg.plot_time(data=data, type='Bb', show=False)
        cfg.plot_time(data=data, type='kappa', show=False)
        cfg.plot_time(data=data, type='epsilon', show=False)
        cfg.plot_time(data=data, type='Omega', show=False)

skumanich()