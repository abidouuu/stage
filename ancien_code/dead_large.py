import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from simu import config
from tqdm import tqdm

n = 10000
epsilons = np.logspace(-5, 0, num=n)
kappas = np.logspace(-5, 1, n)
Lambdas = [1e-3, 1e-2, 1e-1, 1]
workdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(workdir, "data/dead_large")
os.makedirs(datadir, exist_ok=True)

TOL = 1e-15  # tolerance for "B_eq == 0"

# meshgrid for pcolormesh: X, Y have shape (len(kappas), len(epsilons))
X, Y = np.meshgrid(epsilons, kappas)

for Lambda in tqdm(Lambdas, desc="Lambdas", position=0, leave=False):
    # rows -> kappa (y-axis), cols -> epsilon (x-axis)
    mask = np.zeros((len(kappas), len(epsilons)), dtype=bool)
    cfg = config(datadir=datadir, term='short', Lambda=Lambda)

    for i, epsilon in enumerate(tqdm(epsilons, desc="epsilon", position=1, leave=False)):
        for j, kappa in enumerate(kappas):
            cfg.epsiloneq = -epsilon
            cfg.kappaeq = kappa
            B_eq, b_eq = cfg.get_eq()[0]
            mask[j, i] = np.isclose(B_eq, 0.0, atol=TOL)

    # 0 = white background, 1 = gray zone
    img = mask.astype(float)  # shape (n_kappa, n_epsilon)

    cmap = matplotlib.colors.ListedColormap(['white', '0.75'])  # white, light gray
    bounds = [-0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # pcolormesh plots each cell at its real (X, Y) coordinate,
    # so it works correctly on log-spaced (non-uniform-in-linear-space) grids
    ax.pcolormesh(X, Y, img, cmap=cmap, norm=norm, shading='auto')

    ax.set_xlabel(r'$-\epsilon$')
    ax.set_ylabel(r'$\kappa$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(fr'$B_{{eq}}=0$ region — $\Lambda={Lambda}$')

    fname = os.path.join(datadir, f"Beq_zero_Lambda_{Lambda}.png")
    fig.savefig(fname, dpi=150, facecolor='white')  # no transparent=True
    plt.close(fig)