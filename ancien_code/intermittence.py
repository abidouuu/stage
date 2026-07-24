import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Cible les alertes de backend
warnings.filterwarnings("ignore")

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import numpy as np
import matplotlib
matplotlib.use("Agg") # Configuration du backend avant plt
import matplotlib.pyplot as plt
import random
import os
from simu import config
from tqdm import tqdm
from itertools import product


epsilons=[-0.1, 0, 0.1]
kappas=[0,0.01,0.1, 1]
Lambdas=[0.01,0.1]
taus=[1e4,1e5]
deltas=[1e-4,1e-5]
inter_list=[(True,False), (False,True), (True, True)]
workdir = os.path.dirname(os.path.abspath(__file__))
datadir=os.path.join(workdir, "data/intermittence")

def simu_intermittence(show=False):
    for Lambda in tqdm(Lambdas, desc="Lambdas", position=0, leave=False):
        folder_lambda=os.path.join(datadir, "Lambda="+str(Lambda))
        for epsilon in tqdm(epsilons, desc="epsilons", position=1, leave=False):
            folder_epsilon=os.path.join(folder_lambda,"epsilon="+str(epsilon))
            minimas_list_TrueTrue=[]
            minimas_list_TrueFalse=[]
            minimas_list_FalseTrue=[]
            tauepsilon=random.choice(taus)
            taukappa=random.choice(taus)
            deltaepsilon=random.choice(deltas)
            deltakappa=random.choice(deltas)
            for kappa in tqdm(kappas,desc="kappas", position=2, leave=False):
                folder_kappa=os.path.join(folder_epsilon, "kappa="+str(kappa))
                for (inter_kappa,inter_epsilon) in inter_list : 
                    cfg=config(datadir=folder_kappa, term='mid', epsiloneq=epsilon, 
                        Lambda=Lambda,kappaeq=kappa, inter_kappa=inter_kappa,inter_epsilon=inter_epsilon,
                        tfin=1000, simu_title="kappa_"+str(inter_kappa)+"_epsilon_"+str(inter_epsilon))
                    (B_eq,b_eq)=cfg.get_eq()[0]
                    cfg.B0=B_eq
                    cfg.b0=b_eq  
                    cfg.taukappa=taukappa
                    cfg.deltakappa=deltakappa
                    cfg.tauepsilon=tauepsilon
                    cfg.deltaepsilon=deltaepsilon
                    if not (inter_kappa or inter_epsilon) : 
                        raise ValueError("Pas de variation stochastique (inter_kappa et inter_epsilon = False)...")
                    data=cfg.run(save=False)
                    cfg.write_config_file()
                    minimas=cfg.stat_analysis(data)
                    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 10),sharex=True)

                    first = True
                    t=data[:,0]
                    B=data[:,1]
                    b=data[:,2]
                    if inter_kappa : kappa_data=data[:,3]
                    if inter_epsilon : epsilon_data=data[:,4]

                    ax1.plot(t, B,color='blue',lw=1.5,label=r"$B(t)$")
                    ax1.plot(t, b,color='orange',lw=1.5,label=r"$b(t)$")
                    if inter_kappa : ax2.plot(t, kappa_data,color='red',lw=1.5,label=r"$\kappa(t)$")
                    if inter_epsilon : ax2.plot(t, epsilon_data,color='green',lw=1.5,label=r"$\varepsilon(t)$")
                    ax1.set_ylabel("Magnetic Amplitudes")
                    if inter_epsilon and not inter_kappa : ax2.set_ylabel("Dynamo Number")
                    elif inter_kappa and not inter_epsilon : ax2.set_ylabel("Coupling factor")
                    else : ax2.set_ylabel("Stochastic parameters")
                    ax1.grid(True)
                    ax2.grid(True)
                    ax1.legend(fontsize=8)
                    ax2.legend(fontsize=8)

                    fig.tight_layout()
                    savefile_eps = os.path.join(cfg.folder, f"plot.eps")
                    savefile_png = os.path.join(cfg.folder, f"plot.png")
                    plt.savefig(savefile_eps)
                    plt.savefig(savefile_png)
                    if show : plt.show()
                    plt.close(fig)

                    if (inter_kappa, inter_epsilon)==(True, True):
                        minimas_list_TrueTrue.append(minimas)
                    if (inter_kappa, inter_epsilon)==(False, True):
                        minimas_list_FalseTrue.append(minimas)
                    if (inter_kappa, inter_epsilon)==(True, False):
                        minimas_list_TrueFalse.append(minimas)
                    cfg.write_stat_file(minimas)
            cfg.plot_histograms(minimas_list=minimas_list_TrueTrue,differentfolder=folder_epsilon, name="minimas_True_True.png")
            cfg.plot_histograms(minimas_list=minimas_list_FalseTrue,differentfolder=folder_epsilon, name="minimas_False_True.png")
            cfg.plot_histograms(minimas_list=minimas_list_TrueFalse,differentfolder=folder_epsilon, name="minimas_True_False.png")

simu_intermittence()

