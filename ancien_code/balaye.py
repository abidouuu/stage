import numpy as np
import matplotlib.pyplot as plt
import os
from simu import config
from tqdm import tqdm
from math import sqrt
from matplotlib.cm import Blues, Oranges, Greens

workdir = os.path.dirname(os.path.abspath(__file__))
show=False

Lambdas=10.0**np.arange(-3,1)

epsilons_x=np.linspace(-3,3,10000)
kappas_x=10**np.linspace(-3,2,10000)
epsilons_y=[-0.1,-0.01,0,0.01, 0.1]
kappas_y = 10.0**np.arange(-2, 2)

def balaye(vary_epsilon): #x varie continûment, y quelques courbes
    if vary_epsilon : 
        xs=epsilons_x
        ys=kappas_y
        datadir = os.path.join(workdir, "data/balaye/epsilon")
    else : 
        xs=kappas_x
        ys=epsilons_y
        datadir = os.path.join(workdir, "data/balaye/kappa")

    for Lambda in tqdm(Lambdas, desc="Lambda", position=0) :
        cfg = config(datadir=datadir, simu_title="Lambda="+str(Lambda)+"_")
        cfg.Lambda=Lambda
        fig, (axB, axb) = plt.subplots(2, 1,figsize=(10, 10),sharex=True)

        if not vary_epsilon : 
            axB.set_xscale('log')
            axb.set_xscale('log')

        blues = Blues(np.linspace(0.35, 0.95, len(ys)))
        oranges = Oranges(np.linspace(0.35, 0.95, len(ys)))

        if  not vary_epsilon : 
            greens_bool=[a>0 for a in ys]
            greens = Greens(np.linspace(0.35,0.95,len(greens_bool)))

        first = True

        for y_idx, y in tqdm(enumerate(ys),total=len(ys),desc="y",position=1,leave=False):
            B = []
            b = []
            B_landau = []
            b_landau = []

            if vary_epsilon : cfg.kappaeq=y
            else : cfg.epsiloneq=y

            for x in tqdm(xs,desc="x",position=2,leave=False):
                if vary_epsilon : cfg.epsiloneq=x
                else : cfg.kappaeq=x

                (B_eq, b_eq) = cfg.get_eq()[0]
                B.append(B_eq)
                b.append(b_eq)
                if cfg.epsiloneq > 0:
                    B_landau.append(sqrt(cfg.epsiloneq / Lambda))
                else:
                    B_landau.append(0)
                b_landau.append(1)

            green_idx=0
            if vary_epsilon : 
                label=rf"$\kappa={y:.1e}$"
                axB.plot(xs, B_landau,color='green',ls="--",lw=1.5,label="B_landau" if first else None)
                axb.plot(xs, b_landau,color='green',ls="--",lw=1.5,label="b_landau" if first else None)
                first = False
            else : 
                label=rf"$\varepsilon={y:.1e}$"
                axb.plot(xs, b_landau,color='green',ls="--",lw=1.5,label="b_landau" if first else None)
                if ys[y_idx]>=0 : 
                    axB.plot(xs, B_landau,color=greens[green_idx],ls="--",lw=1.5,label="B_landau : " + label)
                    green_idx+=1

            axB.plot(xs, B,color=blues[y_idx],lw=1.5,label=label)
            axb.plot(xs, b,color=oranges[y_idx],lw=1.5,label=label)

        axB.set_ylabel("B")
        axB.grid(True)
        axB.legend(fontsize=8)

        if vary_epsilon : axb.set_xlabel(r"$\varepsilon$")
        else : axb.set_xlabel(r"$\kappa$")
        axb.set_ylabel("b")
        axb.grid(True)
        axb.legend(fontsize=8)

        fig.tight_layout()
        cfg.write_config_file()
        savefile_eps = os.path.join(cfg.folder, f"lambda={str(Lambda)}.eps")
        savefile_png = os.path.join(cfg.folder, f"lambda={str(Lambda)}.png")
        plt.savefig(savefile_eps)
        plt.savefig(savefile_png)
        if show : plt.show()
        plt.close(fig)

for x in [True, False] : balaye(x)