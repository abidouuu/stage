# Fichier général qui fait tous les plots de l'article. Une fonction par graphique. Harmonisation de l'esthétisme.

import numpy as np
import matplotlib.pyplot as plt
import os
from simu import config
from tqdm import tqdm
from math import sqrt
from scipy.signal import find_peaks
from matplotlib.cm import Blues, Oranges, Greens
import random
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency")
warnings.filterwarnings("ignore", message='Creating legend with loc="best" can be slow with large amounts of data.')
import logging
logging.getLogger('matplotlib.backends.backend_ps').setLevel(logging.ERROR)

workdir = os.path.dirname(os.path.abspath(__file__))
datadir=os.path.join(workdir, "data")

#--------------------------------------------------------------------------------------------------------
#-------------- FIGURES 1 & 2 - Solutions analytiques ---------------------------------------------------
#--------------------------------------------------------------------------------------------------------
datadir_short = os.path.join(datadir,"1_Court_terme_analytique")
datadir_fig1=os.path.join(datadir_short,"Figure_1_balaye_kappa")
datadir_fig2=os.path.join(datadir_short,"Figure_2_balaye_epsilon")
def court_terme(vary_epsilon): #x varie continûment, y quelques courbes.

    Lambdas=[0.001,0.01,0.1,1]
    epsilons_x=np.linspace(-3,3,10000)
    kappas_x=10**np.linspace(-3,2,10000)
    epsilons_y=[-0.1,-0.01,0,0.01, 0.1]
    kappas_y = 10.0**np.arange(-2, 2)

    if vary_epsilon : 
        xs=epsilons_x
        ys=kappas_y
        datadir = datadir_fig2
        print("Début de la Figure 2 !")
    else : 
        xs=kappas_x
        ys=epsilons_y
        datadir = datadir_fig1
        print("Début de la Figure 1 !")

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
        plt.close(fig)

def fig_1() : court_terme(False)
def fig_2() : court_terme(True)

#--------------------------------------------------------------------------------------------------------
#-------------- FIGURES 3,4,5,6 : Moyen terme, intermittence --------------------------------------------
#--------------------------------------------------------------------------------------------------------
datadir_mid=os.path.join(datadir, "2_Moyen_terme_Intermittence")
Lambdas=[0.01,0.1]
epsiloneqs=[-0.1,0,0.1]
kappaeqs=[0,0.01,0.1,1]
deltas=[1e-4,1e-5]
taus=[1e4,1e5]
inter_list=[(True,False), (False,True), (True, True)]

def plot_fft(t, signal, label, folder, filename_prefix):
    dt = t[1] - t[0]  # pas de temps, suppose échantillonnage uniforme
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(signal - signal.mean())  # on retire la moyenne (composante DC)
    amplitude = np.abs(fft_vals)

    fig_fft, ax_fft = plt.subplots(figsize=(10, 5))
    ax_fft.plot(freqs, amplitude, lw=1.5, color='purple')
    ax_fft.set_xlabel("Fréquence")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_yscale('log')
    ax_fft.set_title(f"FFT de {label}")
    ax_fft.grid(True)
    fig_fft.tight_layout()

    fig_fft.savefig(os.path.join(folder, f"{filename_prefix}_fft.eps"), bbox_inches="tight")
    fig_fft.savefig(os.path.join(folder, f"{filename_prefix}_fft.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_fft)

    # --- Trois fréquences dominantes (pics distincts) ---
    peaks, _ = find_peaks(amplitude)

    if len(peaks) == 0:
        # aucun pic local détecté (signal trop plat ou trop court) : on retombe sur le max global
        top_freqs = [freqs[np.argmax(amplitude)]]
        top_amps = [amplitude.max()]
    else:
        peak_amps = amplitude[peaks]
        peak_freqs = freqs[peaks]
        n_top = min(3, len(peaks))
        top_idx = np.argsort(peak_amps)[::-1][:n_top]
        top_freqs = peak_freqs[top_idx]
        top_amps = peak_amps[top_idx]

    with open(os.path.join(folder, f"{filename_prefix}_fft_top3.txt"), "w") as f:
        f.write(f"Fréquences dominantes de la FFT de {label}\n")
        f.write("="*50 + "\n")
        if len(peaks) == 0:
            f.write("Aucun pic local détecté, fréquence du maximum global reportée :\n")
        for rank, (freq, amp) in enumerate(zip(top_freqs, top_amps), start=1):
            f.write(f"{rank}. Fréquence = {freq:.6e}   |   Amplitude = {amp:.6e}\n")

def moyen_terme():
    for Lambda in tqdm(Lambdas, desc="Lambda"):
        Lambda_folder=os.path.join(datadir_mid, "Lambda="+str(Lambda))
        for epsiloneq in tqdm(epsiloneqs, desc="epsiloneqs", position=1, leave=False):
            epsiloneq_folder=os.path.join(Lambda_folder,"epsiloneq="+str(epsiloneq))
            minimas_list_TrueTrue=[]
            minimas_list_TrueFalse=[]
            minimas_list_FalseTrue=[]
            tauepsilon=random.choice(taus)
            taukappa=random.choice(taus)
            deltaepsilon=random.choice(deltas)
            deltakappa=random.choice(deltas)
            fig_3_B, ax_3_B = plt.subplots(figsize=(10,5))
            fig_3_B.tight_layout()
            fig_3_b, ax_3_b = plt.subplots(figsize=(10,5))
            fig_3_b.tight_layout()
            for kappaeq in tqdm(kappaeqs,desc="kappas", position=2, leave=False):
                folder_kappa=os.path.join(epsiloneq_folder, "kappaeq="+str(kappaeq))
                for (inter_kappa,inter_epsilon) in inter_list : 
                    cfg=config(datadir=folder_kappa, term='mid', simu_title="kappa_"+str(inter_kappa)+"_epsilon_"+str(inter_epsilon),
                               epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq, inter_kappa=inter_kappa,inter_epsilon=inter_epsilon,
                               taukappa=taukappa, deltakappa=deltakappa, tauepsilon=tauepsilon, deltaepsilon=deltaepsilon,
                               tfin=5e5)
                    (B_eq,b_eq)=cfg.get_eq()[0]
                    cfg.B0=B_eq
                    cfg.b0=b_eq  
                    data=cfg.run(save=False)
                    cfg.write_config_file()
                    minimas=cfg.stat_analysis(data)

                    t=data[:,0]
                    B=data[:,1]
                    b=data[:,2]
                    if inter_kappa:
                        kappa_data = data[:, 3]
                        plot_fft(t, kappa_data, r"$\kappa(t)$", cfg.folder, "kappa")

                    if inter_epsilon:
                        epsilon_data = data[:, 4]
                        plot_fft(t, epsilon_data, r"$\varepsilon(t)$", cfg.folder, "epsilon")
                    
                    if inter_epsilon and inter_kappa : 
                        ax_3_B.plot(t,B, lw=1.5, label=r"$\kappa_eq$="+str(kappaeq))
                        ax_3_b.plot(t,b,lw=1.5, label=r"$\kappa_eq$="+str(kappaeq))

                        fig_5, ax_5 = plt.subplots(figsize=(10, 5))

                        # fond blanc opaque (pas transparent)
                        fig_5.patch.set_facecolor('white')
                        ax_5.set_facecolor('white')

                        xlim = (epsilon_data.min(), epsilon_data.max())
                        ylim = (kappa_data.min(), kappa_data.max())
                        ax_5.set_xlim(xlim)
                        ax_5.set_ylim(ylim)

                        # zone grisée : kappa + epsilon < 0, i.e. kappa < -epsilon
                        x_fill = np.linspace(xlim[0], xlim[1], 500)
                        y_boundary = -x_fill
                        ax_5.fill_between(x_fill, ylim[0], y_boundary,
                                            color='0.85', zorder=0)

                        # trajectoire colorée selon B
                        points = np.array([epsilon_data, kappa_data]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)

                        norm = plt.Normalize(B.min(), B.max())
                        lc = LineCollection(segments, cmap='coolwarm', norm=norm, zorder=2)
                        lc.set_array(B[:-1])  # valeur de B associée à chaque segment
                        lc.set_linewidth(1.5)
                        line = ax_5.add_collection(lc)

                        cbar = fig_5.colorbar(line, ax=ax_5)
                        cbar.set_label(r"$B$")

                        ax_5.set_xlabel(r"$\varepsilon(t)$")
                        ax_5.set_ylabel(r"$\kappa(t)$")
                        ax_5.grid(True, zorder=1, alpha=0.3)

                        fig_5.tight_layout()

                        fig_5.savefig(os.path.join(cfg.folder,f"Fig_5_trajectoire.png"))

                        plt.close(fig_5)
                    
                    fig_4_B_b,ax_4_B_b=plt.subplots(figsize=(10,5))
                    fig_4_stat,ax_4_stat=plt.subplots(figsize=(10,5))

                    ax_4_B_b.plot(t, B,color='blue',lw=1.5,label=r"$B(t)$")
                    ax_4_B_b.plot(t, b,color='orange',lw=1.5,label=r"$b(t)$")
                    if inter_kappa : ax_4_stat.plot(t, kappa_data,color='red',lw=1.5,label=r"$\kappa(t)$")
                    if inter_epsilon : ax_4_stat.plot(t, epsilon_data,color='green',lw=1.5,label=r"$\varepsilon(t)$")
                    ax_4_B_b.set_ylabel("Magnetic Amplitudes")
                    if inter_epsilon and not inter_kappa : ax_4_stat.set_ylabel("Dynamo Number")
                    elif inter_kappa and not inter_epsilon : ax_4_stat.set_ylabel("Coupling factor")
                    else : ax_4_stat.set_ylabel("Stochastic parameters")
                    ax_4_B_b.grid(True)
                    ax_4_stat.grid(True)
                    ax_4_B_b.legend(fontsize=8)
                    ax_4_stat.legend(fontsize=8)

                    fig_4_B_b.tight_layout()
                    fig_4_stat.tight_layout()
                    savefile_eps = os.path.join(cfg.folder, f"Fig_4_B_b.eps")
                    savefile_png = os.path.join(cfg.folder, f"Fig_4_B_b.png")
                    fig_4_B_b.savefig(savefile_eps)
                    fig_4_B_b.savefig(savefile_png)
                    savefile_eps = os.path.join(cfg.folder, f"Fig_4_stat.eps")
                    savefile_png = os.path.join(cfg.folder, f"Fig_4_stat.png")
                    fig_4_stat.savefig(savefile_eps)
                    fig_4_stat.savefig(savefile_png)
                    plt.close(fig_4_B_b)
                    plt.close(fig_4_stat)

                    if (inter_kappa, inter_epsilon)==(True, True):
                        minimas_list_TrueTrue.append(minimas)
                    if (inter_kappa, inter_epsilon)==(False, True):
                        minimas_list_FalseTrue.append(minimas)
                    if (inter_kappa, inter_epsilon)==(True, False):
                        minimas_list_TrueFalse.append(minimas)
                    cfg.write_stat_file(minimas)

            cfg.plot_histograms(minimas_list=minimas_list_TrueTrue,differentfolder=epsiloneq_folder, name="Fig_6_minimas_True_True.png")
            cfg.plot_histograms(minimas_list=minimas_list_FalseTrue,differentfolder=epsiloneq_folder, name="Fig_6_minimas_False_True.png")
            cfg.plot_histograms(minimas_list=minimas_list_TrueFalse,differentfolder=epsiloneq_folder, name="Fig_6_minimas_True_False.png")      
            for ax in [ax_3_B,ax_3_b]:
                ax.set_xlabel("t")
                ax.grid(True)
                ax.legend(fontsize=8)
            Bsavefile_eps = os.path.join(epsiloneq_folder, "Fig_3_B_large.eps")
            Bsavefile_png = os.path.join(epsiloneq_folder, "Fig_3_B_large.png")
            bsavefile_eps = os.path.join(epsiloneq_folder, "Fig_3_b_small.eps")
            bsavefile_png = os.path.join(epsiloneq_folder, "Fig_3_b_small.png")

            fig_3_B.savefig(Bsavefile_eps, bbox_inches="tight")
            fig_3_B.savefig(Bsavefile_png, dpi=300, bbox_inches="tight")
            fig_3_b.savefig(bsavefile_eps, bbox_inches="tight")
            fig_3_b.savefig(bsavefile_png, dpi=300, bbox_inches="tight")

            plt.close(fig_3_B)
            plt.close(fig_3_b)

court_terme()
moyen_terme()