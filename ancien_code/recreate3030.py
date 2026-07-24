import numpy as np
import matplotlib.pyplot as plt
import os
from simu import config
from tqdm import tqdm

workdir = os.path.dirname(os.path.abspath(__file__))
datadir_recreate=os.path.join(workdir, "data/recreate3030")

#4 solutions court termes triviales, comparées aux analytiques
def fig8():
    datadir_fig8 = os.path.join(datadir_recreate, "5.1.1_Figure_8")
    for i in tqdm(range(10),desc="Figure_8"):
        name=f"plot_Bb_{str(i+1)}.eps"
        cfg = config(datadir=datadir_fig8, tfin=150, term='short', simu_title="Figure_8.")
        cfg.plot_time(cfg.run(save=True), type="Bb", eq=True, show=True, name=name)

#analyse de stabilité de la double solution
def fig9():
    datadir_fig9 = os.path.join(datadir_recreate, "5.1.2_Figure_9")
    count_cfg = 0
    count_tries=0
    with tqdm(total=4, desc="Figure_9", unit="cfg") as pbar:
        while count_cfg < 4:
            cfg = config(datadir=datadir_fig9, tfin=200, term='short', simu_title="Figure_9.")
            eq=cfg.get_eq()
            if (not eq==None) and len(eq) == 2:
                cfg.B0, cfg.b0 = eq[1]
                cfg.plot_time(cfg.run(save=True), type="Bb", eq=True, show=False)
                count_cfg += 1
                pbar.update(1)
            else : 
                cfg.delete_folder()
                count_tries+=1
                if count_tries==4000:
                    print("4000 tries brother stop")
                    return None

#B_eq, b_eq en fonction du nombre Dynamo
def fig10(simulate=False):
    datadir_fig10 = os.path.join(datadir_recreate, "5.1.3_Figure 10")
    epsilons = np.linspace(-5, 5, 2000)
    for i in tqdm(range(4), desc="Figure_10"):
        cfg = config(datadir=datadir_fig10, tfin=500, term='short', simu_title="Figure_10.")
        B_sim, b_sim = [], []
        B_analytic, b_analytic = [],[]
        B_landau=[]
        summary = []
        summary_analytic = []
        folder=cfg.folder
        config_in=cfg.config_in
        for epsilon in tqdm(epsilons, desc="Figure 10."+str(i)):
            cfg.epsiloneq = epsilon
            eq=cfg.get_eq()
            if simulate:
                data = cfg.run(save=False)
                B_final=data[-1, 1]
                b_final=data[-1, 2]
            if not eq==None:
                (B_final_analytic, b_final_analytic) = eq[0]
                cfg.delete_folder()
            else: (B_final_analytic, b_final_analytic) = (0,0)
            if epsilon>0:
                B_landau.append(np.sqrt(epsilon/cfg.Lambda)) 
            else : B_landau.append(0)
            B_sim.append(B_final)
            b_sim.append(b_final)
            B_analytic.append(B_final_analytic)
            b_analytic.append(b_final_analytic)
            summary.append((epsilon, B_final, b_final))
            summary_analytic.append((epsilon,B_final_analytic,b_final_analytic))
        os.makedirs(folder,exist_ok=True)
        cfg.write_config_file(differentfile=config_in)
        summary_path = os.path.join(folder, "epsilon_scan_simulated.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("epsilon B_final b_final\n")
            for epsilon_val, B_val, b_val in summary:
                f.write(f"{epsilon_val} {B_val} {b_val}\n")
        
        summary_path = os.path.join(folder, "epsilon_scan_analytic.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("epsilon B_final b_final\n")
            for epsilon_val, B_val, b_val in summary_analytic:
                f.write(f"{epsilon_val} {B_val} {b_val}\n")

        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.tight_layout()
        ax.plot(epsilons, B_sim, color='blue', linestyle='-', lw=1.5, label=r"$B_{eq}(\epsilon)$")
        ax.plot(epsilons, b_sim, color='orange', linestyle='-', lw=1.5, label=r"$b_{eq}(\epsilon)$")
        ax.plot(epsilons, B_landau, color='green', linestyle='--', lw=1.5, label=r"$B_{\text{Landau}}$")
        ax.set_xlabel("t")
        ax.grid(True)
        ax.legend(fontsize=8)
        savefile = os.path.join(folder, "epsilon_simulated.png")
        plt.savefig(savefile)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.tight_layout()
        ax.plot(epsilons, B_analytic, color='blue', linestyle='-', lw=1.5, label=r"$B_{eq}(\epsilon)$")
        ax.plot(epsilons, b_analytic, color='orange', linestyle='-', lw=1.5, label=r"$b_{eq}(\epsilon)$")
        ax.plot(epsilons, B_landau, color='green', linestyle='--', lw=1.5, label=r"$B_{\text{Landau}}$")
        ax.set_xlabel("t")
        ax.grid(True)
        ax.legend(fontsize=8)
        savefile = os.path.join(folder, "epsilon_analytic.png")
        plt.savefig(savefile)
        plt.close(fig)

#moyen terme, 4 solutions triviales
def fig12():
    datadir_fig12 = os.path.join(datadir_recreate, "5.2.1_Figure_12")
    for i in tqdm(range(8),desc="Figure_12"):
        cfg = config(datadir=datadir_fig12, tfin=10000, term='mid', simu_title="Figure_12.")
        cfg_data = cfg.run()
        cfg.plot_time(cfg_data, type="Bb", eq=False, show=False)
        cfg.plot_time(cfg_data, type="epsilon", show=False)
        cfg.plot_time(cfg_data, type="kappa", show=False)

#long terme, trois solutions triviales
def fig14():
    datadir_fig14=os.path.join(datadir_recreate,"5.3_Figure_14")
    nus=[10**(-2),10**(-3),10**(-4),10**(-5)]
    for i in tqdm(range(3), desc="Figure_14"):
        cfg=config(datadir=datadir_fig14, tfin=150000, term='long', simu_title="Figure_14.")
        folder=cfg.folder
        vals=[[],[],[]] #B,b,epsilon
        for nu in tqdm(nus,desc="Figure 14."+str(i+1)):
            cfg.nu=nu
            data=cfg.run(save=False)
            vals[0].append(data[:,1])
            vals[1].append(data[:,2])
            vals[2].append(data[:,5])
        t=data[:,0]
        cfg.delete_folder()
        os.makedirs(folder,exist_ok=True)
        cfg.write_config_file()
        for idx, val in enumerate(vals) :
            fig, ax = plt.subplots(figsize=(10,5))
            fig.tight_layout()
            for x in val :
                ax.plot(t,x,linestyle='-',lw=1.5,label=r"$\nu=$"+str(nu))
            ax.set_xlabel("t")
            ylabel="Large scale magnetic amplitude" if idx==0 else "Small scale magnetic amplitude" if idx==1 else "Dynamo number" if idx==2 else ""
            ax.set_ylabel(ylabel)
            title="B" if idx==0 else "b" if idx==1 else "epsilon" if idx==2 else ""
            ax.grid(True)
            ax.legend(fontsize=8)
            savefile = os.path.join(folder, title+".png")
            plt.savefig(savefile)
            plt.close(fig)
        cfg.delete_folder()

fig8()