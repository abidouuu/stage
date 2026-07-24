import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import random
from typing import Literal
from tqdm import tqdm
from itertools import product
import statistics as stat
from plot_style import PLOT_STYLE, _OKABE_ITO

#configuration : classe de paramètres. par défaut short term 
class config:
    #initialisation
    def __init__(
        self,
        workdir=os.path.dirname(os.path.abspath(__file__)),
        datadir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        term='short',
        simu_title=None,
        B0=None,
        b0=None,
        epsiloneq=None,
        Lambda=None, 
        kappaeq=None,
        tauepsilon=None, 
        taukappa=None, 
        deltaepsilon=None,
        deltakappa=None, 
        inter_kappa=None,
        inter_epsilon=None, 
        dt=0.01,
        tfin=None,
        run_index=None
        ):
            self.workdir = workdir
            # prefer an existing executable (dynamo.exe on Windows, dynamo elsewhere)
            exe_candidates = [os.path.join(workdir, "dynamo.exe"), os.path.join(workdir, "dynamo")]
            self.exec_path = None
            for p in exe_candidates:
                if os.path.exists(p):
                    self.exec_path = p
                    break
            # fallback to the expected name if none found (will raise a clear error later)
            if self.exec_path is None:
                self.exec_path = os.path.join(workdir, "dynamo.exe")
            self.datadir = datadir #dossier dans lequel sont stockées toutes les simulations

            # initialize randomized defaults when not provided
            if B0 is None :
                B0=10**random.uniform(-3, 1)
            if b0 is None :
                b0=10**random.uniform(-3, 1)
            if epsiloneq is None:
                epsiloneq = random.choice([-0.1,-0.01,0,0.01,0.1])
            if Lambda is None:
                Lambda = random.choice([0.1,1])
            if kappaeq is None:
                kappaeq = random.choice([0.01,0.1,1])
            if tauepsilon  is None:
                tauepsilon = 10**4
            if taukappa is None:
                taukappa = random.choice([10**3,10**4])
            if deltaepsilon is None:
                deltaepsilon = random.choice([10**(-4),10**(-5)])
            if deltakappa is None:
                deltakappa = random.choice([10**(-4),10**(-5)])
            if inter_kappa is None :
                inter_kappa=True
            if inter_epsilon is None : 
                inter_epsilon=True
            if simu_title is None :
                simu_title="simu_"
            if tfin is None : 
                if term=='short' : tfin=400
                elif term=='mid' : tfin=2e5
                elif term=='long' : tfin=1e6
                else:tfin=40

            if run_index is not None:
                self.folder = os.path.join(self.datadir, simu_title + str(run_index))
            else:
                simu_number = 1
                while os.path.exists(os.path.join(self.datadir, simu_title + str(simu_number))):
                    simu_number += 1
                self.folder = os.path.join(self.datadir, simu_title + str(simu_number))
            os.makedirs(self.folder, exist_ok=True)
            self.config_in = os.path.join(self.folder,"configuration.in")
            self.stat_file = os.path.join(self.folder,"minimas.txt")

            self.B0=B0
            self.b0=b0

            self.epsiloneq= epsiloneq
            self.Lambda=Lambda
            self.kappaeq=kappaeq

            self.tauepsilon=tauepsilon
            self.deltaepsilon=deltaepsilon
            self.taukappa=taukappa
            self.deltakappa=deltakappa

            self.inter_kappa=inter_kappa
            self.inter_epsilon=inter_epsilon

            self.dt=dt
            self.tfin=tfin
            self.term = term
            if term == 'short':
                self.tauepsilon = 0
                self.taukappa = 0
                self.deltaepsilon = 0
                self.deltakappa = 0

    #solutions analytiques stationnaires
    def get_eq(self, skuma=False):
        epsilon = self.epsiloneq if skuma==False else 1
        Lambda = self.Lambda
        kappa = self.kappaeq

        tol=1e-15 # x=0--> abs(x)<tol ; x<0--> x<-tol ; x>0--> x>tol
        div=1e15

        c1=(kappa**3)-Lambda
        c2=(kappa**2)*epsilon + 2*Lambda
        c3=-Lambda
        Delta=c2**2-4*c1*c3

        sol=[]

        def landau():
            return [(0, 1)]
        
        def verif_cond(x):
            if (abs(kappa)>tol) :
                if (max(0,-epsilon/kappa)-x<-tol) and (x-1<-tol) and x>0 and abs(x)<div : 
                    b_eq=sqrt(x)
                    B_eq=sqrt((epsilon+kappa*x)/Lambda)
                    sol.append((B_eq,b_eq))
        if abs(kappa)<tol : 
            if epsilon> tol : 
                sol=[(sqrt(epsilon/Lambda),1)]
            else : sol=landau()
        elif abs(c1)<tol : #pour éviter de diviser par 0 plus tard
                if abs(c2)<tol : sol=landau()
                else : 
                    x=-c3/c2
                    verif_cond(x)
        elif Delta<-tol : sol=landau()
        elif abs(Delta)<tol : 
            x=-c2/(2*c1)
            verif_cond(x)
        else : 
            xplus=(-c2+sqrt(Delta))/(2*c1)
            verif_cond(xplus)
            xmoins=(-c2-sqrt(Delta))/(2*c1)
            verif_cond(xmoins)
        if len(sol)==0 : sol=landau()
        return sol
        
    #texte de configuration.in (factorisé pour pouvoir être comparé à un fichier existant)
    def _config_text(self, Beq, beq):
        return f"""
            B0 = {self.B0}
            b0 = {self.b0}

            epsiloneq = {self.epsiloneq}
            Lambda = {self.Lambda}
            kappaeq = {self.kappaeq}

            tauepsilon = {self.tauepsilon}
            taukappa = {self.taukappa}
            deltaepsilon = {self.deltaepsilon}
            deltakappa = {self.deltakappa}

            inter_kappa = {'true' if self.inter_kappa else 'false'}
            inter_epsilon = {'true' if self.inter_epsilon else 'false'}

            term = {self.term}

            dt = {self.dt}
            tfin = {self.tfin}

            Beq = {Beq}
            beq = {beq}
        """.lstrip()

    #écrire configuration.in
    def write_config_file(self, differentfile=None):
        eq=self.get_eq()
        (Beq,beq)=eq[0]
        text = self._config_text(Beq, beq)
        target_file = self.config_in if differentfile is None else differentfile
        target_dir = os.path.dirname(target_file)
        os.makedirs(target_dir, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(text)

    def delete_folder(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    #lanceur de simulation de base, mettre le output avec configuration.in
    def run(self, outputname='output.txt', save=False):
        data_path = os.path.join(self.folder, outputname)

        # Compute the dt a fresh run would use BEFORE deciding whether to
        # trust any cached output, so the freshness check below compares
        # like with like.
        self.dt=min(self.dt, 0.01/max(self.epsiloneq, self.Lambda, self.kappaeq))

        # Only reuse a cached output.txt if it was produced with the exact
        # same configuration (same configuration.in content) as the current
        # parameters. Otherwise the folder holds stale data from an earlier
        # experiment (different tfin/dt/etc.) and must be regenerated.
        # Blindly trusting any pre-existing file here was the root cause of
        # run_and_avg's "inconsistent shapes" error: some run_index folders
        # kept old-length data from a previous configuration while others
        # were freshly computed with new parameters.
        if os.path.exists(data_path) and os.path.exists(self.config_in):
            eq = self.get_eq()
            (Beq, beq) = eq[0]
            expected_config = self._config_text(Beq, beq)
            with open(self.config_in, "r", encoding="utf-8") as f:
                existing_config = f.read()
            if existing_config == expected_config:
                return np.genfromtxt(data_path)
            # stale cache: fall through and regenerate

        os.makedirs(self.folder, exist_ok=True)
        self.write_config_file()
        # The C++ program expects the directory containing configuration.in as argv[1].
        # Pass `datadir` (not the full config file path).
        if not os.path.exists(self.exec_path):
            raise FileNotFoundError(f"Executable not found: {self.exec_path}\nCompile the C++ code first (e.g. run `make` in {self.workdir}).")
        subprocess.run([self.exec_path, self.folder], cwd=self.workdir, check=True)
        try:
            data = np.genfromtxt(data_path)
            return data
        finally:
            if not save:
                self.delete_folder()
    
    def run_and_avg(self, n=10, save_all=False, save_figs=False):
        data_runs = []
        self.write_config_file()
        for i in range(n):
            cfg = config(
                workdir=self.workdir,
                datadir=self.folder,
                term=self.term,
                B0=self.B0,
                b0=self.b0,
                epsiloneq=self.epsiloneq,
                Lambda=self.Lambda,
                kappaeq=self.kappaeq,
                tauepsilon=self.tauepsilon,
                taukappa=self.taukappa,
                deltaepsilon=self.deltaepsilon,
                deltakappa=self.deltakappa,
                inter_kappa=self.inter_kappa,
                inter_epsilon=self.inter_epsilon,
                dt=self.dt,
                tfin=self.tfin,
                run_index=i + 1,
            )
            data = cfg.run(save=save_all)
            if save_figs : 
                for type in ['Bb', 'kappa', 'epsilon', 'Omega']:
                            cfg.plot_time(data, type=type, show=False)
            data_runs.append(data)
        if len(data_runs) == 0:
            raise ValueError("No simulation data to average")

        shapes = {d.shape for d in data_runs}
        if len(shapes) != 1:
            raise ValueError(f"Simulation runs returned inconsistent shapes: {shapes}")

        first_data = data_runs[0]
        if first_data.ndim != 2 or first_data.shape[1] < 2:
            raise ValueError("Simulation data must have at least two columns (t and one variable)")

        t = first_data[:, 0]
        values_stack = np.stack([d[:, 1:] for d in data_runs], axis=-1)
        averaged_values = np.mean(values_stack, axis=-1)
        stddevs = np.std(values_stack,axis=-1)
        data_avg = np.column_stack((t, averaged_values))
        return data_runs, data_avg, stddevs
        
    possible_plots = Literal["Bb", "B", "b", "epsilon", "kappa", "Omega"]
    def plot_time(self, data, type: possible_plots,
              eq=False,
              show=True,
              name=None,
              minimas=None,
              eps=False,
              differentfolder=None,
              stddevs=None):

        t = data[:, 0]
        if stddevs is not None : 
            t_ini_Gyr = 1
            t_fin_Gyr = 10
            t_sun_Gyr = 4.6
            t = t_ini_Gyr + (t_fin_Gyr-t_ini_Gyr)*(t/self.tfin)

            B_stddev=stddevs[:,1]
            b_stddevs=stddevs[:,2]
        B = data[:, 1]
        b = data[:, 2]

        ncols = data.shape[1]

        kappa = None
        epsilon = None
        Omega = None

        if ncols > 3:
            kappa = data[:, 3]

        if ncols > 4:
            epsilon = data[:, 4]

        if ncols > 5:
            Omega = data[:, 5]

        if type in ["epsilon", "kappa"] and ncols <= 3:
            raise ValueError(
                f"Simulation {self.term} terme incompatible avec le plot {type}"
            )

        fig, ax = plt.subplots(
            figsize=PLOT_STYLE["figsize"]
        )

        is_B = type in ["Bb", "B"]
        is_b = type in ["Bb", "b"]
        if is_B or is_b : ax.set_ylabel("Magnetic amplitudes")

        # ============================================================
        # Courbes principales
        # ============================================================

        if is_B:
            ax.plot(
                t, B,
                color=PLOT_STYLE["color_B"],
                linestyle='-',
                lw=PLOT_STYLE["linewidth"],
                label=r"$B(t)$"
            )

            if stddevs is not None : 
                ax.fill_between(t,B-B_stddev,B+b_stddevs, color=_OKABE_ITO["sky_blue"])

        if is_b:
            ax.plot(
                t, b,
                color=PLOT_STYLE["color_b"],
                linestyle='-',
                lw=PLOT_STYLE["linewidth"],
                label=r"$b(t)$"
            )

            if stddevs is not None : 
                            ax.fill_between(t,b-b_stddevs,b+B_stddev, color=_OKABE_ITO["sky_orange"])

        # ============================================================
        # Solutions d'équilibre
        # ============================================================

        if eq:

            eq_solutions = self.get_eq()

            if eq_solutions is not None:

                for idx, (Beq, beq) in enumerate(eq_solutions):

                    sign = (
                        '+'
                        if len(eq_solutions) > 1 and idx == 0
                        else '-'
                        if len(eq_solutions) > 1
                        else ''
                    )

                    linestyle = '--' if idx == 0 else '-.'

                    if is_B:

                        labelB = (
                            rf"$B_{{eq}}^{{({sign})}}$"
                            if sign else
                            r"$B_{eq}$"
                        )

                        ax.plot(
                            t,
                            Beq * np.ones_like(t),
                            color=PLOT_STYLE["color_B"],
                            linestyle=linestyle,
                            lw=PLOT_STYLE["linewidth_eq"],
                            label=labelB
                        )

                    if is_b:

                        labelb = (
                            rf"$b_{{eq}}^{{({sign})}}$"
                            if sign else
                            r"$b_{eq}$"
                        )

                        ax.plot(
                            t,
                            beq * np.ones_like(t),
                            color=PLOT_STYLE["color_b"],
                            linestyle=linestyle,
                            lw=PLOT_STYLE["linewidth_eq"],
                            label=labelb
                        )

            else:

                ax.plot(
                    [0], [0],
                    color='none',
                    label="No equilibrium solution"
                )

        # ============================================================
        # Minima
        # ============================================================

        if (
            type in ["B", "Bb"]
            and self.term == 'mid'
            and minimas is not None
        ):

            for centre, duree in minimas:

                ax.axvline(
                    x=centre - 0.5 * duree,
                    color=PLOT_STYLE["color_minima"],
                    linestyle='--',
                    lw=PLOT_STYLE["linewidth"]
                )

                ax.axvline(
                    x=centre + 0.5 * duree,
                    color=PLOT_STYLE["color_minima"],
                    linestyle='--',
                    lw=PLOT_STYLE["linewidth"]
                )

        # ============================================================
        # Paramètres stochastiques
        # ============================================================

        if (
            type == "epsilon"
            and self.inter_epsilon
            and self.term != "short"
        ):

            ax.plot(
                t,
                epsilon,
                color=PLOT_STYLE["color_epsilon"],
                lw=PLOT_STYLE["linewidth"],
                label=r"$\varepsilon(t)$"
            )

            ax.set_ylabel("Dynamo number")

        if (
            type == "kappa"
            and self.term != "short"
        ):

            ax.plot(
                t,
                kappa,
                color=PLOT_STYLE["color_kappa"],
                lw=PLOT_STYLE["linewidth"],
                label=r"$\kappa(t)$"
            )

            ax.set_ylabel("Coupling factor")

        if (
            type == "Omega"
            and self.term == "long"
        ):

            ax.plot(
                t,
                Omega,
                color=PLOT_STYLE["color_omega"],
                lw=PLOT_STYLE["linewidth"],
                label=r"$\Omega(t)$"
            )

            ax.set_ylabel("Rotation speed")

        # ============================================================
        # Epsilon en fond, sur son propre axe (uniquement quand on a des
        # stddevs, i.e. les tracés type "skumanich", et pas sur le plot
        # d'epsilon lui-même pour éviter la redondance)
        # ============================================================

        ax_eps = None

        if (
            stddevs is not None
            and type != "epsilon"
            and epsilon is not None
        ):

            ax_eps = ax.twinx()

            ax_eps.plot(
                t,
                epsilon,
                color="green",
                linestyle=':',
                lw=PLOT_STYLE["linewidth"],
                label=r"$\varepsilon(t)$"
            )

            ax_eps.set_ylabel("Dynamo number", color="green")
            ax_eps.tick_params(axis='y', labelcolor="green")

        # ============================================================
        # Style global
        # ============================================================

        if stddevs is not None :
            ax.set_xlabel("Age (Gyr)")
            plt.axvline(x=t_sun_Gyr, color=_OKABE_ITO["black"], ls='--', 
                        linewidth=PLOT_STYLE["linewidth"],
                        label="Sun")
        else : 
            ax.set_xlabel(r"$\varepsilon \cdot t$")

        ax.tick_params(
            axis="both",
            labelsize=PLOT_STYLE["tick_fontsize"]
        )

        ax.grid(
            True,
            alpha=PLOT_STYLE["grid_alpha"]
        )

        handles, labels = ax.get_legend_handles_labels()

        if ax_eps is not None:
            handles_eps, labels_eps = ax_eps.get_legend_handles_labels()
            handles += handles_eps
            labels += labels_eps

        if handles:
            ax.legend(
                handles,
                labels,
                fontsize=PLOT_STYLE["legend_fontsize"]
            )

        fig.tight_layout()

        # ============================================================
        # Sauvegarde
        # ============================================================

        os.makedirs(self.folder, exist_ok=True)

        target_folder = self.folder if differentfolder is None else differentfolder

        if name is None:

            savefile_png = os.path.join(
                target_folder,
                f"plot_{type}.png"
            )

            if eps:
                savefile_eps = os.path.join(
                    target_folder,
                    f"plot_{type}.eps"
                )

        else:

            savefile_png = os.path.join(
                target_folder,
                f"{name}.png"
            )

            if eps:
                savefile_eps = os.path.join(
                    target_folder,
                    f"{name}.eps"
                )

        fig.savefig(
            savefile_png,
            dpi=PLOT_STYLE["dpi"]
        )

        if eps:
            fig.savefig(
                savefile_eps,
                dpi=PLOT_STYLE["dpi"]
            )

        if show:
            plt.show()

        plt.close(fig)

    def stat_analysis(self, data):
        t = data[:, 0]
        B = data[:, 1]
        eq=self.get_eq()
        (B_eq,b_eq)=eq[0]
        if B_eq==0 : B_eq=max(B)
        threshold=0.1*B_eq
        B_minimas=[x<threshold for x in B]

        minimas = []
        start = None

        for i, val in enumerate(B_minimas + [False]): 
            if (val) and (start is None) and (t[i]>10/max(abs(self.epsiloneq),0.01)):
                start = i
            elif (not val) and (start is not None):
                if ((t[i-1]-t[start])>200) :
                    minimas.append((t[start]+0.5*(t[i-1]-t[start]), t[i-1] - t[start]))
                    start = None
                else : start = None
        
        return minimas
    
    def plot_histograms(self, minimas_list, differentfolder=None, show=False, name=None,
                         labels=None, xlabel="Duration", density=False):
        
        if len(minimas_list) > 0 and isinstance(minimas_list[0], tuple):
            minimas_list = [minimas_list]

        groups = [np.array([duree for (_, duree) in minimas]) for minimas in minimas_list]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) == 0:
            return  

        fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])

        pooled = np.concatenate(groups)
        q75, q25 = np.percentile(pooled, [75, 25])
        iqr = q75 - q25
        if iqr > 0 and pooled.max() > pooled.min():
            bin_width = 2 * iqr / (len(pooled) ** (1 / 3))
            n_bins = int(np.clip(np.ceil((pooled.max() - pooled.min()) / bin_width), 8, 60))
        else:
            n_bins = 15
        bins = np.linspace(pooled.min(), pooled.max(), n_bins + 1)

        palette = PLOT_STYLE["palette"]

        for idx, lengths in enumerate(groups):
            color = palette[idx % len(palette)]
            label = labels[idx] if labels is not None else (
                f"Group {idx + 1}" if len(groups) > 1 else None
            )
            ax.hist(
                lengths, bins=bins, density=density,
                histtype="stepfilled", facecolor=color, edgecolor="none",
                alpha=0.35 if len(groups) > 1 else 0.55,
            )
            ax.hist(
                lengths, bins=bins, density=density,
                histtype="step", edgecolor=color,
                linewidth=PLOT_STYLE["linewidth_eq"], label=label,
            )

        ax.set_xlabel(xlabel, fontsize=PLOT_STYLE["label_fontsize"])
        ax.set_ylabel(
            "Probability density" if density else "Counts",
            fontsize=PLOT_STYLE["label_fontsize"]
        )
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=PLOT_STYLE["tick_fontsize"])

        handles, hlabels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=PLOT_STYLE["legend_fontsize"], frameon=False)

        fig.tight_layout()

        try:
            os.makedirs(self.folder, exist_ok=True)
        except Exception:
            pass

        target_folder = self.folder if differentfolder is None else differentfolder
        base = "plot_minimas" if name is None else os.path.splitext(name)[0]

        os.makedirs(target_folder, exist_ok=True)
        for ext in PLOT_STYLE["save_formats"]:
            fig.savefig(os.path.join(target_folder, f"{base}.{ext}"), dpi=PLOT_STYLE["dpi"])

        if show:
            plt.show()
        plt.close(fig)

    def write_stat_file(self,minimas,differentfile=None):
        text = f"{'No.':<5}{'Centre':<15}{'Durée':<15}"
        for idx, (centre, duree) in enumerate(minimas):
            text += f"\n{idx:<5}{centre:<15.6f}{duree:<15.6f}"
        target_file = self.stat_file if differentfile is None else differentfile
        target_dir = os.path.dirname(target_file)
        os.makedirs(target_dir, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(text)




if __name__ == "__main__":
    workdir = os.path.dirname(os.path.abspath(__file__))
    datadir=os.path.join(workdir, "data/tests")
    for epsilon in tqdm([0.1,0,-0.1]) :
        cfg=config(datadir=datadir, term='long', tfin=20000, simu_title="epsilon="+str(epsilon))
        cfg.epsiloneq=epsilon
        cfg.B0,cfg.b0=cfg.get_eq()[0]
        data=cfg.run(save=True)
        cfg.plot_time(data, type='Bb', show=False)
        cfg.plot_time(data,type='kappa', show=False)
        cfg.plot_time(data,type='epsilon', show=False)
        cfg.plot_time(data,type='Omega', show=False)