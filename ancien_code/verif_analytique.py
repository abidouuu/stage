import numpy as np
import matplotlib.pyplot as plt
import os
from simu import config
from tqdm import tqdm

workdir = os.path.dirname(os.path.abspath(__file__))
datadir=os.path.join(workdir, "data_merde/verif_analytique")

#vérifier l'unicité de la solution analytique
def verif_unique(): 
    for i in tqdm(range(10000)):
        cfgdatadir=os.path.join(datadir,"verif_unique")
        cfg=config(datadir=cfgdatadir, tfin=300, term='short', simu_title=str(i))
        cfg.delete_folder()
        if len(cfg.get_eq())==2 : 
            print("Double solution à la ", i, "ème simulation !")
            cfg.write_config_file()

#vérifier qu'on converge bien vers la solution prédite
def verif_correct():
    tol=1e-3
    for i in tqdm(range(10)):
        cfgdatadir=os.path.join(datadir,"verif_correct")
        cfg=config(datadir=cfgdatadir, tfin=300, term='short',  kappaeq=0, simu_title=str(i), inter_kappa=False)
        eq=cfg.get_eq()
        if len(eq)==1:
            (B_eq,b_eq)=eq[0]
        else : 
            print("Double solution détectée à la", i, "-ème simulation !")
            continue
        data=cfg.run()
        cfg.delete_folder()
        B_final=data[-1,1]
        b_final=data[-1,2]
        B_dif=abs(B_final-B_eq)
        b_dif=abs(b_final-b_eq)
        print(B_eq)
        cfg.plot_time(data, type="Bb", eq=True, show=True)
        if B_dif>tol or b_dif>tol : 
            cfg.tfin=3000
            data=cfg.run()
            cfg.delete_folder()
            B_final=data[-1,1]
            b_final=data[-1,2]
            B_dif=abs(B_final-B_eq)
            b_dif=abs(b_final-b_eq)
            if B_dif>tol or b_dif>tol : 
                print("\nNon-convergence à la ", i, "-ème simulation !")
                cfg.write_config_file()
                cfg.plot_time(data, type="Bb", eq=True, show=True)


verif_correct()