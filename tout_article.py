import numpy as np
import matplotlib.pyplot as plt
import os
import json
from simu import config
from tqdm import tqdm
from math import sqrt
import random
from scipy.signal import find_peaks
from matplotlib.cm import Blues, Oranges, Greens
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore", message="The PostScript backend does not support transparency")
warnings.filterwarnings("ignore", message='Creating legend with loc="best" can be slow with large amounts of data.')
import logging
logging.getLogger('matplotlib.backends.backend_ps').setLevel(logging.ERROR)
from plot_style import PLOT_STYLE, _OKABE_ITO

workdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(workdir, "data_new")


# ==========================================================================================================
# -------------- FONCTIONS UTILITAIRES ---------------------------------------------------------------------
# ==========================================================================================================

def new_figure(figsize=None, nrows=1, ncols=1, sharex=False):
    if figsize is None:
        figsize = PLOT_STYLE["figsize"]

    # layout="constrained" : ajuste automatiquement les marges pour que
    # titres/labels/legendes ne soient jamais coupés, SANS changer la
    # taille finale de la figure (contrairement à tight_layout() suivi
    # de bbox_inches="tight", qui recadre le fichier sauvegardé et fait
    # varier la taille de sortie d'une figure a l'autre). La sortie
    # reste donc toujours exactement figsize * dpi.
    return plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        layout="constrained"
    )

def style_axis(
        ax,
        xlabel=None,
        ylabel=None,
        title=None,
        legend=True,
        grid=True,
        xscale=None,
        yscale=None,
        legend_ncol=1,
        legend_loc="best"):

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=PLOT_STYLE["label_fontsize"])

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PLOT_STYLE["label_fontsize"])

    if title is not None:
        ax.set_title(title, fontsize=PLOT_STYLE["title_fontsize"])

    ax.tick_params(
        axis="both",
        labelsize=PLOT_STYLE["tick_fontsize"]
    )

    if grid:
        ax.grid(True, alpha=PLOT_STYLE["grid_alpha"])

    if xscale is not None:
        ax.set_xscale(xscale)

    if yscale is not None:
        ax.set_yscale(yscale)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if isinstance(legend_loc, str) and legend_loc.startswith("outside"):
                # Une légende "outside ..." doit être posée via fig.legend()
                # (pas ax.legend()) pour que layout="constrained" lui
                # réserve un vrai bloc d'espace dans la figure et ne la
                # laisse jamais déborder/être coupée au moment du save,
                # même si la figure est étroite ou la légende large.
                ax.figure.legend(
                    handles, labels,
                    fontsize=PLOT_STYLE["legend_fontsize"],
                    ncol=legend_ncol,
                    loc=legend_loc,
                    handlelength=1.5,
                    columnspacing=1.0,
                    handletextpad=0.5,
                )
            else:
                ax.legend(
                    fontsize=PLOT_STYLE["legend_fontsize"],
                    ncol=legend_ncol,
                    loc=legend_loc
                )

def style_axis_legend_inside_top(ax, ncol=1):
    """
    Place la légende à l'intérieur du cadre, centrée en haut,
    puis agrandit automatiquement la limite supérieure de y
    pour que la légende ne recouvre pas les courbes.
    """

    handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return

    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=ncol,
        fontsize=PLOT_STYLE["legend_fontsize"],
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.5,
        frameon=False,
        borderaxespad=0.0
    )

    # Force le calcul de la taille réelle de la légende
    fig = ax.figure
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()

    bbox = legend.get_window_extent(renderer)
    bbox_axes = bbox.transformed(ax.transAxes.inverted())

    # hauteur de la légende en coordonnées de l'axe
    legend_height = bbox_axes.height

    ymin, ymax = ax.get_ylim()
    data_range = ymax - ymin

    # On crée une zone vide en haut correspondant
    # à la hauteur de la légende + marge
    ax.set_ylim(
            ymin,
            ymax + 1.25 * legend_height * data_range
        )


def save_fig(fig, folder, filename,
             dpi=None,
             bbox_inches=None,
             close=True):
    # bbox_inches=None (et non "tight") : on veut que la taille de sortie
    # soit exactement figsize (celle passée à new_figure) pour TOUTES les
    # figures d'un même type, indépendamment de la longueur des légendes/
    # labels. "tight" recadre sur le contenu réel et fait varier la taille
    # finale d'une figure à l'autre du même type.

    if dpi is None:
        dpi = PLOT_STYLE["dpi"]

    os.makedirs(folder, exist_ok=True)

    base = os.path.join(folder, filename)

    for ext in PLOT_STYLE["save_formats"]:
        fig.savefig(
            base + "." + ext,
            dpi=dpi,
            bbox_inches=bbox_inches
        )

    if close:
        plt.close(fig)


def fmt_num(x):
    """Formate un nombre en notation décimale (0.01) plutôt que
    scientifique (1e-2), tout en évitant les zéros parasites."""
    s = f"{x:g}"
    return s

def find_empty_spot(ax, grid_size=25, margin=0.08, max_points_per_line=200, avoid_legend=True):
    """Cherche, dans les coordonnées d'axes (0-1), la position la plus
    éloignée de toutes les courbes déjà tracées sur `ax` (et, par
    défaut, de la légende si elle existe). Utile pour placer une
    annotation (texte) sans qu'elle chevauche une courbe ou la légende.

    Principe : on échantillonne une grille de points candidats, on
    calcule la distance minimale de chaque candidat à tous les points
    des courbes (dans le même repère normalisé que les axes, pour ne
    pas dépendre de l'échelle des données), et on garde le candidat
    dont cette distance minimale est la plus grande (= le point le
    plus "vide" du graphique). Les candidats tombant dans la zone de
    la légende sont exclus, sinon les deux "meilleurs coins" se
    disputent souvent le même endroit et le texte finit collé à la
    légende.
    """
    pts = []

    for line in ax.get_lines():
        xdata, ydata = line.get_xdata(), line.get_ydata()
        if len(xdata) == 0:
            continue

        # sous-échantillonnage pour rester rapide sur des courbes à
        # 10000 points
        step = max(1, len(xdata) // max_points_per_line)
        xdata, ydata = xdata[::step], ydata[::step]

        # passage en coordonnées d'axes (0 à 1), indépendant de
        # l'échelle (log ou linéaire) des données
        xy_disp = ax.transData.transform(np.column_stack([xdata, ydata]))
        xy_axes = ax.transAxes.inverted().transform(xy_disp)
        pts.append(xy_axes)

    if pts:
        pts = np.vstack(pts)
        # on ignore ce qui sort du cadre visible
        mask = (pts[:, 0] >= 0) & (pts[:, 0] <= 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= 1)
        pts = pts[mask]
    else:
        pts = np.empty((0, 2))

    xs = np.linspace(margin, 1 - margin, grid_size)
    ys = np.linspace(margin, 1 - margin, grid_size)
    gx, gy = np.meshgrid(xs, ys)
    candidates = np.column_stack([gx.ravel(), gy.ravel()])

    if len(pts) > 0:
        dists = np.sqrt(((candidates[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        min_dist = dists.min(axis=1)
    else:
        min_dist = np.full(len(candidates), np.inf)

    if avoid_legend:
        legend = ax.get_legend()
        if legend is not None:
            fig = ax.figure
            # un draw() est nécessaire pour que la légende ait une
            # position/étendue calculée
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox_disp = legend.get_window_extent(renderer)
            bbox_axes = ax.transAxes.inverted().transform(bbox_disp)
            (lx0, ly0), (lx1, ly1) = bbox_axes
            pad = 0.02
            inside_legend = (
                (candidates[:, 0] >= lx0 - pad) & (candidates[:, 0] <= lx1 + pad) &
                (candidates[:, 1] >= ly0 - pad) & (candidates[:, 1] <= ly1 + pad)
            )
            min_dist = min_dist.copy()
            min_dist[inside_legend] = -np.inf

    best = candidates[np.argmax(min_dist)]
    return best[0], best[1]


# ==========================================================================================================
# -------------- FIGURES 1 & 2 - Solutions analytiques -----------------------------------------------------
# ==========================================================================================================
datadir_short = os.path.join(datadir, "1_Court_terme_analytique")
datadir_fig1 = os.path.join(datadir_short, "Figure_1_balaye_kappa")
datadir_fig2 = os.path.join(datadir_short, "Figure_2_balaye_epsilon")

def best_label_position(xs, curve, other_curves, n_candidates=300):
    xs = np.asarray(xs)
    curve = np.asarray(curve)

    # candidate points (avoid edges)
    idx_candidates = np.linspace(
        int(0.1*len(xs)),
        int(0.9*len(xs)),
        n_candidates
    ).astype(int)

    scores = []

    for idx in idx_candidates:

        x = xs[idx]
        y = curve[idx]

        score = 0

        # -----------------------------------
        # 1) Distance from other curves
        # -----------------------------------
        min_dist = np.inf

        for other in other_curves:

            d = abs(y - other[idx])
            min_dist = min(min_dist, d)

        # reward empty space
        score += min_dist**2


        # -----------------------------------
        # 2) Prefer flat parts of curve
        # -----------------------------------
        if 1 < idx < len(xs)-2:
            slope = abs(
                (curve[idx+2]-curve[idx-2]) /
                (xs[idx+2]-xs[idx-2])
            )

            score -= 0.05*slope


        # -----------------------------------
        # 3) Avoid too extreme x positions
        # -----------------------------------
        edge_penalty = (
            abs(idx-len(xs)/2)/(len(xs)/2)
        )

        score -= 0.1*edge_penalty

        ymin, ymax = np.nanmin(curve), np.nanmax(curve)

        margin = 0.15*(ymax-ymin)

        if y + margin > ymax:
            score -= 100

        if y - margin < ymin:
            score -= 100

        scores.append(score)


    best = idx_candidates[np.argmax(scores)]

    xlab = xs[best]
    ylab = curve[best]


    # decide whether label goes above or below
    # depending on free space
    if best < len(xs)//2:
        side = 1
    else:
        side = -1

    return xlab, ylab, side

def balayage(vary_epsilon, show_lambda=True):  # x varie continûment, y quelques courbes.
    Lambdas = [0.001, 0.01, 0.1, 1]
    epsilons_x = np.linspace(-2, 2, 10000)
    kappas_x = 10 ** np.linspace(-3, 2, 10000)
    epsilons_y = [0.1, 0.01, 0, -0.01, -0.1]
    kappas_y = 10.0 ** np.arange(-2, 2)

    if vary_epsilon:
        xs = epsilons_x
        ys = kappas_y
        datadir_ = datadir_fig2
        print("Balayage de epsilon : start")
    else:
        xs = kappas_x
        ys = epsilons_y
        datadir_ = datadir_fig1
        print("Balayage de kappa : start")

    for Lambda in tqdm(Lambdas, desc="Lambda", position=0):
        cfg = config(datadir=datadir_, simu_title="Lambda=" + str(Lambda) + "_")
        cfg.Lambda = Lambda

        fig, (axB, axb) = new_figure(figsize=PLOT_STYLE["figsize_double_tall"], nrows=2, sharex=True)
        xscale = None if vary_epsilon else 'log'

        # gamme élargie (0.15 à 1.0) pour des couleurs mieux distinguables,
        # notamment entre les courbes voisines
        blues = Blues(np.linspace(0.35, 1.0, len(ys)))
        oranges = Oranges(np.linspace(0.35, 1.0, len(ys)))

        if not vary_epsilon:
            greens_bool = [a >= 0 for a in ys]
            greens = Greens(np.linspace(0.55, 1.0, len(greens_bool)))

        first = True
        # FIX: green_idx must be initialized once, OUTSIDE the y-loop, so that
        # each non-negative-epsilon curve gets a distinct color from the gradient
        # instead of every curve reusing greens[0].
        green_idx = 0

        landau_curves = []
        landau_colors = []
        positive_eps = []
        for y_idx, y in enumerate(ys):
            B, b = [], []
            B_landau, b_landau = [], []

            if vary_epsilon:
                cfg.kappaeq = y
            else:
                cfg.epsiloneq = y

            for x in xs:
                if vary_epsilon:
                    cfg.epsiloneq = x
                else:
                    cfg.kappaeq = x

                (B_eq, b_eq) = cfg.get_eq()[0]
                B.append(B_eq)
                b.append(b_eq)
                if cfg.epsiloneq > 0:
                    B_landau.append(sqrt(cfg.epsiloneq / Lambda))
                else:
                    B_landau.append(0)
                b_landau.append(1)

            if vary_epsilon:
                label = rf"$\kappa={fmt_num(y)}$"
                axB.plot(xs, B_landau, color='green', ls="--", lw=PLOT_STYLE["linewidth"], label=r"$B_\text{eq}$" if first else None)
                axb.plot(xs, b_landau, color='green', ls="--", lw=PLOT_STYLE["linewidth"], label=r"$b_\text{eq}$" if first else None)
                first = False
            else:
                label = rf"$\varepsilon={fmt_num(y)}$"
                axb.plot(xs, b_landau, color='green', ls="--", lw=PLOT_STYLE["linewidth"], label=r"$b_\text{eq}$" if first else None)
                first=False
                if y >= 0:
                    # on garde une référence à LA couleur utilisée ici, pour
                    # pouvoir colorer le label epsilon correspondant plus tard
                    # avec exactement la même couleur (cf. landau_colors ci-dessous)
                    color_landau = greens[green_idx]
                    axB.plot(xs, B_landau, color=color_landau, ls="--", lw=PLOT_STYLE["linewidth"], label=None)
                    green_idx += 1

            axB.plot(xs, B, color=blues[y_idx], lw=PLOT_STYLE["linewidth"], label=label)
            axb.plot(xs, b, color=oranges[y_idx], lw=PLOT_STYLE["linewidth"],label=label)

            if not vary_epsilon and y >= 0:
                landau_curves.append(np.asarray(B_landau))
                positive_eps.append(y)
                landau_colors.append(color_landau)

        # Placement automatique des labels epsilon
        if not vary_epsilon:

            # de la marge verticale autour des courbes B, pour que les labels
            # (qui se placent avec un offset de 10 pts au-dessus/en-dessous
            # du point choisi) aient la place d'exister sans sortir du cadre.
            # A faire AVANT la boucle d'annotation, pour que best_label_position
            # et l'annotate voient les limites finales de l'axe.
            axB.autoscale(enable=True, axis='y')
            axB.margins(y=0.18)

            for eps, curve, color in zip(positive_eps, landau_curves, landau_colors):

                xlab, ylab, side = best_label_position(
                    xs,
                    curve,
                    [c for c in landau_curves if c is not curve]
                )

                axB.annotate(
                    rf"$\varepsilon={fmt_num(eps)}$",
                    xy=(xlab, ylab),
                    xytext=(0, 10*side),
                    textcoords="offset points",
                    color=color,
                    fontsize=PLOT_STYLE["legend_fontsize"],
                    ha="center",
                    va="bottom" if side > 0 else "top",
                    annotation_clip=True,
                    clip_on=True,
                )
        style_axis(axB, ylabel="B", xscale=xscale)
        style_axis(axb, xlabel=r"$\varepsilon$" if vary_epsilon else r"$\kappa$", ylabel="b", xscale=xscale)

        if show_lambda:
            if vary_epsilon : 
                axB.text(
                    0.8, 0.1, rf"$\Lambda={fmt_num(Lambda)}$",
                    transform=axB.transAxes,
                    fontsize=20,
                    color=_OKABE_ITO["vermillion"],
                    va="center", ha="center"
                )
            else : 
                axb.text(
                    0.8, 0.4, rf"$\Lambda={fmt_num(Lambda)}$",
                    transform=axb.transAxes,
                    fontsize=20,
                    color=_OKABE_ITO["vermillion"],
                    va="center", ha="center"
                )

        # layout="constrained" (défini dans new_figure) ajuste automatiquement
        # les marges pour que rien ne soit coupé, tout en gardant une sortie
        # toujours a exactement figsize * dpi, quel que soit le contenu.
        cfg.write_config_file()
        save_fig(fig, cfg.folder, f"Lambda={Lambda}")

    if vary_epsilon: print("Balayage de epsilon : fini")
    else: print("Balayage de kappa : fini")


def fig_1(show_lambda=True): balayage(False, show_lambda=show_lambda)
def fig_2(show_lambda=True): balayage(True, show_lambda=show_lambda)

def court_terme(show_lambda=True):
    fig_1(show_lambda=show_lambda)
    fig_2(show_lambda=show_lambda)


# ==========================================================================================================
# -------------- FIGURES 3,4,5,6 : Moyen terme, intermittence ---------------------------------------------
# ==========================================================================================================
datadir_mid = os.path.join(datadir, "2_Moyen_terme_Intermittence")
Lambdas = [0.1]
epsiloneqs = [-0.1,0,0.1]
kappaeqs = [0,0.1, 1]
inter_list = [(True, False), (False, True), (True, True)]  # kappa, epsilon


def plot_fig5(data, folder):
    t=data[:,0]
    B=data[:,1]
    kappa_data=data[:,3]
    epsilon_data=data[:,4]
    n_parts = 10
    n = len(epsilon_data)

    # length of each chunk
    chunk_size = n // n_parts

    for i in range(n_parts):
        start = i * chunk_size

        # make sure the last chunk takes the remaining points
        if i == n_parts - 1:
            end = n
        else:
            end = (i + 1) * chunk_size
        t_part = t[start:end]
        t_start = t_part[0]
        t_end = t_part[-1]
        eps = epsilon_data[start:end]
        kap = kappa_data[start:end]
        B_part = B[start:end]

        plot_fig5_part(eps, kap, B_part, folder, i+1,round(t_start),round(t_end))


def plot_fig5_part(epsilon_data, kappa_data, B, folder, part_number,t_start,t_end):
    fig_5, ax_5 = new_figure()
    fig_5.patch.set_facecolor('white')
    ax_5.set_facecolor('white')

    xlim = (epsilon_data.min(), epsilon_data.max())
    ylim = (kappa_data.min(), kappa_data.max())

    ax_5.set_xlim(xlim)
    ax_5.set_ylim(ylim)

    # zone grisée : kappa + epsilon < 0
    x_fill = np.linspace(xlim[0], xlim[1], 500)
    y_boundary = -x_fill
    ax_5.fill_between(
        x_fill,
        ylim[0],
        y_boundary,
        color='0.85',
        zorder=0
    )

    # trajectoire colorée selon B
    points = np.array([epsilon_data, kappa_data]).T.reshape(-1, 1, 2)
    segments = np.concatenate(
        [points[:-1], points[1:]],
        axis=1
    )

    norm = plt.Normalize(B.min(), B.max())

    lc = LineCollection(
        segments,
        cmap='coolwarm',
        norm=norm,
        zorder=2
    )

    lc.set_array(B[:-1])
    lc.set_linewidth(PLOT_STYLE["linewidth"])

    line = ax_5.add_collection(lc)

    cbar = fig_5.colorbar(line, ax=ax_5)
    cbar.set_label(r"$B$")

    style_axis(
        ax_5,
        xlabel=r"$\varepsilon(t)$",
        ylabel=r"$\kappa(t)$",
        legend=False
    )

    ax_5.grid(True, zorder=1, alpha=0.3)

    save_fig(
        fig_5,
        os.path.join(folder,"trajectoires"),
        f"[{t_start},{t_end}]",
        close=True
    )

def plot_fft(t, signal, label, folder, filename_prefix, max_freq=0.1):
    dt = t[1] - t[0]
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(signal - signal.mean())
    amplitude = np.abs(fft_vals)

    freq_mask = freqs <= max_freq
    freqs_filtered = freqs[freq_mask]
    amplitude_filtered = amplitude[freq_mask]

    fig_fft, ax_fft = new_figure()
    ax_fft.plot(freqs_filtered, amplitude_filtered, lw=PLOT_STYLE["linewidth"], color='purple')
    style_axis(ax_fft, xlabel="Fréquence", ylabel="Amplitude", xscale='log', yscale='log',
               title=f"FFT de {label}", legend=False)
    save_fig(fig_fft, folder, f"{filename_prefix}_fft", dpi=300)

    peaks, _ = find_peaks(amplitude_filtered)

    if len(peaks) == 0:
        top_freqs = [freqs_filtered[np.argmax(amplitude_filtered)]]
        top_amps = [amplitude_filtered.max()]
    else:
        peak_amps = amplitude_filtered[peaks]
        peak_freqs = freqs_filtered[peaks]
        n_top = min(3, len(peaks))
        top_idx = np.argsort(peak_amps)[::-1][:n_top]
        top_freqs = peak_freqs[top_idx]
        top_amps = peak_amps[top_idx]

    with open(os.path.join(folder, f"{filename_prefix}_fft_top3.txt"), "w") as f:
        f.write(f"Fréquences dominantes de la FFT de {label}\n")
        f.write("=" * 50 + "\n")
        if len(peaks) == 0:
            f.write("Aucun pic local détecté, fréquence du maximum global reportée :\n")
        for rank, (freq, amp) in enumerate(zip(top_freqs, top_amps), start=1):
            f.write(f"{rank}. Fréquence = {freq:.6e}   |   Amplitude = {amp:.6e}\n")

def plot_fig4(t, B, b, kappa_data, epsilon_data, inter_kappa, inter_epsilon, folder):
    fig_4_B_b, ax_4_B_b = new_figure()
    fig_4_stat, ax_4_stat = new_figure()

    ax_4_B_b.plot(t, B, color=PLOT_STYLE["color_B"], lw=PLOT_STYLE["linewidth"], label=r"$B(t)$")
    ax_4_B_b.plot(t, b, color=PLOT_STYLE["color_b"], lw=PLOT_STYLE["linewidth"], label=r"$b(t)$")
    if inter_kappa:
        ax_4_stat.plot(t, kappa_data, color='red', lw=PLOT_STYLE["linewidth"], label=r"$\kappa(t)$")
    if inter_epsilon:
        ax_4_stat.plot(t, epsilon_data, color='green', lw=PLOT_STYLE["linewidth"], label=r"$\varepsilon(t)$")

    if inter_epsilon and not inter_kappa:
        stat_ylabel = "Dynamo Number"
    elif inter_kappa and not inter_epsilon:
        stat_ylabel = "Coupling factor"
    else:
        stat_ylabel = "Stochastic parameters"

    style_axis(ax_4_B_b, xlabel=r"$\varepsilon \cdot t$", ylabel="Magnetic Amplitudes")
    style_axis(ax_4_stat, xlabel=r"$\varepsilon \cdot t$", ylabel=stat_ylabel)

    save_fig(fig_4_B_b, folder, "Fig_4_B_b")
    save_fig(fig_4_stat, folder, "Fig_4_stat")

def _skumanich_cache_meta(kind, cfg: config, data_avg, stride):
    """Fingerprint of everything that changes the result of a skumanich_500 /
    skumanich_analytique call. If any of this differs from what's stored on
    disk, the cache is considered stale and the computation is redone."""
    return {
        "kind": kind,
        "Lambda": cfg.Lambda,
        "epsiloneq": cfg.epsiloneq,
        "kappaeq": cfg.kappaeq,
        "tfin": cfg.tfin,
        "taukappa": cfg.taukappa,
        "deltaepsilon": cfg.deltaepsilon,
        "deltakappa": cfg.deltakappa,
        "stride": stride,
        "n_points": int(data_avg.shape[0]),
    }

def _load_skumanich_cache(folder, filenames, meta):
    """Return the cached arrays (in the order of `filenames`) if a matching,
    complete cache is found in `folder`; otherwise None."""
    meta_path = os.path.join(folder, "cache_meta.json")
    if not os.path.exists(meta_path):
        return None
    if not all(os.path.exists(os.path.join(folder, f)) for f in filenames):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if cached_meta != meta:
        return None
    try:
        return [np.genfromtxt(os.path.join(folder, f)) for f in filenames]
    except OSError:
        return None

def _save_skumanich_cache(folder, arrays, meta):
    os.makedirs(folder, exist_ok=True)
    for name, arr in arrays.items():
        np.savetxt(os.path.join(folder, name), arr)
    with open(os.path.join(folder, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

def skumanich_500(folder, cfg :config, data_avg):
    stride = 160000
    meta = _skumanich_cache_meta("skumanich_500", cfg, data_avg, stride)
    cached = _load_skumanich_cache(folder, ["output.txt", "stddev.txt"], meta)
    if cached is not None:
        return cached[0], cached[1]

    epsilons=data_avg[:,4][::stride]
    ts=data_avg[:,0][::stride]
    kappas=data_avg[:,3][::stride]
    Omegas=data_avg[:,5][::stride]
    B_means=[]
    B_stddevs=[]
    b_means=[]
    b_stddevs=[]
    B0=cfg.B0
    b0=cfg.b0
    kappaeq=cfg.kappaeq
    kappa0=None
    for epsiloneq in tqdm(epsilons, desc="skumanich_500 inner", leave=False): 
        cfg_new= config(datadir=folder, term='mid',
                    epsiloneq=epsiloneq, Lambda=cfg.Lambda, kappaeq=kappaeq, run_index=1,
                    inter_kappa=True, inter_epsilon=True, B0=B0, b0=b0,
                    kappa0=kappaeq if kappa0 is None else kappa0,
                    taukappa=cfg.taukappa, deltaepsilon=cfg.deltaepsilon, deltakappa=cfg.deltakappa,
                    tfin=cfg.tfin/10)
        data=cfg_new.run(save=False)
        B_means.append(np.mean(data[:,1]))
        B_stddevs.append(np.std(data[:,1]))
        b_means.append(np.mean(data[:,2]))
        b_stddevs.append(np.std(data[:,2]))
        B0=data[-1,1]
        b0=data[-1,2]
        kappa0=data[-1,3]
    data_new = np.column_stack((ts, B_means, b_means, kappas, epsilons, Omegas))
    stddev_new = np.column_stack((np.zeros_like(ts), B_stddevs, b_stddevs))

    _save_skumanich_cache(folder, {"output.txt": data_new, "stddev.txt": stddev_new}, meta)

    return data_new, stddev_new

def skumanich_analytique(folder, cfg :config, data_avg):
    stride = 100000
    meta = _skumanich_cache_meta("skumanich_analytique", cfg, data_avg, stride)
    cached = _load_skumanich_cache(folder, ["output.txt"], meta)
    if cached is not None:
        return cached[0]

    epsilons=data_avg[:,4][::stride]
    ts=data_avg[:,0][::stride]
    kappas=data_avg[:,3][::stride]
    Omegas=data_avg[:,5][::stride]
    B_eqs=[]
    b_eqs=[]
    for epsiloneq in epsilons : 
        cfg_new= config(datadir=folder, term='mid',
                    epsiloneq=epsiloneq, Lambda=cfg.Lambda, kappaeq=cfg.kappaeq, run_index=1,
                    inter_kappa=True, inter_epsilon=True,
                    taukappa=cfg.taukappa, deltaepsilon=cfg.deltaepsilon, deltakappa=cfg.deltakappa,
                    tfin=10)
        B_eq,b_eq = cfg_new.get_eq()[0]
        B_eqs.append(B_eq)
        b_eqs.append(b_eq)
    data_new = np.column_stack((ts, B_eqs, b_eqs, kappas, epsilons, Omegas))

    _save_skumanich_cache(folder, {"output.txt": data_new}, meta)
    return data_new

def plot_B_omega(data, folder, name="Fig_B_omega", stddevs=None, data_analytique=None):
    Omega = data[:, 5]
    B = data[:, 1]

    order = np.argsort(Omega)
    Omega_sorted = Omega[order]
    B_sorted = B[order]

    fig, ax = new_figure()

    # rasterized=True : cette courbe vient d'une simu 'long' et contient
    # énormément de points. La tracer en vectoriel (comme avant) génère un
    # chemin gigantesque dans le .eps/.pdf (fichiers de ~1.7 Go observés).
    # En la rasterisant, elle est convertie en image bitmap (à la résolution
    # `dpi` de save_fig) et intégrée dans le fichier vectoriel, qui redevient
    # léger (texte, axes, légende restent en vectoriel net). zorder fixé
    # explicitement pour que matplotlib regroupe correctement les couches
    # rasterisées entre elles.
    ax.plot(
        Omega_sorted, B_sorted,
        color=PLOT_STYLE["color_B"],
        lw=PLOT_STYLE["linewidth"],
        label=r"$B(\Omega)$",
        rasterized=True,
        zorder=1,
    )

    if stddevs is not None:
        B_stddev = stddevs[:, 1][order]
        B_moins = [x if x > 0 else 0 for x in (B_sorted - B_stddev)]
        ax.fill_between(
            Omega_sorted,
            B_moins,
            B_sorted + B_stddev,
            color=_OKABE_ITO["sky_blue"],
            alpha=0.25,
            rasterized=True,
            zorder=1,
        )

    if data_analytique is not None:
        Omega_a = data_analytique[:, 5]
        B_a = data_analytique[:, 1]
        order_a = np.argsort(Omega_a)
        ax.plot(
            Omega_a[order_a], B_a[order_a],
            color="black",
            linestyle="--",
            lw=PLOT_STYLE["linewidth"],
            label=r"Analytical $B(\Omega)$"
            # celle-ci reste vectorielle : c'est la courbe analytique, peu de points
        )

    style_axis(ax, xlabel=r"$\Omega$", ylabel="B")
    save_fig(fig, folder, name)


def get_freq(minimas, tfin):
    # minimas is a list of (t_center, duree) tuples, one per detected minimum.
    # There can be zero, one, or many of them - don't assume a fixed count.
    if len(minimas) == 0:
        return 0.0
    durees = [duree for (_, duree) in minimas]
    return sum(durees) / tfin

def big_simus():
    tqdm.write("Simulations d'intermittence : début")
    for Lambda in tqdm(Lambdas, desc="Lambda", position=0, leave=True):
        Lambda_folder = os.path.join(datadir_mid, "Lambda=" + str(Lambda))
        for epsiloneq in tqdm(epsiloneqs, desc="epsiloneqs", position=1, leave=False):
            epsiloneq_folder = os.path.join(Lambda_folder, "epsiloneq=" + str(epsiloneq))

            for (inter_kappa, inter_epsilon) in inter_list:

                if (not inter_kappa) and inter_epsilon:
                    folder_fig_3 = os.path.join(epsiloneq_folder, "kappa_dependency")
                    fig_3_B, ax_3_B = new_figure()
                    fig_3_b, ax_3_b = new_figure()
                    minimas_folder = os.path.join(folder_fig_3, "minimas")
                    data_list_per_kappa = []  # une entrée par kappa, chacune = liste des 10 simus non-moyennées
                    for run_idx, kappaeq in enumerate(tqdm(kappaeqs, desc="kappa_dependency", position=2, leave=False)):
                        cfg = config(datadir=folder_fig_3, term='mid',
                                     epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq,
                                     inter_kappa=inter_kappa, inter_epsilon=inter_epsilon,
                                     run_index=run_idx + 1)
                        (B_eq, b_eq) = cfg.get_eq()[0]
                        cfg.B0 = B_eq
                        cfg.b0 = b_eq
                        data_list, data_avg, _ = cfg.run_and_avg(save_all=True, save_figs=True, n=3)
                        data_list_per_kappa.append(data_list)
                        minimas_total = []
                        for data in data_list:
                            minimas = cfg.stat_analysis(data)
                            cfg.write_stat_file(minimas=minimas)
                            minimas_total.extend(minimas)
                        freq=get_freq(minimas, cfg.tfin)
                        cfg.plot_histograms(minimas_list=minimas_total, differentfolder=minimas_folder,
                                             name="kappaeq=" + str(kappaeq), freq=freq)
                        cfg.plot_time(data=data_avg, type='epsilon', differentfolder=folder_fig_3, name="stat_kappa="+str(kappaeq))
                        t = data_avg[:, 0]
                        B = data_avg[:, 1]
                        b = data_avg[:, 2]
                        
                        ax_3_B.plot(t, B, lw=PLOT_STYLE["linewidth"], label=r"$\kappa$=" + str(kappaeq))
                        ax_3_b.plot(t, b, lw=PLOT_STYLE["linewidth"], label=r"$\kappa$=" + str(kappaeq))

                    for ax in [ax_3_B, ax_3_b]:
                        style_axis(
                            ax,
                            xlabel="t",
                            legend=False
                        )
                        style_axis_legend_inside_top(
                            ax,
                            ncol=len(kappaeqs)
                        )

                    save_fig(fig_3_B, folder_fig_3, "Fig_3_B_large", dpi=300)
                    save_fig(fig_3_b, folder_fig_3, "Fig_3_b_small", dpi=300)

                    # Graphes équivalents, mais pour chaque simulation individuelle
                    # (non-moyennée) : pour chaque indice de run r (1 a 10), on
                    # superpose les 3 courbes B (une par kappa) issues de la
                    # r-ieme simulation de chaque kappa, et pareil pour b.
                    folder_runs = os.path.join(folder_fig_3, "simus_individuelles")
                    n_runs = min(len(dl) for dl in data_list_per_kappa)
                    for r in range(n_runs):
                        fig_r_B, ax_r_B = new_figure()
                        fig_r_b, ax_r_b = new_figure()
                        for k, kappaeq in enumerate(kappaeqs):
                            data_r = data_list_per_kappa[k][r]
                            t = data_r[:, 0]
                            B = data_r[:, 1]
                            b = data_r[:, 2]
                            ax_r_B.plot(t, B, lw=PLOT_STYLE["linewidth"], label=r"$\kappa$=" + str(kappaeq))
                            ax_r_b.plot(t, b, lw=PLOT_STYLE["linewidth"], label=r"$\kappa$=" + str(kappaeq))

                        for ax in [ax_r_B, ax_r_b]:
                            style_axis(
                                ax,
                                xlabel="t",
                                legend=False
                            )

                            style_axis_legend_inside_top(
                                ax,
                                ncol=len(kappaeqs)
                            )
                                
                        save_fig(fig_r_B, folder_runs, f"Fig_3_Big_run{r + 1:02d}", dpi=300)
                        save_fig(fig_r_b, folder_runs, f"Fig_3_b_run{r + 1:02d}", dpi=300)

                    cfg.write_config_file()

                if inter_kappa and (not inter_epsilon):
                    skumanich_folder = os.path.join(epsiloneq_folder, "skumanich")

                    for kappaeq in tqdm(kappaeqs, desc="skumanich", position=2, leave=False):
                        kappaeq_folder = os.path.join(skumanich_folder, "kappaeq=" + str(kappaeq))
                        kappaeq_mean_folder = os.path.join(kappaeq_folder, "mean")
                        kappaeq_100_folder = os.path.join(kappaeq_folder, "500")
                        kappaeq_analytique_folder = os.path.join(kappaeq_folder, "analytique")
                        try:
                            cfg = config(datadir=kappaeq_mean_folder, term='long',
                                         epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq, run_index=1,
                                         inter_kappa=inter_kappa, inter_epsilon=inter_epsilon)
                            (B_eq, b_eq) = cfg.get_eq(skuma=True)[0]
                            cfg.B0 = B_eq
                            cfg.b0 = b_eq
                            _, data_avg, stddevs = cfg.run_and_avg(save_all=True, save_figs=True)
                            data_100, std_100 = skumanich_500(folder=kappaeq_100_folder, cfg=cfg, data_avg=data_avg)
                            for type in ['Bb', 'kappa', 'epsilon', 'Omega']:
                                cfg.plot_time(data_avg, type=type, show=False, stddevs=stddevs, differentfolder=kappaeq_mean_folder)
                                cfg.plot_time(data_100, type=type, show=False, stddevs=std_100, differentfolder=kappaeq_100_folder)
                            data_analytique = skumanich_analytique(folder=kappaeq_analytique_folder, cfg=cfg, data_avg=data_avg)
                            cfg.plot_time(data=data_analytique, type="Bb", show=False, differentfolder=kappaeq_analytique_folder, analytique=True)

                            # les deux comparaisons B(Omega) vont directement dans kappaeq_folder,
                            # à côté des sous-dossiers mean/500/analytique
                            plot_B_omega(data_avg, kappaeq_folder, name="Fig_B_omega_mean",
                                         stddevs=stddevs, data_analytique=data_analytique)
                            plot_B_omega(data_100, kappaeq_folder, name="Fig_B_omega_500",
                                         stddevs=std_100, data_analytique=data_analytique)
                        except Exception as e:
                            tqdm.write(f"[skumanich] echec pour Lambda={Lambda}, epsiloneq={epsiloneq}, "
                                       f"kappaeq={kappaeq} : {e}")
                            continue
                        

                if inter_kappa and inter_epsilon:
                    comportements_folder = os.path.join(epsiloneq_folder, "comportements_divers")
                    for kappaeq in tqdm([0,0.1,0.3,1], desc="intermittency", position=2, leave=False):
                        kappaeq_folder = os.path.join(comportements_folder, "kappaeq=" + str(kappaeq))
                        minimas_list = []
                        cfg = None
                        for i in range(10):
                            cfg = config(datadir=kappaeq_folder, term='mid',
                                         epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq, run_index=i + 1,
                                         inter_kappa=inter_kappa, inter_epsilon=inter_epsilon)
                            (B_eq, b_eq) = cfg.get_eq()[0]
                            cfg.B0 = B_eq
                            cfg.b0 = b_eq
                            data = cfg.run(save=True)
                            t = data[:, 0]
                            B = data[:, 1]
                            b = data[:, 2]
                            kappa_data = data[:, 3]
                            epsilon_data = data[:, 4]
                            minimas = cfg.stat_analysis(data=data)
                            minimas_list.extend(minimas)
                            #plot_fft(t, kappa_data, r"$\kappa(t)$", cfg.folder, "kappa")
                            plot_fig5(data, cfg.folder)
                            plot_fig4(t, B, b, kappa_data, epsilon_data, inter_kappa, inter_epsilon, cfg.folder)
                        freq=get_freq(minimas=minimas_list,tfin=10*cfg.tfin)
                        cfg.plot_histograms(minimas_list=minimas_list, differentfolder=kappaeq_folder, freq=freq)
    tqdm.write("Simulations d'intermittence : fin")


# ==========================================================================================================
# -------------- SELECTION DE FIGURES POUR L'ARTICLE (.png + .eps uniquement) ------------------------------
# ==========================================================================================================
# But : ajouter, dans un dossier de resultats deja rempli (fichiers .txt et
# configuration.in existants, figures deja supprimees), UNIQUEMENT les
# figures listees ci-dessous, en .png et .eps (cf.
# plot_style.PLOT_STYLE["save_formats"]).
#
# On reconstruit systematiquement les memes cfg / mêmes chemins de dossiers
# que big_simus(), pour que cfg.run()/cfg.run_and_avg() retrouvent les
# donnees deja simulees sur disque (configuration.in identique => pas de
# resimulation) et se contentent de relire output.txt. Le seul cas qui sera
# reellement resimule est kappaeq=0.3 dans comportements_divers, puisque
# cette valeur n'existait pas avant.
#
# A la difference de big_simus(), on evite ici tout appel de fonction qui
# produirait des figures NON demandees (plot_time des runs individuels,
# skumanich_500, Fig_B_omega_500, plot_fig4/plot_fig5 sur les simus non
# selectionnees, etc.), pour ne pas remplir le dossier de .eps inutiles.

def _fig5_last_part_only(data, folder):
    """Identique a plot_fig5(), mais ne trace/sauvegarde que la derniere des
    10 subdivisions temporelles (pour tfin=150000 : t in [135000, 150000])."""
    t = data[:, 0]
    B = data[:, 1]
    kappa_data = data[:, 3]
    epsilon_data = data[:, 4]
    n_parts = 10
    n = len(epsilon_data)
    chunk_size = n // n_parts

    i = n_parts - 1  # derniere subdivision uniquement
    start = i * chunk_size
    end = n
    t_part = t[start:end]
    t_start = t_part[0]
    t_end = t_part[-1]
    eps = epsilon_data[start:end]
    kap = kappa_data[start:end]
    B_part = B[start:end]

    plot_fig5_part(eps, kap, B_part, folder, i + 1, round(t_start), round(t_end))


def fig_kappa_dependency_selection():
    """kappa_dependency : Lambda=0.1, epsiloneq=0, simus_individuelles,
    run02 (figures B et b superposant les 3 courbes de kappa pour la
    2e simu individuelle, r_wanted=1 en index 0-based)."""
    Lambda = 0.1
    epsiloneq = 0
    Lambda_folder = os.path.join(datadir_mid, "Lambda=" + str(Lambda))
    epsiloneq_folder = os.path.join(Lambda_folder, "epsiloneq=" + str(epsiloneq))
    folder_fig_3 = os.path.join(epsiloneq_folder, "kappa_dependency")
    folder_runs = os.path.join(folder_fig_3, "simus_individuelles")

    r_wanted = 1  # index 0-based -> "run02"

    data_list_per_kappa = []
    for run_idx, kappaeq in enumerate(kappaeqs):
        cfg = config(datadir=folder_fig_3, term='mid',
                     epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq,
                     inter_kappa=False, inter_epsilon=True,
                     run_index=run_idx + 1)
        (B_eq, b_eq) = cfg.get_eq()[0]
        cfg.B0 = B_eq
        cfg.b0 = b_eq
        data_list, _, _ = cfg.run_and_avg(save_all=True, save_figs=False, n=3)
        data_list_per_kappa.append(data_list)

    fig_r_B, ax_r_B = new_figure()
    fig_r_b, ax_r_b = new_figure()
    for k, kappaeq in enumerate(kappaeqs):
        data_r = data_list_per_kappa[k][r_wanted]
        t = data_r[:, 0]
        B = data_r[:, 1]
        b = data_r[:, 2]
        ax_r_B.plot(t, B, lw=PLOT_STYLE["linewidth"], label=r"$\kappa$=" + str(kappaeq))
        ax_r_b.plot(t, b, lw=PLOT_STYLE["linewidth"], label=r"$\kappa$=" + str(kappaeq))

    for ax in [ax_r_B, ax_r_b]:
        style_axis(
            ax,
            xlabel="t",
            legend=False
        )

        style_axis_legend_inside_top(
            ax,
            ncol=len(kappaeqs)
        )

        save_fig(fig_r_B, folder_runs, f"Fig_3_Big_run{r_wanted + 1:02d}", dpi=300)
        save_fig(fig_r_b, folder_runs, f"Fig_3_b_run{r_wanted + 1:02d}", dpi=300)


def fig_comportements_divers_selection():
    """comportements_divers (Lambda=0.1) :
      - Fig_4 (B_b, stat) pour les 3 simulations specifiques demandees ;
      - Fig_5 (trajectoire), derniere subdivision seulement, uniquement
        pour epsiloneq=-0.1, kappaeq=0.1, simu_3 ;
      - plot_minimas pour (epsiloneq, kappaeq) in
        {(-0.1, 0.1), (0, 1), (0.1, 1)} (donnees deja existantes, pas de
        resimulation) et pour kappaeq=0.3 avec les trois epsiloneq
        (nouveau -> resimule, sans les graphiques temporels Fig_4/Fig_5)."""
    Lambda = 0.1
    inter_kappa, inter_epsilon = True, True

    # (epsiloneq, kappaeq) -> (indice de simu a tracer en Fig_4, tracer aussi Fig_5 ?)
    special_simu = {
        (-0.1, 0.1): (3, True),
    }

    # + les combinaisons kappaeq=0.3 (nouvelles), pour lesquelles seul
    # plot_minimas est demande, sans Fig_4/Fig_5.
    combos = list(special_simu.keys())

    for (epsiloneq, kappaeq) in combos:
        epsiloneq_folder = os.path.join(datadir_mid, "Lambda=" + str(Lambda),
                                         "epsiloneq=" + str(epsiloneq))
        comportements_folder = os.path.join(epsiloneq_folder, "comportements_divers")
        kappaeq_folder = os.path.join(comportements_folder, "kappaeq=" + str(kappaeq))

        wanted = special_simu.get((epsiloneq, kappaeq))
        wanted_idx = wanted[0] if wanted else None
        wanted_fig5 = wanted[1] if wanted else False

        minimas_list = []
        cfg = None
        for i in range(10):
            cfg = config(datadir=kappaeq_folder, term='mid',
                         epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq, run_index=i + 1,
                         inter_kappa=inter_kappa, inter_epsilon=inter_epsilon)
            (B_eq, b_eq) = cfg.get_eq()[0]
            cfg.B0 = B_eq
            cfg.b0 = b_eq
            data = cfg.run(save=True)
            minimas = cfg.stat_analysis(data=data)
            minimas_list.extend(minimas)

            
            if wanted_idx is not None and (i + 1) == wanted_idx:
                t = data[:, 0]
                B = data[:, 1]
                b = data[:, 2]
                kappa_data = data[:, 3]
                epsilon_data = data[:, 4]
                plot_fig4(t, B, b, kappa_data, epsilon_data, inter_kappa, inter_epsilon, cfg.folder)
                if wanted_fig5:
                   _fig5_last_part_only(data, cfg.folder)

        freq = get_freq(minimas=minimas_list, tfin=10 * cfg.tfin)
        cfg.plot_histograms(minimas_list=minimas_list, differentfolder=kappaeq_folder, freq=freq)


def fig_skumanich_selection():
    """skumanich : uniquement les sous-dossiers 'mean' et 'analytique' (pas
    de '500'), pour les trois couples (epsiloneq, kappaeq) demandes."""
    Lambda = 0.1
    inter_kappa, inter_epsilon = True, False

    pairs = [(-0.1, 0.1), (0, 0.1), (0.1, 1)]

    for (epsiloneq, kappaeq) in pairs:
        epsiloneq_folder = os.path.join(datadir_mid, "Lambda=" + str(Lambda),
                                         "epsiloneq=" + str(epsiloneq))
        skumanich_folder = os.path.join(epsiloneq_folder, "skumanich")
        kappaeq_folder = os.path.join(skumanich_folder, "kappaeq=" + str(kappaeq))
        kappaeq_mean_folder = os.path.join(kappaeq_folder, "mean")
        kappaeq_analytique_folder = os.path.join(kappaeq_folder, "analytique")

        try:
            cfg = config(datadir=kappaeq_mean_folder, term='long',
                         epsiloneq=epsiloneq, Lambda=Lambda, kappaeq=kappaeq, run_index=1,
                         inter_kappa=inter_kappa, inter_epsilon=inter_epsilon)
            (B_eq, b_eq) = cfg.get_eq(skuma=True)[0]
            cfg.B0 = B_eq
            cfg.b0 = b_eq
            _, data_avg, stddevs = cfg.run_and_avg(save_all=True, save_figs=False)
            #for type in ['Bb', 'kappa', 'epsilon', 'Omega']:
                #cfg.plot_time(data_avg, type=type, show=False, stddevs=stddevs, differentfolder=kappaeq_mean_folder)
            data_analytique = skumanich_analytique(folder=kappaeq_analytique_folder, cfg=cfg, data_avg=data_avg)
            #cfg.plot_time(data=data_analytique, type="Bb", show=False, differentfolder=kappaeq_analytique_folder, analytique=True)
            plot_B_omega(data_avg, kappaeq_folder, name="Fig_B_omega_mean",
                         stddevs=stddevs, data_analytique=data_analytique)
        except Exception as e:
            tqdm.write(f"[skumanich] echec pour Lambda={Lambda}, epsiloneq={epsiloneq}, "
                       f"kappaeq={kappaeq} : {e}")
            continue


def figures_article():
    """Point d'entree unique : ne genere QUE les figures selectionnees pour
    l'article, en .png et .eps, en reutilisant les donnees deja simulees
    quand elles existent (seul kappaeq=0.3 dans comportements_divers est un
    nouveau cas et sera donc resimule)."""

    """tqdm.write("Balayages (analytique, toutes les figures) : debut")
    court_terme()
    tqdm.write("Balayages (analytique, toutes les figures) : fin")"""

    """tqdm.write("kappa_dependency (selection) : debut")
    fig_kappa_dependency_selection()
    tqdm.write("kappa_dependency (selection) : fin")"""

    tqdm.write("comportements_divers (selection) : debut")
    fig_comportements_divers_selection()
    tqdm.write("comportements_divers (selection) : fin")

    tqdm.write("skumanich (selection) : debut")
    fig_skumanich_selection()
    tqdm.write("skumanich (selection) : fin")


if __name__ == "__main__":
    figures_article()