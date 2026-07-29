import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008).
_OKABE_ITO = {
    "black":         "#000000",
    "orange":        "#BB9138",
    "sky_blue":      "#508CAE",
    "bluish_green":  "#248D71",
    "yellow":        "#F0E442",
    "blue":          "#235F82",
    "vermillion":    "#D55E00",
    "reddish_purple":"#CC79A7",
    "sky_orange":    "#D6AA68",
}

PLOT_STYLE = {
    # ---- Figure sizes (inches), matched to APS/PRE column widths ----------
    # Default single-panel figure, sized for a single-column PRE figure.
    "figsize": (3.375, 2.6),
    # Single column, taller (e.g. stacked/shared-x subplots).
    "figsize_tall": (3.375, 5.0),
    # Full double-column width, for wide single-row figures.
    "figsize_double": (7.0, 3.0),
    # Full double-column width, taller (e.g. stacked subplots spanning both columns).
    "figsize_double_tall": (8, 8),

    # ---- Line weights ------------------------------------------------------
    # Curves must stay legible after being shrunk into a single column at
    # print resolution; 3 pt (the old value) is far too heavy at that size.
    "linewidth": 1.3,
    "linewidth_eq": 0.9,      # dashed equilibrium/reference lines
    "linewidth_minima": 0.8,  # dashed vertical minima markers
    "markersize": 4,

    # ---- Font sizes (points, at final print size) --------------------------
    "label_fontsize": 15,
    "title_fontsize": 15,
    "legend_fontsize": 15,
    "tick_fontsize": 15,

    # ---- Grid / axes --------------------------------------------------------
    "grid_alpha": 0.25,

    # ---- Output --------------------------------------------------------
    "dpi": 600,                     # raster fallback resolution
    "save_formats": ["png"], # always keep a vector copy for print

    # ---- Semantic colors (Okabe-Ito) ----------------------------------------
    "color_B": _OKABE_ITO["blue"],
    "color_b": _OKABE_ITO["vermillion"],
    "color_kappa": _OKABE_ITO["bluish_green"],
    "color_epsilon": _OKABE_ITO["reddish_purple"],
    "color_omega": _OKABE_ITO["sky_blue"],
    "color_minima": _OKABE_ITO["black"],

    # Qualitative palette for multi-curve/multi-group plots (histograms,
    # Lambda/kappa/epsilon sweeps, etc.). Cycle through with idx % len(palette).
    "palette": [
        _OKABE_ITO["blue"],
        _OKABE_ITO["vermillion"],
        _OKABE_ITO["bluish_green"],
        _OKABE_ITO["reddish_purple"],
        _OKABE_ITO["orange"],
        _OKABE_ITO["sky_blue"],
        _OKABE_ITO["yellow"],
        _OKABE_ITO["black"],
    ],
}

plt.rcParams.update({
    # ---- Fonts: serif, matching APS/PRE body text (Times-like) ------------
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": PLOT_STYLE["label_fontsize"],
    "axes.labelsize": PLOT_STYLE["label_fontsize"],
    "axes.titlesize": PLOT_STYLE["title_fontsize"],
    "xtick.labelsize": PLOT_STYLE["tick_fontsize"],
    "ytick.labelsize": PLOT_STYLE["tick_fontsize"],
    "legend.fontsize": PLOT_STYLE["legend_fontsize"],

    # ---- Lines / markers ----------------------------------------------------
    "lines.linewidth": PLOT_STYLE["linewidth"],
    "lines.markersize": PLOT_STYLE["markersize"],
    "axes.linewidth": 0.8,  # spine thickness

    # ---- Ticks: inward, on all sides, with minor ticks (standard APS look) --
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    # ---- Grid: light, unobtrusive (many PRE figures use none at all; this
    # is subtle enough to keep or turn off per-axis with grid=False) --------
    "axes.grid": False,
    "grid.alpha": PLOT_STYLE["grid_alpha"],
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.color": "0.5",

    # ---- Legend: clean, no box ------------------------------------------
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.4,
    "legend.labelspacing": 0.3,

    # ---- Default color cycle (Okabe-Ito) so any un-styled plot still looks
    # publication-appropriate rather than falling back to tab10. -------------
    "axes.prop_cycle": plt.cycler(color=PLOT_STYLE["palette"]),

    # ---- Saving ------------------------------------------------------------
    "savefig.dpi": PLOT_STYLE["dpi"],
    # IMPORTANT : ne PAS mettre "tight" ici. bbox="tight" recadre chaque figure
    # sur son contenu réel (légende, labels, ticks), donc deux figures avec le
    # même figsize peuvent finir avec des dimensions différentes en pixels dès
    # que leur contenu diffère un peu (légende plus longue, etc.). En laissant
    # None, la taille de sortie est toujours exactement figsize * dpi, donc
    # deux figures du même type (même figsize) ont TOUJOURS exactement la
    # même taille finale.
    "savefig.bbox": None,
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,

    # ---- Misc ----------------------------------------------------------
    "figure.dpi": 150,  # on-screen preview only; savefig.dpi governs output
})