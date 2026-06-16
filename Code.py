#!/usr/bin/env python3


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import warnings

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

warnings.filterwarnings("ignore")

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]

REF_COLOR  = "#222222"
REF_LWIDTH = 1.8

# Bornes de normalisation fixes
NORM_XMIN = 0.0
NORM_XMAX = 500.0

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────
def _float(s) -> float:
    return float(str(s).strip().replace(",", "."))

def _try_float(s, default=0.0) -> float:
    try:
        return _float(s)
    except (ValueError, AttributeError, TypeError):
        return default

def _is_num(cell) -> bool:
    s = str(cell).strip()
    if s.lower() in ("nan", "none", "", "inf", "-inf"):
        return False
    try:
        v = float(s.replace(",", "."))
        return not (v != v)
    except (ValueError, AttributeError):
        return False

def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    w = min(window, len(arr))
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.pad(arr, pad, mode="edge")
    out = np.convolve(padded, kernel, mode="valid")
    return out[:len(arr)]

def _trapz_interval(dist: np.ndarray, inten: np.ndarray,
                    x_min: float, x_max: float) -> float:
    mask = (dist >= x_min) & (dist <= x_max)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(inten[mask], dist[mask]))


# ──────────────────────────────────────────────────────────────────────────────
# Parsing fichier brut ImageJ (multi-profils)
# ──────────────────────────────────────────────────────────────────────────────
def parse_imagej_file(filepath: str) -> dict:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, header=None, dtype=str)
    else:
        for enc in ("cp1252", "latin1", "utf-8"):
            try:
                df = pd.read_csv(filepath, header=None, sep=";",
                                 encoding=enc, dtype=str)
                break
            except Exception:
                continue
        else:
            raise ValueError("Encodage CSV non reconnu.")

    row0 = df.iloc[0].tolist()
    profiles = {}
    col = 0
    while col < len(row0) - 1:
        cell = str(row0[col]).strip() if pd.notna(row0[col]) else ""
        if not cell or cell.lower() == "nan":
            col += 1
            continue
        depth_cell = str(row0[col + 1]).strip() if pd.notna(row0[col + 1]) else ""
        try:
            depth = _float(depth_cell)
        except ValueError:
            col += 1
            continue
        name = cell.strip()
        data_start = None
        for row_idx in range(1, len(df)):
            if (col + 1 < df.shape[1]
                    and _is_num(df.iloc[row_idx, col])
                    and _is_num(df.iloc[row_idx, col + 1])):
                data_start = row_idx
                break
        if data_start is None:
            col += 3
            continue
        distances, intensities = [], []
        for row_idx in range(data_start, len(df)):
            if col + 1 >= df.shape[1]:
                break
            if not _is_num(df.iloc[row_idx, col]) or not _is_num(df.iloc[row_idx, col + 1]):
                break
            distances.append(_float(df.iloc[row_idx, col]))
            intensities.append(_float(df.iloc[row_idx, col + 1]))
        if len(distances) > 1:
            profiles[name] = {
                "depth": depth,
                "distance": np.array(distances, dtype=float),
                "intensity": np.array(intensities, dtype=float),
            }
        col += 3

    if not profiles:
        raise ValueError("Aucun profil détecté.\n"
                         "Vérifiez le format : ligne 0 = nom;profondeur;;nom;profondeur;;")
    return profiles


# ──────────────────────────────────────────────────────────────────────────────
# Parsing fichier de référence (un seul profil)
# ──────────────────────────────────────────────────────────────────────────────
def parse_reference_file(filepath: str) -> dict:
    profiles = parse_imagej_file(filepath)
    if not profiles:
        raise ValueError("Aucun profil détecté dans le fichier de référence.")
    name = next(iter(profiles))
    return {**profiles[name], "name": name}


# ──────────────────────────────────────────────────────────────────────────────
# Chargement d'un projet exporté
# ──────────────────────────────────────────────────────────────────────────────
def load_project(filepath: str) -> dict:
    wb_raw = pd.read_excel(filepath, sheet_name=None, header=0)

    if "Récapitulatif" not in wb_raw:
        raise ValueError("Feuille 'Récapitulatif' introuvable.")

    recap = wb_raw["Récapitulatif"]
    profiles = {}
    corrections = {}

    for _, row in recap.iterrows():
        name = str(row["Profil"]).strip()
        if name not in wb_raw:
            continue
        sheet = wb_raw[name].dropna(how="all")
        dist_col  = "Distance brute (µm)"
        inten_col = "Intensité brute"
        if dist_col not in sheet.columns or inten_col not in sheet.columns:
            continue
        dist  = pd.to_numeric(sheet[dist_col],  errors="coerce").dropna().values
        inten = pd.to_numeric(sheet[inten_col], errors="coerce").dropna().values
        n = min(len(dist), len(inten))
        if n < 2:
            continue
        profiles[name] = {
            "depth":     float(row.get("Profondeur (µm)", 0) or 0),
            "distance":  dist[:n].astype(float),
            "intensity": inten[:n].astype(float),
        }
        corrections[name] = {
            "bg":          _try_float(row.get("Background soustrait", 0), 0.0),
            "zero_shift":  _try_float(row.get("Décalage distance (µm)", 0), 0.0),
            "zero_active": _try_float(row.get("Décalage distance (µm)", 0), 0.0) != 0.0,
            "flipped":     str(row.get("Retourné (↔)", "non")).strip().lower() == "oui",
            "scale_width": _try_float(row.get("Largeur mesurée (µm)", 0), 0.0),
        }

    if not profiles:
        raise ValueError("Aucune donnée brute récupérable depuis ce fichier projet.")

    params = {
        "smooth_active": False, "smooth_window": 5,
        "norm_active":   False,
    }
    if "Paramètres" in wb_raw:
        p = wb_raw["Paramètres"].dropna(how="all")
        pdict = {str(r.iloc[0]).strip(): (str(r.iloc[1]).strip() if len(r) > 1 else "")
                 for _, r in p.iterrows()}
        params["smooth_active"] = pdict.get("smooth_active", "False").lower() == "true"
        params["smooth_window"] = int(_try_float(pdict.get("smooth_window", "5"), 5))
        params["norm_active"]   = pdict.get("norm_active",   "False").lower() == "true"

    return {"profiles": profiles, "corrections": corrections, "params": params}


# ──────────────────────────────────────────────────────────────────────────────
# Export Excel
# ──────────────────────────────────────────────────────────────────────────────
def export_excel(filepath: str, corrected_data: list, global_params: dict):
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        summary_rows = []
        for d in corrected_data:
            summary_rows.append({
                "Profil":                  d["name"],
                "Profondeur (µm)":         d["depth"],
                "Retourné (↔)":           "oui" if d.get("flipped") else "non",
                "Décalage distance (µm)":  d["zero_shift"],
                "Largeur mesurée (µm)":    d.get("scale_width") or "—",
                "Background soustrait":    d["bg"],
                "Facteur normalisation":   round(d.get("norm_factor", 1.0), 6),
                "Fenêtre lissage (pts)":   d["smooth_window"] if d["smooth_window"] > 1 else "—",
                "N points":                len(d["distance_corr"]),
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Récapitulatif", index=False)

        param_rows = [
            ("smooth_active", str(global_params.get("smooth_active", False))),
            ("smooth_window", str(global_params.get("smooth_window", 5))),
            ("norm_active",   str(global_params.get("norm_active",   False))),
        ]
        pd.DataFrame(param_rows, columns=["Paramètre", "Valeur"]).to_excel(
            writer, sheet_name="Paramètres", index=False)

        for d in corrected_data:
            cols = {
                "Distance brute (µm)":    d["distance_raw"],
                "Intensité brute":        d["intensity_raw"],
                "Distance corrigée (µm)": d["distance_corr"],
                "Intensité corrigée":     d["intensity_corr"],
            }
            if d["intensity_smooth"] is not None:
                cols[f"Intensité lissée (w={d['smooth_window']})"] = d["intensity_smooth"]
            max_len = max(len(v) for v in cols.values())
            data_dict = {}
            for k, v in cols.items():
                arr = np.full(max_len, np.nan); arr[:len(v)] = v
                data_dict[k] = arr
            pd.DataFrame(data_dict).to_excel(writer, sheet_name=d["name"][:31], index=False)

        max_len = max(len(d["distance_corr"]) for d in corrected_data)
        plots_dict = {}
        for d in corrected_data:
            n = len(d["distance_corr"])
            dist_col  = np.full(max_len, np.nan); dist_col[:n] = d["distance_corr"]
            inten_col = np.full(max_len, np.nan)
            src = d["intensity_smooth"] if d["intensity_smooth"] is not None else d["intensity_corr"]
            inten_col[:len(src)] = src
            plots_dict[f"Dist – {d['name']}"] = dist_col
            plots_dict[f"Int – {d['name']}"]  = inten_col
        pd.DataFrame(plots_dict).to_excel(writer, sheet_name="Plots finaux", index=False)

        ws = writer.sheets["Récapitulatif"]
        for col_cells in ws.columns:
            max_w = max(len(str(c.value or "")) for c in col_cells) + 4
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_w, 40)


# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Visualisateur de profils ImageJ")
        self.root.geometry("1520x800")
        self.root.minsize(900, 600)

        self.profiles: dict = {}
        self.sorted_names: list = []

        self.prof_visible: dict = {}
        self.prof_flip:    dict = {}
        self.bg_val:       dict = {}
        self.zero_active:  dict = {}
        self.zero_val:     dict = {}
        self.scale_val:    dict = {}

        # Profil de référence
        self.reference: dict | None = None

        self._pick_target: str | None = None

        self.vline_250 = tk.BooleanVar(value=False)

        self._build_style()
        self._build_layout()

    def _build_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Title.TLabel",       font=("Helvetica", 9, "bold"))
        s.configure("Small.TLabel",       font=("Helvetica", 8))
        s.configure("Small.TCheckbutton", font=("Helvetica", 8))
        s.configure("Ref.TLabel",         font=("Helvetica", 9, "bold"),
                    foreground="#228B22")

    def _build_layout(self):
        # ── Barre 1 ────────────────────────────────────────────────────────────
        top = ttk.Frame(self.root, padding=(6, 4))
        top.pack(side="top", fill="x")

        ttk.Button(top, text="📂 Importer Excel / CSV",
                   command=self.import_file).pack(side="left", padx=4)
        self._sep(top)
        ttk.Button(top, text="✓ Tout",  command=self.select_all).pack(side="left", padx=2)
        ttk.Button(top, text="✗ Rien",  command=self.select_none).pack(side="left", padx=2)
        self._sep(top)

        ttk.Label(top, text="Lissage :").pack(side="left")
        self.smooth_active = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="actif", variable=self.smooth_active,
                        command=self.update_plot).pack(side="left", padx=2)
        ttk.Label(top, text="fenêtre :").pack(side="left", padx=(4, 0))
        self.smooth_win = tk.IntVar(value=5)
        wspin = ttk.Spinbox(top, from_=2, to=500, increment=1,
                            textvariable=self.smooth_win, width=5,
                            command=self.update_plot)
        wspin.pack(side="left", padx=2)
        wspin.bind("<Return>",   lambda _: self.update_plot())
        wspin.bind("<FocusOut>", lambda _: self.update_plot())
        ttk.Label(top, text="pts", style="Small.TLabel").pack(side="left", padx=(0, 4))
        self.show_raw = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="brut (α)", variable=self.show_raw,
                        command=self.update_plot).pack(side="left", padx=2)
        self._sep(top)

        ttk.Checkbutton(top, text="┆ 250 µm", variable=self.vline_250,
                        command=self.update_plot).pack(side="left", padx=4)
        self._sep(top)

        ttk.Button(top, text="🔄 Mettre à jour", command=self.update_plot).pack(side="left", padx=4)
        self._sep(top)
        ttk.Button(top, text="💾 Exporter Excel (corrigé)",
                   command=self.export).pack(side="left", padx=4)
        self._sep(top)
        ttk.Button(top, text="📊 Aires",
                   command=self.show_areas_popup).pack(side="left", padx=4)
        self._sep(top)
        ttk.Button(top, text="📄 Exporter figure (.csv)",
                   command=self.export_figure_csv).pack(side="left", padx=4)

        # ── Barre 2 : référence ────────────────────────────────────────────────
        top2 = ttk.Frame(self.root, padding=(6, 2))
        top2.pack(side="top", fill="x")

        ttk.Button(top2, text="📥 Importer référence",
                   command=self.import_reference).pack(side="left", padx=4)
        self._sep(top2)

        ttk.Label(top2, text="Référence :").pack(side="left")
        self._ref_label = ttk.Label(top2, text="(aucune)", style="Ref.TLabel",
                                    width=24, anchor="w")
        self._ref_label.pack(side="left", padx=(2, 8))

        ttk.Label(top2, text=f"Aire réf. [{NORM_XMIN:.0f}–{NORM_XMAX:.0f} µm] :").pack(side="left")
        self._ref_area_label = ttk.Label(top2, text="—", width=14,
                                         foreground="#1f77b4",
                                         font=("Helvetica", 9, "bold"))
        self._ref_area_label.pack(side="left", padx=(2, 8))
        self._sep(top2)

        self.norm_active = tk.BooleanVar(value=False)
        ttk.Checkbutton(top2, text="✓ Afficher corrigé",
                        variable=self.norm_active,
                        command=self.update_plot).pack(side="left", padx=6)
        self._sep(top2)

        ttk.Button(top2, text="✗ Retirer référence",
                   command=self._remove_reference).pack(side="left", padx=4)

        # ── Corps principal ─────────────────────────────────────────────────────
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        left = ttk.LabelFrame(paned, text=" Profils ", padding=4)
        paned.add(left, weight=1)
        self._build_profile_panel(left)

        center = ttk.Frame(paned)
        paned.add(center, weight=6)
        self._build_plot_panel(center)

        right = ttk.LabelFrame(paned, text=" Correction distance = 0 ", padding=4)
        paned.add(right, weight=2)
        self._build_correction_panel(right)

    def _sep(self, parent):
        ttk.Separator(parent, orient="vertical").pack(
            side="left", fill="y", padx=6, pady=2)

    # ── Panneau gauche ──────────────────────────────────────────────────────────
    def _build_profile_panel(self, parent):
        canvas = tk.Canvas(parent, bg="white", highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self._prof_frame = ttk.Frame(canvas)
        self._prof_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._prof_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        ttk.Label(self._prof_frame, text="(importer un fichier)",
                  foreground="gray").pack(pady=20)

    def _rebuild_profile_list(self):
        for w in self._prof_frame.winfo_children():
            w.destroy()
        if not self.sorted_names:
            ttk.Label(self._prof_frame, text="(aucun profil)",
                      foreground="gray").pack(pady=20)
            return
        for i, name in enumerate(self.sorted_names):
            depth = self.profiles[name]["depth"]
            color = PALETTE[i % len(PALETTE)]
            row = ttk.Frame(self._prof_frame)
            row.pack(fill="x", padx=4, pady=2)
            tk.Canvas(row, width=12, height=12, bg=color,
                      highlightthickness=1, highlightbackground="#888"
                      ).pack(side="left", padx=(0, 3))
            ttk.Checkbutton(row, variable=self.prof_visible[name],
                            command=self.update_plot).pack(side="left")
            ttk.Label(row, text=f"{name}  ({depth:.0f} µm)",
                      style="Small.TLabel").pack(side="left")
            ttk.Checkbutton(row, text="↔", variable=self.prof_flip[name],
                            command=self.update_plot,
                            style="Small.TCheckbutton").pack(side="left", padx=(6, 0))

    # ── Zone de tracé ──────────────────────────────────────────────────────────
    def _build_plot_panel(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self._init_axes()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.mpl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb = ttk.Frame(parent)
        tb.grid(row=1, column=0, sticky="ew")
        self.nav = NavigationToolbar2Tk(self.mpl_canvas, tb)
        self.nav.update()
        self._pick_label = ttk.Label(parent, text="", foreground="#c05000",
                                     font=("Helvetica", 9, "bold"))
        self._pick_label.grid(row=2, column=0, pady=2)
        self.mpl_canvas.mpl_connect("button_press_event", self._on_pick_click)

    def _init_axes(self):
        self.ax.set_xlabel("Distance (µm)", fontsize=11)
        self.ax.set_ylabel("Intensité (niveaux de gris)", fontsize=11)
        self.ax.set_title("Profils d'intensité", fontsize=12)
        self.ax.grid(True, alpha=0.3, linestyle="--", color="#aaa")
        self.fig.tight_layout()

    # ── Panneau droit ──────────────────────────────────────────────────────────
    def _build_correction_panel(self, parent):
        ttk.Label(
            parent,
            text="BG : background à soustraire\n"
                 "✓  activer correction Δx\n"
                 "🎯  clic sur le graphe → zéro\n"
                 "‹ ›  ajuster au pas du profil\n"
                 "L.mes : largeur mesurée → rescale → 500 µm",
            style="Small.TLabel", foreground="#555", justify="left",
        ).pack(fill="x", padx=4, pady=(0, 5))
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=2)

        hdr = ttk.Frame(parent)
        hdr.pack(fill="x", padx=4)
        for ci, (txt, w) in enumerate([
            ("Profil", 8), ("BG", 6), ("✓", 3), ("dist=0 (µm)", 8), ("🎯 ‹ ›", 7), ("L.mes (µm)", 8)
        ]):
            ttk.Label(hdr, text=txt, width=w,
                      style="Title.TLabel").grid(row=0, column=ci, sticky="w", padx=1)
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=3)

        canvas = tk.Canvas(parent, bg="white", highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self._corr_frame = ttk.Frame(canvas)
        self._corr_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._corr_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        ttk.Label(self._corr_frame, text="(importer un fichier)",
                  foreground="gray").pack(pady=20)

    def _rebuild_correction_panel(self):
        for w in self._corr_frame.winfo_children():
            w.destroy()
        if not self.sorted_names:
            ttk.Label(self._corr_frame, text="(aucun profil)",
                      foreground="gray").pack(pady=20)
            return
        for name in self.sorted_names:
            row = ttk.Frame(self._corr_frame)
            row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=name, width=9,
                      style="Small.TLabel", anchor="w").grid(row=0, column=0, sticky="w")
            bg_e = ttk.Entry(row, textvariable=self.bg_val[name], width=7)
            bg_e.grid(row=0, column=1, padx=(2, 2))
            bg_e.bind("<Return>",   lambda _: self.update_plot())
            bg_e.bind("<FocusOut>", lambda _: self.update_plot())
            ttk.Checkbutton(row, variable=self.zero_active[name],
                            command=self.update_plot).grid(row=0, column=2)
            ent = ttk.Entry(row, textvariable=self.zero_val[name], width=8)
            ent.grid(row=0, column=3, padx=(2, 0))
            ent.bind("<Return>",   lambda _: self.update_plot())
            ent.bind("<FocusOut>", lambda _: self.update_plot())
            ttk.Button(row, text="🎯", width=3,
                       command=lambda n=name: self._activate_pick(n)
                       ).grid(row=0, column=4, padx=(2, 0))
            bf = ttk.Frame(row)
            bf.grid(row=0, column=5, padx=(1, 0))
            ttk.Button(bf, text="‹", width=2,
                       command=lambda n=name: self._step_zero(n, -1)).pack(side="left")
            ttk.Button(bf, text="›", width=2,
                       command=lambda n=name: self._step_zero(n, +1)).pack(side="left")
            sc_e = ttk.Entry(row, textvariable=self.scale_val[name], width=7)
            sc_e.grid(row=0, column=6, padx=(4, 0))
            sc_e.bind("<Return>",   lambda _: self.update_plot())
            sc_e.bind("<FocusOut>", lambda _: self.update_plot())

    # ── Import profils principaux ──────────────────────────────────────────────
    def import_file(self):
        path = filedialog.askopenfilename(
            title="Importer les profils ImageJ",
            filetypes=[("Excel / CSV", "*.xlsx *.xls *.csv"), ("Tous", "*.*")],
        )
        if not path:
            return

        is_project = False
        if path.lower().endswith((".xlsx", ".xls")):
            try:
                xl = pd.ExcelFile(path)
                is_project = "Récapitulatif" in xl.sheet_names
            except Exception:
                pass

        try:
            if is_project:
                self._load_project(path)
            else:
                profiles = parse_imagej_file(path)
                self._init_from_profiles(profiles)
        except Exception as exc:
            messagebox.showerror("Erreur de lecture", str(exc))
            return

        n = len(self.profiles)
        self.root.title(
            f"Visualisateur de profils ImageJ — "
            f"{os.path.basename(path)}  ({n} profil{'s' if n > 1 else ''})"
            + ("  [PROJET]" if is_project else "")
        )

    # ── Import référence ───────────────────────────────────────────────────────
    def import_reference(self):
        path = filedialog.askopenfilename(
            title="Importer le profil de référence",
            filetypes=[("Excel / CSV", "*.xlsx *.xls *.csv"), ("Tous", "*.*")],
        )
        if not path:
            return
        try:
            ref = parse_reference_file(path)
        except Exception as exc:
            messagebox.showerror("Erreur référence", str(exc))
            return
        self.reference = ref
        self._update_ref_ui()
        self.update_plot()

    def _remove_reference(self):
        self.reference = None
        self.norm_active.set(False)
        self._ref_label.config(text="(aucune)")
        self._ref_area_label.config(text="—")
        self.update_plot()

    def _update_ref_ui(self):
        if self.reference is None:
            self._ref_label.config(text="(aucune)")
            self._ref_area_label.config(text="—")
            return
        name  = self.reference["name"]
        depth = self.reference["depth"]
        self._ref_label.config(text=f"{name}  ({depth:.0f} µm)")
        area = _trapz_interval(
            self.reference["distance"], self.reference["intensity"],
            NORM_XMIN, NORM_XMAX)
        self._ref_area_label.config(text="—" if np.isnan(area) else f"{area:.1f}")

    def _ref_area(self) -> float:
        if self.reference is None:
            return np.nan
        return _trapz_interval(
            self.reference["distance"], self.reference["intensity"],
            NORM_XMIN, NORM_XMAX)

    # ── Init / chargement ─────────────────────────────────────────────────────
    def _init_from_profiles(self, profiles: dict):
        self.profiles     = profiles
        self.sorted_names = sorted(profiles, key=lambda n: profiles[n]["depth"])
        self.prof_visible = {n: tk.BooleanVar(value=True)  for n in profiles}
        self.prof_flip    = {n: tk.BooleanVar(value=False)  for n in profiles}
        self.bg_val       = {n: tk.StringVar(value="0")     for n in profiles}
        self.zero_active  = {n: tk.BooleanVar(value=False)  for n in profiles}
        self.zero_val     = {n: tk.StringVar(value="0")     for n in profiles}
        self.scale_val    = {n: tk.StringVar(value="")      for n in profiles}
        self._refresh_ui()

    def _load_project(self, path: str):
        data        = load_project(path)
        profiles    = data["profiles"]
        corrections = data["corrections"]
        params      = data["params"]

        self.profiles     = profiles
        self.sorted_names = sorted(profiles, key=lambda n: profiles[n]["depth"])
        self.prof_visible = {n: tk.BooleanVar(value=True) for n in profiles}
        self.prof_flip    = {n: tk.BooleanVar(value=corrections[n]["flipped"]) for n in profiles}
        self.bg_val       = {n: tk.StringVar(value=str(corrections[n]["bg"])) for n in profiles}
        self.zero_active  = {n: tk.BooleanVar(value=corrections[n]["zero_active"]) for n in profiles}
        self.zero_val     = {n: tk.StringVar(value=str(corrections[n]["zero_shift"])) for n in profiles}
        self.scale_val    = {n: tk.StringVar(value=str(corrections[n].get("scale_width", ""))) for n in profiles}

        self.smooth_active.set(params["smooth_active"])
        self.smooth_win.set(params["smooth_window"])
        self.norm_active.set(params["norm_active"])

        self._refresh_ui()
        messagebox.showinfo(
            "Projet restauré",
            f"{len(profiles)} profils chargés.\n"
            "Toutes les corrections ont été restaurées.\n"
            "Ré-importez la référence si nécessaire."
        )

    def _refresh_ui(self):
        self._rebuild_profile_list()
        self._rebuild_correction_panel()
        self._update_ref_ui()
        self.update_plot()

    # ── Mode pick ──────────────────────────────────────────────────────────────
    def _activate_pick(self, name: str):
        self._pick_target = name
        self.mpl_canvas.get_tk_widget().config(cursor="crosshair")
        self._pick_label.config(
            text=f"🎯  Cliquez sur le graphe → zéro de « {name} »    [Échap = annuler]")
        self.root.bind("<Escape>", self._cancel_pick)

    def _cancel_pick(self, _=None):
        self._pick_target = None
        self.mpl_canvas.get_tk_widget().config(cursor="")
        self._pick_label.config(text="")
        self.root.unbind("<Escape>")

    def _on_pick_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self._pick_target is None:
            return
        name = self._pick_target
        self.zero_val[name].set(f"{event.xdata:.4f}")
        self.zero_active[name].set(True)
        self._cancel_pick()
        self.update_plot()

    def _step_zero(self, name: str, direction: int):
        dist = self.profiles[name]["distance"]
        if len(dist) < 2:
            return
        step    = float(dist[1] - dist[0])
        current = _try_float(self.zero_val[name].get(), 0.0)
        self.zero_val[name].set(f"{current + direction * step:.6g}")
        self.zero_active[name].set(True)
        self.update_plot()

    # ── Sélection ──────────────────────────────────────────────────────────────
    def select_all(self):
        for v in self.prof_visible.values(): v.set(True)
        self.update_plot()

    def select_none(self):
        for v in self.prof_visible.values(): v.set(False)
        self.update_plot()

    # ── Calcul des données corrigées ───────────────────────────────────────────
    def _get_corrected(self, name: str, _skip_norm: bool = False) -> dict:
        data = self.profiles[name]
        bg   = _try_float(self.bg_val.get(name, tk.StringVar(value="0")).get(), 0.0)

        dist_raw  = data["distance"].copy()
        inten_raw = data["intensity"].copy()

        dist_corr  = dist_raw.copy()
        inten_corr = inten_raw - bg

        zero_shift = 0.0
        if self.zero_active.get(name) and self.zero_active[name].get():
            zero_shift = _try_float(self.zero_val[name].get(), 0.0)
            dist_corr  = dist_corr - zero_shift

        flipped = bool(self.prof_flip.get(name) and self.prof_flip[name].get())
        if flipped:
            dist_corr = dist_corr[-1] + dist_corr[0] - dist_corr

        scale_width  = _try_float(self.scale_val.get(name, tk.StringVar(value="")).get(), 0.0)
        scale_factor = (500.0 / scale_width) if scale_width > 0 else 1.0
        if scale_factor != 1.0:
            dist_corr = dist_corr * scale_factor

        norm_factor = 1.0
        if (not _skip_norm
                and self.norm_active.get()
                and self.reference is not None):
            ref_area  = self._ref_area()
            this_area = _trapz_interval(dist_corr, inten_corr, NORM_XMIN, NORM_XMAX)
            a_ref  = abs(ref_area)
            a_this = abs(this_area)
            if (not np.isnan(a_ref) and not np.isnan(a_this)
                    and a_ref > 0 and a_this > 0):
                norm_factor = a_ref / a_this
            inten_corr = inten_corr * norm_factor

        win          = self.smooth_win.get()
        inten_smooth = smooth(inten_corr, win) if self.smooth_active.get() and win > 1 else None

        return {
            "name":             name,
            "depth":            data["depth"],
            "distance_raw":     dist_raw,
            "intensity_raw":    inten_raw,
            "distance_corr":    dist_corr,
            "intensity_corr":   inten_corr,
            "intensity_smooth": inten_smooth,
            "smooth_window":    win,
            "zero_shift":       zero_shift,
            "flipped":          flipped,
            "bg":               bg,
            "norm_factor":      norm_factor,
            "scale_width":      scale_width,
            "scale_factor":     scale_factor,
        }

    # ── Mise à jour du graphe ──────────────────────────────────────────────────
    def update_plot(self):
        self.ax.clear()
        self._init_axes()

        do_smooth = self.smooth_active.get()
        win       = self.smooth_win.get()
        show_raw  = self.show_raw.get()
        do_norm   = self.norm_active.get() and self.reference is not None

        any_bg = any(_try_float(self.bg_val[n].get(), 0) != 0
                     for n in self.sorted_names if n in self.bg_val)

        # ── Référence en pointillés noirs ──
        if self.reference is not None:
            rd = self.reference["distance"]
            ri = self.reference["intensity"]
            rl = f"[réf] {self.reference['name']}  ({self.reference['depth']:.0f} µm)"
            if do_smooth and win > 1:
                ri_s = smooth(ri, win)
                if show_raw:
                    self.ax.plot(rd, ri, color=REF_COLOR, lw=0.7, alpha=0.20, ls="--")
                self.ax.plot(rd, ri_s, color=REF_COLOR, lw=REF_LWIDTH,
                             ls="--", alpha=0.85, label=rl)
            else:
                self.ax.plot(rd, ri, color=REF_COLOR, lw=REF_LWIDTH,
                             ls="--", alpha=0.85, label=rl)

        # ── Profils principaux ──
        n_plotted = 0
        for i, name in enumerate(self.sorted_names):
            if not self.prof_visible.get(name, tk.BooleanVar(value=False)).get():
                continue
            c     = self._get_corrected(name)
            color = PALETTE[i % len(PALETTE)]
            label = f"{name}  ({c['depth']:.0f} µm)"
            if do_norm:
                label += f"  ×{c['norm_factor']:.3f}"

            if do_smooth and c["intensity_smooth"] is not None:
                if show_raw:
                    self.ax.plot(c["distance_corr"], c["intensity_corr"],
                                 color=color, lw=0.8, alpha=0.25)
                self.ax.plot(c["distance_corr"], c["intensity_smooth"],
                             color=color, lw=2.0, alpha=0.90, label=label)
            else:
                self.ax.plot(c["distance_corr"], c["intensity_corr"],
                             color=color, lw=1.6, alpha=0.85, label=label)
            n_plotted += 1

        # ── Bande de normalisation ──
        if do_norm:
            self.ax.axvspan(NORM_XMIN, NORM_XMAX, alpha=0.06, color="#2ca02c", zorder=0)
            for xv in (NORM_XMIN, NORM_XMAX):
                self.ax.axvline(xv, color="#2ca02c", lw=1.2, ls="--", alpha=0.5)

        # ── Ligne 250 µm ──
        if self.vline_250.get():
            self.ax.axvline(250, color="#d62728", lw=1.5, ls="--", alpha=0.8,
                            label="250 µm", zorder=5)

        if n_plotted or self.reference is not None:
            self.ax.legend(fontsize=8, loc="best", framealpha=0.85, edgecolor="#ccc")

        corrections_active = any(self.zero_active[n].get()
                                 for n in self.sorted_names if n in self.zero_active)
        parts = []
        if any_bg:             parts.append("bg individuel")
        if corrections_active: parts.append("Δx corrigé")
        if do_norm:            parts.append(f"normalisé [{NORM_XMIN:.0f}–{NORM_XMAX:.0f} µm]")
        if do_smooth:          parts.append(f"lissage w={win}")
        suffix = f"  [{', '.join(parts)}]" if parts else ""
        self.ax.set_title(f"Profils d'intensité{suffix}", fontsize=11)
        self.fig.tight_layout()
        self.mpl_canvas.draw()

    # ── Popup tableau des aires ────────────────────────────────────────────────
    def show_areas_popup(self):
        if not self.profiles:
            messagebox.showwarning("Aires", "Aucun profil chargé.")
            return

        rows = []
        for name in self.sorted_names:
            c     = self._get_corrected(name)
            dist  = c["distance_corr"]
            inten = c["intensity_corr"]
            rows.append({
                "name":     name,
                "depth":    c["depth"],
                "a0_250":   _trapz_interval(dist, inten, 0,   250),
                "a250_500": _trapz_interval(dist, inten, 250, 500),
            })

        win = tk.Toplevel(self.root)
        win.title("Aires sous la courbe (intensité corrigée)")
        win.geometry("680x360")
        win.resizable(True, True)

        ttk.Label(win,
                  text="Intensité corrigée (BG soustrait + Δx + flip + normalisation)",
                  font=("Helvetica", 9, "italic"), foreground="#555"
                  ).pack(pady=(8, 2))

        cols = ("Profil", "Prof. (µm)", "% aire [0 → 250 µm]", "% aire [250 → 500 µm]")
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=6)

        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        for col in cols:
            tree.heading(col, text=col)
            w = 160 if "→" in col else (110 if "Prof" in col else 90)
            tree.column(col, width=w, anchor="center")

        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        def fmt(v):
            return f"{v:.1f} %" if not np.isnan(v) else "—"

        for i, r in enumerate(rows):
            tag = "even" if i % 2 == 0 else "odd"
            total = r["a0_250"] + r["a250_500"]
            pct_0   = (r["a0_250"]   / total * 100) if total and not np.isnan(total) else np.nan
            pct_250 = (r["a250_500"] / total * 100) if total and not np.isnan(total) else np.nan
            tree.insert("", "end", tags=(tag,),
                        values=(r["name"], f"{r['depth']:.0f}", fmt(pct_0), fmt(pct_250)))

        tree.tag_configure("even", background="#f5f8ff")
        tree.tag_configure("odd",  background="white")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(0, 8))

        def copy_tsv():
            lines = ["\t".join(cols)]
            for r in rows:
                total = r["a0_250"] + r["a250_500"]
                pct_0   = (r["a0_250"]   / total * 100) if total and not np.isnan(total) else np.nan
                pct_250 = (r["a250_500"] / total * 100) if total and not np.isnan(total) else np.nan
                lines.append("\t".join([r["name"], f"{r['depth']:.0f}", fmt(pct_0), fmt(pct_250)]))
            win.clipboard_clear()
            win.clipboard_append("\n".join(lines))
            messagebox.showinfo("Copié", "Tableau copié (TSV).", parent=win)

        ttk.Button(btn_frame, text="📋 Copier (TSV)", command=copy_tsv).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Fermer", command=win.destroy).pack(side="right", padx=4)

    # ── Export figure CSV ──────────────────────────────────────────────────────
    def export_figure_csv(self):
        if not self.profiles:
            messagebox.showwarning("Export CSV", "Aucun profil chargé.")
            return
        visible = [n for n in self.sorted_names
                   if self.prof_visible.get(n, tk.BooleanVar(value=False)).get()]
        if not visible:
            messagebox.showwarning("Export CSV", "Aucun profil sélectionné.")
            return
        path = filedialog.asksaveasfilename(
            title="Enregistrer la figure en CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Tous", "*.*")],
        )
        if not path:
            return

        do_smooth = self.smooth_active.get()
        series = {}
        for name in visible:
            c     = self._get_corrected(name)
            depth = c["depth"]
            inten = c["intensity_smooth"] if (do_smooth and c["intensity_smooth"] is not None) \
                    else c["intensity_corr"]
            series[f"{name} ({depth:.0f} µm) — Distance (µm)"] = c["distance_corr"]
            series[f"{name} ({depth:.0f} µm) — Intensité"]     = inten

        max_len = max(len(v) for v in series.values())
        data_dict = {}
        for k, v in series.items():
            arr = np.full(max_len, np.nan); arr[:len(v)] = v
            data_dict[k] = arr

        pd.DataFrame(data_dict).to_csv(
            path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
        messagebox.showinfo("Export CSV réussi",
                            f"{path}\n{len(visible)} profil(s) "
                            f"({'lissés' if do_smooth else 'corrigés non lissés'}).")

    # ── Export Excel ────────────────────────────────────────────────────────────
    def export(self):
        if not self.profiles:
            messagebox.showwarning("Export", "Aucun profil chargé.")
            return
        path = filedialog.asksaveasfilename(
            title="Enregistrer le fichier Excel / projet",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("Tous", "*.*")],
        )
        if not path:
            return
        corrected = [self._get_corrected(n) for n in self.sorted_names]
        global_params = {
            "smooth_active": self.smooth_active.get(),
            "smooth_window": self.smooth_win.get(),
            "norm_active":   self.norm_active.get(),
        }
        try:
            export_excel(path, corrected, global_params)
            messagebox.showinfo(
                "Export réussi",
                f"{path}\n{len(corrected)} profil(s) exporté(s).\n"
                "Ce fichier peut être réouvert pour continuer les corrections.\n"
                "Note : la référence n'est pas sauvegardée, ré-importez-la à la réouverture."
            )
        except Exception as exc:
            messagebox.showerror("Erreur d'export", str(exc))


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
