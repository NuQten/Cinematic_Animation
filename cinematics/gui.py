import tkinter as tk
from tkinter import ttk, messagebox
from math import pi, radians, atan2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .geometry import Point2D, Segment2D, Structure2D
from .joints import (
    LiaisonBati2D,
    LiaisonPivot2D,
    LiaisonGlissiere2D,
    LiaisonPivotGlissant2D,
    LiaisonEncastrement2D,
)
from .motors import MotorRotatif2D, MotorLinear2D
from .mechanism import Mechanism
from .constants import Colors


COLORS = ["black", "royalblue", "crimson", "forestgreen", "darkorange", "purple", "teal"]


class MechanismGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Simulateur Cinématique 2D")
        self.root.geometry("1280x780")

        # Données du modèle
        self.points: dict[str, Point2D] = {}
        self.initial_coords: dict[str, tuple[float, float]] = {}
        self.segments: dict[str, Segment2D] = {}
        self.structures: dict[str, Structure2D] = {}
        self.liaisons: list[tuple[str, object]] = []
        self.motors: list[tuple[str, object]] = []

        self.anim = None
        self.meca: Mechanism | None = None

        self._build_layout()
        self._redraw_preview()

    # =========================================================================
    # CONSTRUCTION DE L'INTERFACE
    # =========================================================================
    def _build_layout(self) -> None:
        # Panneau gauche : Onglets de création
        left_frame = ttk.Frame(self.root, width=430, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        # Barre d'outils rapide en haut à gauche
        top_tools = ttk.Frame(left_frame)
        top_tools.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(top_tools, text="Charger exemple (Bielle-Manivelle)", command=self._load_example).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(top_tools, text="Tout effacer", command=self._clear_all).pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_tab_points()
        self._build_tab_segments()
        self._build_tab_structures()
        self._build_tab_liaisons()
        self._build_tab_motors()

        # Panneau droit : Graphique + Contrôles d'animation
        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Barre de contrôle de l'animation
        ctrl_bar = ttk.LabelFrame(right_frame, text="Simulation & Animation", padding=8)
        ctrl_bar.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(ctrl_bar, text="dt (s) :").pack(side=tk.LEFT)
        self.var_dt = tk.StringVar(value="0.05")
        ttk.Entry(ctrl_bar, textvariable=self.var_dt, width=6).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(ctrl_bar, text="Frames :").pack(side=tk.LEFT)
        self.var_frames = tk.StringVar(value="200")
        ttk.Entry(ctrl_bar, textvariable=self.var_frames, width=6).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Button(ctrl_bar, text="▶ Animer", command=self._start_animation).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="⏸ Pause / Reprendre", command=self._toggle_pause).pack(side=tk.LEFT, padx=3)
        ttk.Button(ctrl_bar, text="⏹ Stop & Réinitialiser", command=self._stop_animation).pack(side=tk.LEFT, padx=3)

        self.lbl_statut = ttk.Label(ctrl_bar, text="Statut : -", font=("Segoe UI", 9, "bold"))
        self.lbl_statut.pack(side=tk.RIGHT, padx=5)

        # Zone Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._is_paused = False

    # --- Onglet 1 : Points ---
    def _build_tab_points(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="1. Points")

        form = ttk.LabelFrame(tab, text="Nouveau Point2D", padding=8)
        form.pack(fill=tk.X)

        self.var_pt_name = tk.StringVar(value="O")
        self.var_pt_x = tk.StringVar(value="0.0")
        self.var_pt_y = tk.StringVar(value="0.0")

        ttk.Label(form, text="Nom :").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.var_pt_name, width=15).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(form, text="X :").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.var_pt_x, width=15).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(form, text="Y :").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.var_pt_y, width=15).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Button(form, text="Ajouter le point", command=self._add_point).grid(row=3, column=0, columnspan=2, pady=6, sticky="ew")

        self.list_points = tk.Listbox(tab, height=15)
        self.list_points.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        ttk.Button(tab, text="Supprimer sélection", command=self._del_point).pack(fill=tk.X)

    # --- Onglet 2 : Segments ---
    def _build_tab_segments(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="2. Segments")

        form = ttk.LabelFrame(tab, text="Nouveau Segment2D", padding=8)
        form.pack(fill=tk.X)

        self.cb_seg_p1 = ttk.Combobox(form, state="readonly")
        self.cb_seg_p2 = ttk.Combobox(form, state="readonly")
        self.cb_seg_color = ttk.Combobox(form, values=COLORS, state="readonly")
        self.cb_seg_color.set("black")

        ttk.Label(form, text="Point 1 :").grid(row=0, column=0, sticky="w", pady=2)
        self.cb_seg_p1.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(form, text="Point 2 :").grid(row=1, column=0, sticky="w", pady=2)
        self.cb_seg_p2.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(form, text="Couleur :").grid(row=2, column=0, sticky="w", pady=2)
        self.cb_seg_color.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Button(form, text="Ajouter le segment", command=self._add_segment).grid(row=3, column=0, columnspan=2, pady=6, sticky="ew")

        self.list_segments = tk.Listbox(tab, height=15)
        self.list_segments.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        ttk.Button(tab, text="Supprimer sélection", command=self._del_segment).pack(fill=tk.X)

    # --- Onglet 3 : Structures ---
    def _build_tab_structures(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="3. Structures")

        form = ttk.LabelFrame(tab, text="Nouvelle Structure2D", padding=8)
        form.pack(fill=tk.X)

        self.var_st_name = tk.StringVar(value="Solide_1")
        self.cb_st_color = ttk.Combobox(form, values=COLORS, state="readonly")
        self.cb_st_color.set("royalblue")

        ttk.Label(form, text="Nom :").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.var_st_name).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(form, text="Couleur :").grid(row=1, column=0, sticky="w", pady=2)
        self.cb_st_color.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(form, text="Segments (Ctrl+clic) :").grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 2))
        self.lb_st_segs = tk.Listbox(form, selectmode=tk.MULTIPLE, height=5, exportselection=False)
        self.lb_st_segs.grid(row=3, column=0, columnspan=2, sticky="ew")

        ttk.Button(form, text="Créer la structure", command=self._add_structure).grid(row=4, column=0, columnspan=2, pady=6, sticky="ew")

        self.list_structures = tk.Listbox(tab, height=10)
        self.list_structures.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        ttk.Button(tab, text="Supprimer sélection", command=self._del_structure).pack(fill=tk.X)

    # --- Onglet 4 : Liaisons ---
    def _build_tab_liaisons(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="4. Liaisons")

        form = ttk.LabelFrame(tab, text="Nouvelle Liaison", padding=8)
        form.pack(fill=tk.X)

        types = [
            "LiaisonBati2D",
            "LiaisonPivot2D",
            "LiaisonGlissiere2D",
            "LiaisonPivotGlissant2D",
            "LiaisonEncastrement2D",
        ]
        self.cb_liaison_type = ttk.Combobox(form, values=types, state="readonly")
        self.cb_liaison_type.set("LiaisonBati2D")
        self.cb_liaison_type.bind("<<ComboboxSelected>>", lambda e: self._update_liaison_fields())

        ttk.Label(form, text="Type :").grid(row=0, column=0, sticky="w", pady=2)
        self.cb_liaison_type.grid(row=0, column=1, sticky="ew", pady=2)

        self.lbl_l1 = ttk.Label(form, text="Structure 1 :")
        self.cb_l_st1 = ttk.Combobox(form, state="readonly")
        self.lbl_l1.grid(row=1, column=0, sticky="w", pady=2)
        self.cb_l_st1.grid(row=1, column=1, sticky="ew", pady=2)

        self.lbl_l2 = ttk.Label(form, text="Structure 2 :")
        self.cb_l_st2 = ttk.Combobox(form, state="readonly")
        self.lbl_l2.grid(row=2, column=0, sticky="w", pady=2)
        self.cb_l_st2.grid(row=2, column=1, sticky="ew", pady=2)

        self.lbl_l3 = ttk.Label(form, text="Point :")
        self.cb_l_pt1 = ttk.Combobox(form, state="readonly")
        self.lbl_l3.grid(row=3, column=0, sticky="w", pady=2)
        self.cb_l_pt1.grid(row=3, column=1, sticky="ew", pady=2)

        self.lbl_l4 = ttk.Label(form, text="Angle bâti (deg) :")
        self.var_l_angle = tk.StringVar(value="0.0")
        self.entry_l_angle = ttk.Entry(form, textvariable=self.var_l_angle)
        self.cb_l_seg = ttk.Combobox(form, state="readonly")
        self.lbl_l4.grid(row=4, column=0, sticky="w", pady=2)
        self.entry_l_angle.grid(row=4, column=1, sticky="ew", pady=2)

        ttk.Button(form, text="Ajouter la liaison", command=self._add_liaison).grid(row=5, column=0, columnspan=2, pady=6, sticky="ew")

        self.list_liaisons = tk.Listbox(tab, height=11)
        self.list_liaisons.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        ttk.Button(tab, text="Supprimer sélection", command=self._del_liaison).pack(fill=tk.X)

        self._update_liaison_fields()

    # --- Onglet 5 : Moteurs ---
    def _build_tab_motors(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="5. Moteurs")

        form = ttk.LabelFrame(tab, text="Nouveau Moteur", padding=8)
        form.pack(fill=tk.X)

        self.cb_mot_type = ttk.Combobox(form, values=["MotorRotatif2D", "MotorLinear2D"], state="readonly")
        self.cb_mot_type.set("MotorRotatif2D")
        self.cb_mot_type.bind("<<ComboboxSelected>>", lambda e: self._update_motor_fields())

        ttk.Label(form, text="Type :").grid(row=0, column=0, sticky="w", pady=2)
        self.cb_mot_type.grid(row=0, column=1, sticky="ew", pady=2)

        # Champs Rotatif
        self.lbl_m_seg = ttk.Label(form, text="Segment moteur :")
        self.cb_m_seg = ttk.Combobox(form, state="readonly")

        # Champs Linéaire
        self.lbl_m_p1 = ttk.Label(form, text="Point 1 :")
        self.cb_m_p1 = ttk.Combobox(form, state="readonly")
        self.lbl_m_p2 = ttk.Label(form, text="Point 2 :")
        self.cb_m_p2 = ttk.Combobox(form, state="readonly")

        # Vitesse & Bornes
        self.lbl_m_vit = ttk.Label(form, text="Vitesse (rad/s) :")
        self.var_m_vit = tk.StringVar(value="1.57")
        self.entry_m_vit = ttk.Entry(form, textvariable=self.var_m_vit)

        self.lbl_m_min = ttk.Label(form, text="Course min (opt) :")
        self.var_m_min = tk.StringVar(value="")
        self.entry_m_min = ttk.Entry(form, textvariable=self.var_m_min)

        self.lbl_m_max = ttk.Label(form, text="Course max (opt) :")
        self.var_m_max = tk.StringVar(value="")
        self.entry_m_max = ttk.Entry(form, textvariable=self.var_m_max)

        self.btn_add_mot = ttk.Button(form, text="Ajouter le moteur", command=self._add_motor)

        self.list_motors = tk.Listbox(tab, height=11)
        self.list_motors.pack(fill=tk.BOTH, expand=True, pady=(8, 4))
        ttk.Button(tab, text="Supprimer sélection", command=self._del_motor).pack(fill=tk.X)

        self._update_motor_fields()

    # =========================================================================
    # LOGIQUE DE MISE À JOUR DES CHAMPS DYNAMIQUES
    # =========================================================================
    def _update_liaison_fields(self) -> None:
        ltype = self.cb_liaison_type.get()
        self.cb_l_st2.grid_remove()
        self.lbl_l2.grid_remove()
        self.entry_l_angle.grid_remove()
        self.cb_l_seg.grid_remove()
        self.lbl_l4.grid_remove()

        if ltype == "LiaisonBati2D":
            self.lbl_l1.config(text="Structure :")
            self.lbl_l3.config(text="Point d'ancrage :")
            self.lbl_l4.config(text="Angle bâti (deg) :")
            self.lbl_l4.grid(row=4, column=0, sticky="w", pady=2)
            self.entry_l_angle.grid(row=4, column=1, sticky="ew", pady=2)

        elif ltype in ("LiaisonPivot2D", "LiaisonEncastrement2D"):
            self.lbl_l1.config(text="Structure 1 :")
            self.lbl_l2.config(text="Structure 2 :")
            self.lbl_l2.grid(row=2, column=0, sticky="w", pady=2)
            self.cb_l_st2.grid(row=2, column=1, sticky="ew", pady=2)
            self.lbl_l3.config(text="Point commun :")

        elif ltype in ("LiaisonGlissiere2D", "LiaisonPivotGlissant2D"):
            self.lbl_l1.config(text="Structure rail :")
            self.lbl_l2.config(text="Structure mobile :")
            self.lbl_l2.grid(row=2, column=0, sticky="w", pady=2)
            self.cb_l_st2.grid(row=2, column=1, sticky="ew", pady=2)
            self.lbl_l3.config(text="Point coulissant :")
            self.lbl_l4.config(text="Segment rail :")
            self.lbl_l4.grid(row=4, column=0, sticky="w", pady=2)
            self.cb_l_seg.grid(row=4, column=1, sticky="ew", pady=2)

    def _update_motor_fields(self) -> None:
        for w in (
            self.lbl_m_seg, self.cb_m_seg,
            self.lbl_m_p1, self.cb_m_p1,
            self.lbl_m_p2, self.cb_m_p2,
            self.lbl_m_vit, self.entry_m_vit,
            self.lbl_m_min, self.entry_m_min,
            self.lbl_m_max, self.entry_m_max,
            self.btn_add_mot,
        ):
            w.grid_remove()

        if self.cb_mot_type.get() == "MotorRotatif2D":
            self.lbl_m_seg.grid(row=1, column=0, sticky="w", pady=2)
            self.cb_m_seg.grid(row=1, column=1, sticky="ew", pady=2)
            self.lbl_m_vit.config(text="Vitesse (rad/s) :")
            self.lbl_m_vit.grid(row=2, column=0, sticky="w", pady=2)
            self.entry_m_vit.grid(row=2, column=1, sticky="ew", pady=2)
            self.btn_add_mot.grid(row=3, column=0, columnspan=2, pady=6, sticky="ew")
        else:
            self.lbl_m_p1.grid(row=1, column=0, sticky="w", pady=2)
            self.cb_m_p1.grid(row=1, column=1, sticky="ew", pady=2)
            self.lbl_m_p2.grid(row=2, column=0, sticky="w", pady=2)
            self.cb_m_p2.grid(row=2, column=1, sticky="ew", pady=2)
            self.lbl_m_vit.config(text="Vitesse (u/s) :")
            self.lbl_m_vit.grid(row=3, column=0, sticky="w", pady=2)
            self.entry_m_vit.grid(row=3, column=1, sticky="ew", pady=2)
            self.lbl_m_min.grid(row=4, column=0, sticky="w", pady=2)
            self.entry_m_min.grid(row=4, column=1, sticky="ew", pady=2)
            self.lbl_m_max.grid(row=5, column=0, sticky="w", pady=2)
            self.entry_m_max.grid(row=5, column=1, sticky="ew", pady=2)
            self.btn_add_mot.grid(row=6, column=0, columnspan=2, pady=6, sticky="ew")

    def _refresh_selectors(self) -> None:
        pt_names = list(self.points.keys())
        seg_names = list(self.segments.keys())
        st_names = list(self.structures.keys())

        for cb in (self.cb_seg_p1, self.cb_seg_p2, self.cb_l_pt1, self.cb_m_p1, self.cb_m_p2):
            cb["values"] = pt_names
        for cb in (self.cb_l_seg, self.cb_m_seg):
            cb["values"] = seg_names
        for cb in (self.cb_l_st1, self.cb_l_st2):
            cb["values"] = st_names

        self.lb_st_segs.delete(0, tk.END)
        for s in seg_names:
            self.lb_st_segs.insert(tk.END, s)

    # =========================================================================
    # ACTIONS D'AJOUT / SUPPRESSION
    # =========================================================================
    def _add_point(self) -> None:
        try:
            name = self.var_pt_name.get().strip()
            if not name or name in self.points:
                raise ValueError("Nom de point vide ou déjà utilisé.")
            x = float(self.var_pt_x.get())
            y = float(self.var_pt_y.get())
            self.points[name] = Point2D(x, y, name=name)
            self.initial_coords[name] = (x, y)
            self.list_points.insert(tk.END, f"{name} ({x}, {y})")
            self._refresh_selectors()
            self._redraw_preview()
        except Exception as e:
            messagebox.showerror("Erreur Point", str(e))

    def _del_point(self) -> None:
        sel = self.list_points.curselection()
        if not sel:
            return
        name = list(self.points.keys())[sel[0]]
        del self.points[name]
        del self.initial_coords[name]
        self.list_points.delete(sel[0])
        self._refresh_selectors()
        self._redraw_preview()

    def _add_segment(self) -> None:
        try:
            n1, n2 = self.cb_seg_p1.get(), self.cb_seg_p2.get()
            if not n1 or not n2:
                raise ValueError("Sélectionnez deux points.")
            seg = Segment2D(self.points[n1], self.points[n2], color=self.cb_seg_color.get())
            if seg.name in self.segments:
                raise ValueError("Ce segment existe déjà.")
            self.segments[seg.name] = seg
            self.list_segments.insert(tk.END, f"{seg.name} (L={seg.longeur:.2f})")
            self._refresh_selectors()
            self._redraw_preview()
        except Exception as e:
            messagebox.showerror("Erreur Segment", str(e))

    def _del_segment(self) -> None:
        sel = self.list_segments.curselection()
        if not sel:
            return
        name = list(self.segments.keys())[sel[0]]
        del self.segments[name]
        self.list_segments.delete(sel[0])
        self._refresh_selectors()
        self._redraw_preview()

    def _add_structure(self) -> None:
        try:
            name = self.var_st_name.get().strip()
            if not name or name in self.structures:
                raise ValueError("Nom de structure vide ou déjà utilisé.")
            indices = self.lb_st_segs.curselection()
            if not indices:
                raise ValueError("Sélectionnez au moins un segment.")
            seg_keys = list(self.segments.keys())
            selected_segs = [self.segments[seg_keys[i]] for i in indices]
            st = Structure2D(*selected_segs, name=name, color=self.cb_st_color.get())
            self.structures[name] = st
            self.list_structures.insert(tk.END, f"{name} : {[s.name for s in selected_segs]}")
            self._refresh_selectors()
            self._redraw_preview()
        except Exception as e:
            messagebox.showerror("Erreur Structure", str(e))

    def _del_structure(self) -> None:
        sel = self.list_structures.curselection()
        if not sel:
            return
        name = list(self.structures.keys())[sel[0]]
        del self.structures[name]
        self.list_structures.delete(sel[0])
        self._refresh_selectors()
        self._redraw_preview()

    def _add_liaison(self) -> None:
        try:
            ltype = self.cb_liaison_type.get()
            st1 = self.structures[self.cb_l_st1.get()]
            pt1 = self.points[self.cb_l_pt1.get()]

            if ltype == "LiaisonBati2D":
                ang = radians(float(self.var_l_angle.get()))
                liaison = LiaisonBati2D(st1, pt1, orientation=ang)
                desc = f"Bâti : {st1.name} en {pt1.name}"
            elif ltype == "LiaisonPivot2D":
                st2 = self.structures[self.cb_l_st2.get()]
                liaison = LiaisonPivot2D(st1, st2, point1=pt1)
                desc = f"Pivot : {st1.name} / {st2.name} en {pt1.name}"
            elif ltype == "LiaisonEncastrement2D":
                st2 = self.structures[self.cb_l_st2.get()]
                liaison = LiaisonEncastrement2D(st1, st2, point1=pt1)
                desc = f"Encastrement : {st1.name} / {st2.name} en {pt1.name}"
            elif ltype == "LiaisonGlissiere2D":
                st2 = self.structures[self.cb_l_st2.get()]
                seg = self.segments[self.cb_l_seg.get()]
                liaison = LiaisonGlissiere2D(st1, st2, seg, point_coulissant=pt1)
                desc = f"Glissière : {st2.name} sur {seg.name} (pt {pt1.name})"
            else:
                st2 = self.structures[self.cb_l_st2.get()]
                seg = self.segments[self.cb_l_seg.get()]
                liaison = LiaisonPivotGlissant2D(st1, st2, seg, point_coulissant=pt1)
                desc = f"Pivot-Glissant : {st2.name} sur {seg.name} (pt {pt1.name})"

            self.liaisons.append((desc, liaison))
            self.list_liaisons.insert(tk.END, desc)
            self._redraw_preview()
        except Exception as e:
            messagebox.showerror("Erreur Liaison", str(e))

    def _del_liaison(self) -> None:
        sel = self.list_liaisons.curselection()
        if not sel:
            return
        self.liaisons.pop(sel[0])
        self.list_liaisons.delete(sel[0])
        self._redraw_preview()

    def _add_motor(self) -> None:
        try:
            mtype = self.cb_mot_type.get()
            vit = float(self.var_m_vit.get())
            if mtype == "MotorRotatif2D":
                seg = self.segments[self.cb_m_seg.get()]
                u = seg.vector_director
                ang_init = atan2(u[1], u[0])
                mot = MotorRotatif2D(seg, angle_initial=ang_init, vitesse_angulaire=vit)
                desc = f"Rotatif sur {seg.name} (ω={vit:.2f} rad/s)"
            else:
                p1 = self.points[self.cb_m_p1.get()]
                p2 = self.points[self.cb_m_p2.get()]
                c_min = float(self.var_m_min.get()) if self.var_m_min.get().strip() else None
                c_max = float(self.var_m_max.get()) if self.var_m_max.get().strip() else None
                mot = MotorLinear2D(p1, p2, vitesse_lineaire=vit, course_min=c_min, course_max=c_max)
                mot.course_min = c_min
                mot.course_max = c_max
                desc = f"Linéaire {p1.name}-{p2.name} (v={vit:.2f})"

            self.motors.append((desc, mot))
            self.list_motors.insert(tk.END, desc)
            self._redraw_preview()
        except Exception as e:
            messagebox.showerror("Erreur Moteur", str(e))

    def _del_motor(self) -> None:
        sel = self.list_motors.curselection()
        if not sel:
            return
        self.motors.pop(sel[0])
        self.list_motors.delete(sel[0])
        self._redraw_preview()

    # =========================================================================
    # PRÉVISUALISATION ET ANIMATION
    # =========================================================================
    def _redraw_preview(self) -> None:
        self.ax.clear()
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.set_aspect("equal", adjustable="box")

        # Dessin des structures ou segments isolés
        if self.structures:
            for st in self.structures.values():
                st.draw(self.ax)
        else:
            for seg in self.segments.values():
                seg.draw(self.ax)

        # Dessin des liaisons
        for _, liaison in self.liaisons:
            liaison.draw(self.ax)

        # Dessin des points et de leurs noms
        for name, pt in self.points.items():
            self.ax.plot(pt.x, pt.y, "ko", markersize=4)
            self.ax.annotate(name, (pt.x, pt.y), textcoords="offset points", xytext=(6, 6))

        # Mise à jour du statut cinématique si le mécanisme est assemblable
        if self.structures and self.liaisons:
            try:
                meca = Mechanism(
                    list(self.structures.values()),
                    [l for _, l in self.liaisons],
                    [m for _, m in self.motors],
                )
                self.lbl_statut.config(text=f"Statut : {meca.state.name}")
            except Exception:
                self.lbl_statut.config(text="Statut : Incomplet")
        else:
            self.lbl_statut.config(text="Statut : -")

        self.ax.autoscale()
        self.ax.margins(0.2)
        self.canvas.draw()

    def _start_animation(self) -> None:
        if not self.structures or not self.liaisons or not self.motors:
            messagebox.showwarning(
                "Mécanisme incomplet",
                "Il faut au moins une structure, une liaison et un moteur pour lancer l'animation.",
            )
            return

        self._stop_animation()
        try:
            dt = float(self.var_dt.get())
            frames = int(self.var_frames.get())

            self.meca = Mechanism(
                structures=list(self.structures.values()),
                liaisons=[l for _, l in self.liaisons],
                motors=[m for _, m in self.motors],
            )
            self._is_paused = False
            self.anim = self.meca.animate(self.ax, dt=dt, frames=frames, repeat=True)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Erreur d'animation", str(e))

    def _toggle_pause(self) -> None:
        if self.anim is None:
            return
        if self._is_paused:
            self.anim.resume()
            self._is_paused = False
        else:
            self.anim.pause()
            self._is_paused = True

    def _stop_animation(self) -> None:
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

        # Restauration des positions initiales des points
        for name, (x0, y0) in self.initial_coords.items():
            if name in self.points:
                self.points[name].x = x0
                self.points[name].y = y0

        # Réinitialisation des consignes initiales des moteurs
        for _, mot in self.motors:
            if isinstance(mot, MotorRotatif2D):
                u = mot.segment.vector_director
                mot.set_target(atan2(u[1], u[0]))
            elif isinstance(mot, MotorLinear2D):
                mot.set_target(mot.point1.distance(mot.point2))

        self._redraw_preview()

    # =========================================================================
    # EXEMPLE PRÉ-REMPLI & RÉINITIALISATION
    # =========================================================================
    def _clear_all(self) -> None:
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        self.points.clear()
        self.initial_coords.clear()
        self.segments.clear()
        self.structures.clear()
        self.liaisons.clear()
        self.motors.clear()

        for lb in (self.list_points, self.list_segments, self.list_structures, self.list_liaisons, self.list_motors):
            lb.delete(0, tk.END)

        self._refresh_selectors()
        self._redraw_preview()

    def _load_example(self) -> None:
        self._clear_all()

        # Points (Bielle-Manivelle)
        for name, x, y in [("O", 0.0, 0.0), ("A", 0.0, 2.0), ("B", 4.0, 0.0), ("R", 6.0, 0.0)]:
            self.points[name] = Point2D(x, y, name=name)
            self.initial_coords[name] = (x, y)
            self.list_points.insert(tk.END, f"{name} ({x}, {y})")

        # Segments
        for p1, p2, col in [("O", "R", "black"), ("O", "A", "royalblue"), ("A", "B", "crimson")]:
            seg = Segment2D(self.points[p1], self.points[p2], color=col)
            self.segments[seg.name] = seg
            self.list_segments.insert(tk.END, f"{seg.name} (L={seg.longeur:.2f})")

        # Structures
        for st_name, seg_name, col in [
            ("Bati", "[O R]", "black"),
            ("Manivelle", "[O A]", "royalblue"),
            ("Bielle", "[A B]", "crimson"),
        ]:
            st = Structure2D(self.segments[seg_name], name=st_name, color=col)
            self.structures[st_name] = st
            self.list_structures.insert(tk.END, f"{st_name} : ['{seg_name}']")

        # Liaisons
        bati = self.structures["Bati"]
        manivelle = self.structures["Manivelle"]
        bielle = self.structures["Bielle"]

        liaisons_demo = [
            ("Bâti : Bati en R", LiaisonBati2D(bati, self.points["R"], orientation=pi / 4)),
            ("Pivot : Bati / Manivelle en O", LiaisonPivot2D(bati, manivelle, point1=self.points["O"])),
            ("Pivot : Manivelle / Bielle en A", LiaisonPivot2D(manivelle, bielle, point1=self.points["A"])),
            ("Pivot-Glissant : Bielle sur [O R] (pt B)", LiaisonPivotGlissant2D(bati, bielle, self.segments["[O R]"], point_coulissant=self.points["B"])),
        ]
        for desc, l in liaisons_demo:
            self.liaisons.append((desc, l))
            self.list_liaisons.insert(tk.END, desc)

        # Moteur
        mot = MotorRotatif2D(self.segments["[O A]"], angle_initial=pi / 2, vitesse_angulaire=pi / 2)
        desc_m = "Rotatif sur [O A] (ω=1.57 rad/s)"
        self.motors.append((desc_m, mot))
        self.list_motors.insert(tk.END, desc_m)

        self._refresh_selectors()
        self._redraw_preview()


if __name__ == "__main__":
    root = tk.Tk()
    app = MechanismGUI(root)
    root.mainloop()