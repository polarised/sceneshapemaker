#!/usr/bin/env python3
"""
Synthetic 3D dataset generator with GUI controls.
Each scene is rendered from 3 configurable camera angles.
"""

import json
import os
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from trimesh import creation

# ── Palette / sizes ───────────────────────────────────────────────────────────

COLOR_PALETTE = {
    "red":     [255,   0,   0],
    "blue":    [  0,   0, 255],
    "green":   [  0, 200,   0],
    "yellow":  [255, 220,   0],
    "cyan":    [  0, 210, 210],
    "magenta": [220,   0, 220],
    "orange":  [255, 140,   0],
    "purple":  [130,   0, 160],
    "white":   [230, 230, 230],
    "teal":    [  0, 160, 140],
}

SIZE_MAP = {"small": 0.6, "large": 1.3}
SHAPE_NAMES = ["cube", "sphere", "cylinder", "pyramid"]


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class GenConfig:
    output_dir: str = "synthetic_3d_dataset"
    num_images: int = 10
    resolution: Tuple[int, int] = (512, 512)

    shape_counts: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        n: (0, 2) for n in SHAPE_NAMES
    })

    shape_spread: float = 4.0
    min_distance: float = 2.0       # lower → more occlusion between objects
    camera_distance: float = 2.5    # normalized Plotly units; ≥2.0 keeps all objects in frame
    cam_el_min: float = 30.0        # minimum camera elevation (degrees)
    cam_el_max: float = 60.0        # maximum camera elevation (degrees)

    # Augmentation toggles
    vary_lighting:     bool = True   # ambient / diffuse / specular jitter
    vary_background:   bool = True   # background grey level
    vary_floor:        bool = True   # floor RGB colour
    vary_scale:        bool = True   # ±25 % scale jitter within size category
    vary_rotation_3d:  bool = True   # small pitch / roll in addition to yaw
    vary_color:        bool = True   # ±25 RGB jitter on each shape colour
    vary_cam_distance: bool = True   # ±10 % zoom jitter per scene
    vary_floor_pattern: bool = True  # checker / stripes / noise floor texture


# ── Shape factories ───────────────────────────────────────────────────────────

def _make_cube(s):     return creation.box(extents=[s, s, s])
def _make_sphere(s):   return creation.icosphere(subdivisions=3, radius=s * 0.5)
def _make_cylinder(s): return creation.cylinder(radius=s * 0.5, height=s, sections=32)
def _make_pyramid(s):  return creation.cone(radius=s * 0.5, height=s, sections=4)

_FACTORIES = {
    "cube": _make_cube, "sphere": _make_sphere,
    "cylinder": _make_cylinder, "pyramid": _make_pyramid,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_color(jitter: bool = False) -> Tuple[str, List[int]]:
    name = np.random.choice(list(COLOR_PALETTE))
    rgb  = np.array(COLOR_PALETTE[name], dtype=float)
    if jitter:
        rgb = np.clip(rgb + np.random.uniform(-25, 25, 3), 0, 255)
    return name, [int(v) for v in rgb] + [255]


def _ground(mesh: trimesh.Trimesh, floor_z: float = 0.0) -> None:
    mesh.apply_translation([0, 0, floor_z + 0.01 - mesh.bounds[0, 2]])


def _rotate(mesh: trimesh.Trimesh, full_3d: bool = False) -> None:
    R = trimesh.transformations.rotation_matrix
    mesh.apply_transform(R(np.random.uniform(0, 2 * np.pi), [0, 0, 1]))
    if full_3d:
        mesh.apply_transform(R(np.random.uniform(-0.2, 0.2), [1, 0, 0]))
        mesh.apply_transform(R(np.random.uniform(-0.2, 0.2), [0, 1, 0]))


# ── Scene generation ──────────────────────────────────────────────────────────

def generate_scene(cfg: GenConfig):
    meshes, labels, metadata, placed = [], [], [], []

    for name in SHAPE_NAMES:
        lo, hi = cfg.shape_counts[name]
        count = int(np.random.randint(lo, hi + 1))
        for _ in range(count):
            size_name = np.random.choice(["small", "large"])
            scale = SIZE_MAP[size_name]
            if cfg.vary_scale:
                scale *= np.random.uniform(0.75, 1.25)

            color_name, rgba = _random_color(jitter=cfg.vary_color)
            mesh = _FACTORIES[name](scale)
            mesh.visual.face_colors = rgba

            _rotate(mesh, full_3d=cfg.vary_rotation_3d)

            for _ in range(200):
                xy = np.random.uniform(-cfg.shape_spread, cfg.shape_spread, 2)
                if all(np.linalg.norm(xy - p) >= cfg.min_distance for p in placed):
                    placed.append(xy)
                    break
            else:
                continue  # skip shape if no valid position found

            mesh.apply_translation([xy[0], xy[1], 0])
            _ground(mesh)
            meshes.append(mesh)
            labels.append(name)
            metadata.append({
                "shape": name, "color": color_name,
                "size": size_name, "scale": round(scale, 3),
            })

    # Guarantee at least one object
    if not meshes:
        color_name, rgba = _random_color()
        m = _make_cube(SIZE_MAP["small"])
        m.visual.face_colors = rgba
        _rotate(m)
        _ground(m)
        meshes.append(m)
        labels.append("cube")
        metadata.append({"shape": "cube", "color": color_name,
                          "size": "small", "scale": SIZE_MAP["small"]})

    # Re-centre XY so the camera target (0,0) is always correct for every angle
    all_xy = np.mean([m.vertices[:, :2].mean(axis=0) for m in meshes], axis=0)
    for m in meshes:
        m.apply_translation([-all_xy[0], -all_xy[1], 0])

    return meshes, labels, metadata


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_scene(
    meshes: list,
    img_path: str,
    cfg: GenConfig,
    scene_rng_state: dict,
) -> None:
    import plotly.graph_objects as go

    bg_v     = scene_rng_state["bg_v"]
    floor_c  = scene_rng_state["floor_c"]
    ambient  = scene_rng_state["ambient"]
    diffuse  = scene_rng_state["diffuse"]
    specular = scene_rng_state["specular"]

    floor_c2      = scene_rng_state["floor_c2"]
    floor_pattern = scene_rng_state["floor_pattern"]

    bg = f"rgb({bg_v},{bg_v},{bg_v})"

    cam_d = cfg.camera_distance
    if cfg.vary_cam_distance:
        cam_d *= np.random.uniform(0.90, 1.10)

    # Random single camera per scene, elevation clamped to always see all objects
    az = np.random.uniform(0, 2 * np.pi)
    el = np.radians(np.random.uniform(cfg.cam_el_min, cfg.cam_el_max))
    d  = cam_d

    # Look at the vertical centre of the scene (objects sit on the floor)
    cx, cy, cz = 0.0, 0.0, 0.5

    fig = go.Figure()

    # Textured floor via Surface trace
    floor_side = cfg.shape_spread * 2.5
    N = 60
    half = floor_side / 2
    xs = np.linspace(-half, half, N)
    ys = np.linspace(-half, half, N)
    Xf, Yf = np.meshgrid(xs, ys)
    Zf = np.full_like(Xf, -0.01)

    freq = np.random.randint(3, 8)
    if floor_pattern == "checker":
        sc = ((np.floor(Xf / (floor_side / freq)) + np.floor(Yf / (floor_side / freq))) % 2)
    elif floor_pattern == "stripes_x":
        sc = (np.sin(Xf * freq * np.pi / floor_side) > 0).astype(float)
    elif floor_pattern == "stripes_y":
        sc = (np.sin(Yf * freq * np.pi / floor_side) > 0).astype(float)
    elif floor_pattern == "noise":
        sc = np.random.uniform(0, 1, (N, N))
    else:
        sc = np.zeros((N, N))

    c1 = f"rgb({floor_c[0]},{floor_c[1]},{floor_c[2]})"
    c2 = f"rgb({floor_c2[0]},{floor_c2[1]},{floor_c2[2]})"
    fig.add_trace(go.Surface(
        x=Xf, y=Yf, z=Zf, surfacecolor=sc,
        colorscale=[[0.0, c1], [1.0, c2]],
        cmin=0, cmax=1, showscale=False, opacity=1.0,
    ))

    for mesh in meshes:
        c = mesh.visual.face_colors[0]
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            color=f"rgb({c[0]},{c[1]},{c[2]})", opacity=1.0, showlegend=False,
            lighting=dict(ambient=ambient, diffuse=diffuse, specular=specular),
        ))
        v = mesh.vertices
        ex, ey, ez = [], [], []
        for edge in mesh.edges_unique:
            ex += [v[edge[0], 0], v[edge[1], 0], None]
            ey += [v[edge[0], 1], v[edge[1], 1], None]
            ez += [v[edge[0], 2], v[edge[1], 2], None]
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="rgb(30,30,30)", width=2),
            showlegend=False, hoverinfo="none",
        ))

    fig.update_layout(
        scene_camera=dict(
            eye=dict(
                x=cx + d * np.cos(el) * np.cos(az),
                y=cy + d * np.cos(el) * np.sin(az),
                z=cz + d * np.sin(el),
            ),
            center=dict(x=cx, y=cy, z=cz),
            up=dict(x=0, y=0, z=1),
        ),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor=bg,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=bg,
    )

    fig.write_image(img_path, width=cfg.resolution[0], height=cfg.resolution[1])


# ── Generation runner ─────────────────────────────────────────────────────────

def run_generation(cfg: GenConfig, progress_cb=None, done_cb=None, error_cb=None) -> None:
    import csv
    try:
        os.makedirs(cfg.output_dir, exist_ok=True)

        csv_path = os.path.join(cfg.output_dir, "labels.csv")
        fieldnames = ["filename"] + SHAPE_NAMES + ["objects"]
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                start_idx = sum(1 for _ in csv.reader(f)) - 1  # rows minus header
            csv_mode   = "a"
            write_header = False
        else:
            start_idx    = 0
            csv_mode     = "w"
            write_header = True

        rows = []
        for i in range(cfg.num_images):
            meshes, labels, metadata = generate_scene(cfg)

            bg_v = int(np.random.randint(200, 256)) if cfg.vary_background else 255
            floor_c = (
                [int(np.random.randint(160, 230)) for _ in range(3)]
                if cfg.vary_floor else [220, 220, 220]
            )
            ambient  = float(np.random.uniform(0.55, 1.0)) if cfg.vary_lighting else 1.0
            diffuse  = float(np.random.uniform(0.0, 0.45)) if cfg.vary_lighting else 0.0
            specular = float(np.random.uniform(0.0, 0.25)) if cfg.vary_lighting else 0.0
            _patterns = ["solid", "solid", "checker", "stripes_x", "stripes_y", "noise"]
            floor_pattern = np.random.choice(_patterns) if cfg.vary_floor_pattern else "solid"
            floor_c2 = [int(np.random.randint(130, 210)) for _ in range(3)]
            scene_rng_state = dict(bg_v=bg_v, floor_c=floor_c, floor_c2=floor_c2,
                                   floor_pattern=floor_pattern,
                                   ambient=ambient, diffuse=diffuse, specular=specular)

            fname = f"scene_{start_idx + i:05d}.png"
            render_scene(meshes, os.path.join(cfg.output_dir, fname), cfg, scene_rng_state)

            rows.append({
                "filename": fname,
                **{name: labels.count(name) for name in SHAPE_NAMES},
                "objects": json.dumps(metadata),
            })

            if progress_cb:
                progress_cb(i + 1, cfg.num_images)

        with open(csv_path, csv_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    except Exception as exc:
        if error_cb:
            error_cb(str(exc))
        return

    if done_cb:
        done_cb()


# ── GUI helpers ───────────────────────────────────────────────────────────────

class _Spin(ttk.Frame):
    """Spinbox with an optional unit label."""
    def __init__(self, parent, var, lo, hi, inc=1, width=6, unit="", **kw):
        super().__init__(parent, **kw)
        ttk.Spinbox(self, from_=lo, to=hi, increment=inc,
                    textvariable=var, width=width).pack(side="left")
        if unit:
            ttk.Label(self, text=unit).pack(side="left", padx=(2, 0))


# ── Main application ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Synthetic 3D Dataset Generator")
        self.resizable(False, False)
        self._q: queue.Queue = queue.Queue()
        self._build_ui()
        self._poll()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        P = dict(padx=8, pady=4)

        # ── Output ────────────────────────────────────────────────────────────
        f = ttk.LabelFrame(self, text="Output")
        f.grid(row=0, column=0, columnspan=2, sticky="ew", **P)

        ttk.Label(f, text="Directory:").grid(row=0, column=0, sticky="w", **P)
        self.v_dir = tk.StringVar(value="synthetic_3d_dataset")
        ttk.Entry(f, textvariable=self.v_dir, width=32).grid(row=0, column=1, sticky="ew", **P)
        ttk.Button(f, text="Browse…", command=self._browse).grid(row=0, column=2, **P)

        ttk.Label(f, text="Scenes:").grid(row=1, column=0, sticky="w", **P)
        self.v_num = tk.IntVar(value=10)
        ttk.Spinbox(f, from_=1, to=99999, textvariable=self.v_num, width=8).grid(row=1, column=1, sticky="w", **P)

        ttk.Label(f, text="Resolution:").grid(row=2, column=0, sticky="w", **P)
        rf = ttk.Frame(f)
        rf.grid(row=2, column=1, sticky="w", **P)
        self.v_rw = tk.IntVar(value=512)
        self.v_rh = tk.IntVar(value=512)
        ttk.Spinbox(rf, from_=128, to=2048, increment=64, textvariable=self.v_rw, width=6).pack(side="left")
        ttk.Label(rf, text=" × ").pack(side="left")
        ttk.Spinbox(rf, from_=128, to=2048, increment=64, textvariable=self.v_rh, width=6).pack(side="left")

        # ── Per-shape counts ──────────────────────────────────────────────────
        fs = ttk.LabelFrame(self, text="Objects per scene  (min – max)")
        fs.grid(row=1, column=0, sticky="nsew", **P)

        self.v_min: Dict[str, tk.IntVar] = {}
        self.v_max: Dict[str, tk.IntVar] = {}
        for r, name in enumerate(SHAPE_NAMES):
            ttk.Label(fs, text=name.capitalize() + ":").grid(row=r, column=0, sticky="w", padx=8, pady=3)
            mn, mx = tk.IntVar(value=0), tk.IntVar(value=2)
            ttk.Spinbox(fs, from_=0, to=20, textvariable=mn, width=4).grid(row=r, column=1, padx=4)
            ttk.Label(fs, text="–").grid(row=r, column=2)
            ttk.Spinbox(fs, from_=0, to=20, textvariable=mx, width=4).grid(row=r, column=3, padx=4)
            self.v_min[name] = mn
            self.v_max[name] = mx

        # ── Layout & camera ───────────────────────────────────────────────────
        fl = ttk.LabelFrame(self, text="Layout & camera")
        fl.grid(row=1, column=1, sticky="nsew", **P)

        specs = [
            ("Spread:", "v_spread", tk.DoubleVar, 4.0, 0.5, 20.0, 0.5, " u"),
            ("Min dist\n(↓ = more occlusion):", "v_mindist", tk.DoubleVar, 2.0, 0.0, 10.0, 0.25, " u"),
            ("Camera distance:", "v_camdist", tk.DoubleVar, 2.5, 0.5, 5.0, 0.1, " u"),
        ]
        for r, (lbl, attr, vtype, default, lo, hi, inc, unit) in enumerate(specs):
            ttk.Label(fl, text=lbl).grid(row=r, column=0, sticky="w", padx=8, pady=3)
            var = vtype(value=default)
            setattr(self, attr, var)
            _Spin(fl, var, lo, hi, inc=inc, width=6, unit=unit).grid(
                row=r, column=1, sticky="w", padx=4, pady=3)

        # ── Camera elevation range ────────────────────────────────────────────
        fc = ttk.LabelFrame(self, text="Camera  (azimuth randomised per scene)")
        fc.grid(row=2, column=0, columnspan=2, sticky="ew", **P)

        ttk.Label(fc, text="Elevation min:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.v_el_min = tk.DoubleVar(value=30.0)
        _Spin(fc, self.v_el_min, 5, 85, inc=5, width=5, unit="°").grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(fc, text="Elevation max:").grid(row=0, column=2, sticky="w", padx=8, pady=4)
        self.v_el_max = tk.DoubleVar(value=60.0)
        _Spin(fc, self.v_el_max, 5, 85, inc=5, width=5, unit="°").grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(fc, text="(30–60° keeps every object fully in frame)",
                  foreground="grey").grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 4))

        # ── Augmentation ──────────────────────────────────────────────────────
        fv = ttk.LabelFrame(self, text="Augmentation  (checked = randomise each scene)")
        fv.grid(row=3, column=0, columnspan=2, sticky="ew", **P)

        self.v_v_lighting  = tk.BooleanVar(value=True)
        self.v_v_bg        = tk.BooleanVar(value=True)
        self.v_v_floor     = tk.BooleanVar(value=True)
        self.v_v_scale     = tk.BooleanVar(value=True)
        self.v_v_rot3d     = tk.BooleanVar(value=True)
        self.v_v_color     = tk.BooleanVar(value=True)
        self.v_v_camdist   = tk.BooleanVar(value=True)
        self.v_v_floorpat  = tk.BooleanVar(value=True)

        checks = [
            ("Lighting (ambient/diffuse)", self.v_v_lighting),
            ("Background grey",            self.v_v_bg),
            ("Floor colour",               self.v_v_floor),
            ("Floor pattern\n(checker/stripes/noise)", self.v_v_floorpat),
            ("Scale jitter ±25 %",         self.v_v_scale),
            ("Full 3-D rotation",          self.v_v_rot3d),
            ("Colour jitter ±25 RGB",      self.v_v_color),
            ("Camera zoom jitter ±10 %",   self.v_v_camdist),
        ]
        for idx, (lbl, var) in enumerate(checks):
            ttk.Checkbutton(fv, text=lbl, variable=var).grid(
                row=idx // 4, column=idx % 4, sticky="w", padx=10, pady=3)

        # ── Progress + button ─────────────────────────────────────────────────
        fb = ttk.Frame(self)
        fb.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 2))

        self.progress = ttk.Progressbar(fb, orient="horizontal", length=380, mode="determinate")
        self.progress.pack(side="left")

        self.v_status = tk.StringVar(value="Ready")
        ttk.Label(fb, textvariable=self.v_status, width=20).pack(side="left", padx=8)

        self.btn = ttk.Button(self, text="  Generate  ", command=self._start)
        self.btn.grid(row=5, column=0, columnspan=2, pady=10)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _browse(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.v_dir.set(d)

    def _build_config(self) -> GenConfig:
        return GenConfig(
            output_dir=self.v_dir.get(),
            num_images=self.v_num.get(),
            resolution=(self.v_rw.get(), self.v_rh.get()),
            shape_counts={n: (self.v_min[n].get(), self.v_max[n].get()) for n in SHAPE_NAMES},
            shape_spread=self.v_spread.get(),
            min_distance=self.v_mindist.get(),
            camera_distance=self.v_camdist.get(),
            cam_el_min=self.v_el_min.get(),
            cam_el_max=self.v_el_max.get(),
            vary_lighting=self.v_v_lighting.get(),
            vary_background=self.v_v_bg.get(),
            vary_floor=self.v_v_floor.get(),
            vary_scale=self.v_v_scale.get(),
            vary_rotation_3d=self.v_v_rot3d.get(),
            vary_color=self.v_v_color.get(),
            vary_cam_distance=self.v_v_camdist.get(),
            vary_floor_pattern=self.v_v_floorpat.get(),
        )

    def _start(self) -> None:
        cfg = self._build_config()
        self.btn.config(state="disabled")
        self.progress.config(maximum=cfg.num_images, value=0)
        self.v_status.set("Starting…")

        def _prog(done, total):
            self._q.put(("progress", done, total))

        def _done():
            self._q.put(("done", cfg.output_dir, cfg.num_images))

        def _err(msg):
            self._q.put(("error", msg))

        threading.Thread(
            target=run_generation, args=(cfg, _prog, _done, _err), daemon=True
        ).start()

    def _poll(self) -> None:
        try:
            while True:
                ev = self._q.get_nowait()
                if ev[0] == "progress":
                    _, done, total = ev
                    self.progress["value"] = done
                    self.v_status.set(f"{done} / {total}")
                elif ev[0] == "done":
                    _, out_dir, n = ev
                    self.progress["value"] = self.progress["maximum"]
                    self.v_status.set("Done!")
                    self.btn.config(state="normal")
                    messagebox.showinfo(
                        "Done",
                        f"Generated {n} images\n→ saved in '{out_dir}'",
                    )
                elif ev[0] == "error":
                    self.v_status.set("Error!")
                    self.btn.config(state="normal")
                    messagebox.showerror("Error", ev[1])
        except queue.Empty:
            pass
        self.after(100, self._poll)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    App().mainloop()
