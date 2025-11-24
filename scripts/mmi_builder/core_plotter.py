import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import shapely.affinity as affinity
import tidy3d as td
import numpy as np


def plot_mmi_layout(structures, title="MMI Layout Preview"):
    """Plot x–y geometry (schematic view) of Tidy3D structures."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for s in structures:
        geom = s.geometry
        if isinstance(geom, td.Box):
            cx, cy, _ = geom.center
            sx, sy, _ = geom.size
            rect = Polygon([
                (cx - sx/2, cy - sy/2),
                (cx + sx/2, cy - sy/2),
                (cx + sx/2, cy + sy/2),
                (cx - sx/2, cy + sy/2),
            ])
            ax.plot(*rect.exterior.xy, 'k-', lw=1)
        
        elif hasattr(geom, 'points'):
            pts = np.array(geom.points)
            ax.plot(pts[:, 0], pts[:, 1], 'r-', lw=1)
    
    ax.set_aspect('equal')
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title)
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.show()

    
