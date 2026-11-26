from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import optimize
from collections.abc import Callable, Sequence


from .geometry import Point2D, Structure2D, Segment2D, to_structure2d
from .joints import Liaison, LiaisonBati2D
from .motors import Motor
from .constants import Colors, Statut, TOLERANCE


class Mechanism:
    def __init__(
        self,
        structures: list[Structure2D | Segment2D],
        liaisons: list[Liaison],
        motors: list[Motor] | None = None,
    ) -> None:

        self.structures = [to_structure2d(s) for s in structures]
        self.liaisons = liaisons
        self.motors = motors or []

        # Identifie tout les points du mecanisme
        all_points = set()
        for structure in self.structures:
            all_points.update(structure.points)

        # Identifie tout les points fixe
        self.points_fixes: set[Point2D] = set()
        for liaison in self.liaisons:
            if isinstance(liaison, LiaisonBati2D):
                self.points_fixes.update(liaison.bati.points)
                self.points_fixes.add(liaison.point1)

        # Identifie les points mobiles
        self.points_mobiles: list[Point2D] = [
            point for point in all_points if point not in self.points_fixes
        ]

        # Vecteur d'état inital X0 = [xA, yA, xB, yB, ...]
        self.X0 = np.array(
            [coord for point in self.points_mobiles for coord in (point.x, point.y)]
        )

        self.X = None

        self.colors = [structure.color for structure in self.structures]

    @property
    def state(self) -> Statut:

        Ec = 3*len(self.structures)
        Ic = 0
        for liaison in self.liaisons:
            Ic += 3 - liaison.nb_ddl
        m = len(self.motors)

        h = Ic + m - Ec
        if h == 0 : 
            return Statut.ISOSTATIC
        elif h > 0 : 
            return Statut.HYPERSTATIC
        else : 
            return Statut.HYPOSTATIC
        
    def _update_positions(self, X: np.ndarray) -> None:
        for i, point in enumerate(self.points_mobiles):
            point.x = float(X[2 * i])
            point.y = float(X[2 * i + 1])

    def systeme_constraintes(self, X: np.ndarray) -> list[float]:
        self._update_positions(X)
        residus: list[float] = []

        # 1. Contraintes de rigidité interne de chaque structures
        for st in self.structures:
            residus.extend(st.equations())

        # 2. Contraintes cinématique des liaisons
        for liaison in self.liaisons:
            residus.extend(liaison.equations())

        for motor in self.motors:
            residus.extend(motor.equations())

        return residus

    def solve(self, tolerance: float = TOLERANCE) -> bool:
        X = self.X if self.X is not None else self.X0
        sol = optimize.least_squares(self.systeme_constraintes, X, method="lm")

        if sol.success and sol.cost < tolerance:
            self.X = sol.x
            self._update_positions(self.X)
            return True

        return False

    def draw(self, ax: plt.Axes) -> plt.Axes:

        for i, structure in enumerate(self.structures):
            structure.draw(ax, self.colors[i])

        for liaison in self.liaisons:
            liaison.draw(ax)

        return ax

    def _step_animation(
            self,
            frame: int,
            ax: plt.Axes, 
            dt: float,
            xlim: tuple,
            ylim: tuple
    ) -> list:

        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")

        for motor in self.motors:
            motor.update(dt)

        self.solve()
        self.draw(ax)

        return []

    def animate(
        self,
        ax: plt.Axes,
        dt: float = 0.05,
        frames: int = 200,
        repeat: bool = True
    ) -> FuncAnimation:

        longueur_totale = sum(
            seg.longeur for st in self.structures for seg in st.segments
        )
        centre_x = np.mean([p.x for st in self.structures for p in st.points])
        centre_y = np.mean([p.y for st in self.structures for p in st.points])

        xlim = (centre_x - longueur_totale, centre_x + longueur_totale)
        ylim = (centre_y - longueur_totale, centre_y + longueur_totale)

        return FuncAnimation(
            ax.get_figure(),
            self._step_animation,
            frames= frames,
            fargs= (ax, dt, xlim, ylim),
            interval = int(dt * 1000),
            repeat= repeat
        )


if __name__ == "__main__":
    pass
