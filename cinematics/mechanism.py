from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from .geometry import Point2D, Structure2D, Segment2D, to_structure2d
from .joints import Liaison, LiaisonBati2D
from .motors import Motor
from .constants import Colors


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

    def solve(self) -> bool:
        X = self.X if self.X is not None else self.X0
        sol = optimize.root(self.systeme_constraintes, X, method="hybr")

        if sol.success:
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


if __name__ == "__main__":
    pass
