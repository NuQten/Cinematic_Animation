import math  
from abc import ABC, abstractmethod
from math import atan2, cos, isclose, pi, sin 
from typing import Literal, overload

import numpy as np

from .errors import JonctionError
from .geometry import (
    Droite2D, 
    Point2D,
    Segment2D,
    Structure2D,
    to_structure2d,
    det_2_vector2d,
)


class Liaison(ABC):
    def __init__(
        self, structure1: Structure2D, structure2: Structure2D, ddl: np.ndarray
    ):

        self.structure1 = to_structure2d(structure1)
        self.structure2 = to_structure2d(structure2)

        if self.structure1 == self.structure2:
            raise ValueError("Les deux structures doivent être différentes")

        self.ddl = ddl
        # 0: Pas de liberté / 1: Liberté
        # (Tx, Rx)
        # (Ty, Ry)
        # (Tz, Rz)

    @property
    def nb_ddl(self) -> int:
        return np.sum(self.ddl)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} entre {self.structure1.name} et {self.structure2.name} au point {self.centre.name}"

    @abstractmethod
    @property
    def centre(self) -> Point2D | None:
        """Renvoi le centre de la liaison"""

    @abstractmethod
    def equations(self) -> list[float]:
        """Retourne les contraintes scalaires de la liaison (doivent valoir 0.0)"""


class LiaisonPivot2D(Liaison):
    def __init__(
        self,
        structure1: Structure2D,
        structure2: Structure2D,
        point1: Point2D,
        point2: Point2D | None = None,
    ) -> None:

        # En 2D : Bloque Tx, Ty; Autorise : Rz
        ddl = np.array(
            [
                [0, 0],  # Tx, Rx
                [0, 0],  # Ty, Ry
                [0, 1],  # Tz, Rz
            ]
        )

        super().__init__(structure1, structure2, ddl)

        # Si un seul point passé alors on considère que le point1 est partagé entre les deux structures
        p1 = point1
        p2 = point2 if point2 is not None else point1

        if p1 not in self.structure1:
            raise JonctionError(f"Le point {p1} n'appartient pas à la strucutre 1")
        if p2 not in self.structure2:
            raise JonctionError(f"Le point {p2} n'appartient pas à la strucutre 2")
        if p1 != p2:
            raise JonctionError(
                "Les points doivent de la liaison pivot doivent être initialement superposés"
            )

        self.point1 = p1
        self.point2 = p2

    @property
    def center(self) -> Point2D:
        return self.point1

    def equations(self) -> list[float]:
        return [self.point1.x - self.point2.x, self.point1.y - self.point2.y]


class LiaisonGlissiere2D(Liaison):
    def __init__(
        self,
        structure_rail: Structure2D | Segment2D,
        structure_mobile: Structure2D | Segment2D,
        segment_directeur: Segment2D,
        point_coulissant: Point2D,
    ) -> None:
        """Liaison glissière 2D reliant deux segment

        Args:
            segment1 (Segment2D): 1er Segment relié à la laisson glissière
            segment2 (Segment2D): 2eme Segment relié à la laisson glissière
            axe_translation (tuple[float, float] | Segment2D | float, optional): Defaults = (1,0, 0.0)
                - tuple[float, float] : Vecteur directeur (Ux, Uy)
                - Segment2D : Vecteur lié au segment
                - float : phi [rad] [0; pi] -> (Ux, Uy) <=> (cos(phi), sin(phi))
        """

        # En 2d : Bloque : Rx, Ty; Autorise : Tx
        ddl = np.array(
            [
                [1, 0],  # Tx, Rx
                [0, 0],  # Ty, Ry
                [0, 0],  # Tz, Rz
            ]
        )

        super().__init__(structure_rail, structure_mobile, ddl)

        self.structure_rail = structure_rail
        self.structure_mobile = structure_mobile

        if segment_directeur not in self.structure_rail:
            raise ValueError("Le segment directeur doit être à la structure de guidage")
        if point_coulissant not in structure_mobile:
            raise ValueError(
                "Le point coulissant doit appartenir à la structure mobile"
            )
        if not segment_directeur.droite2D.is_include(point_coulissant):
            raise ValueError(
                "Le point coulisant doit être initialement aligné sur le rail"
            )

        self.segment_directeur = segment_directeur
        self.point_coulissant = point_coulissant

        # Initial value
        self._ref_segment_mobile = self.structure2.segments[0]
        u_rail = self.segment_directeur.vector_director
        u_guide = self._ref_segment_mobile.vector_director
        dot0 = u_rail @ u_guide
        det0 = det_2_vector2d(u_rail, u_guide)
        self._initial_theta = atan2(det0, dot0)

    @property
    def vector_director(self) -> np.ndarray:
        """Vecteur unitaire directeur u de la translation"""
        return self.segment_directeur.vector_director

    @property
    def vector_normal(self) -> np.ndarray:
        """Vecteur normal v de la translation (+90° à la translation)"""
        return self.segment_directeur.vector_normal

    @property
    def angle_from_ux(self) -> float:
        """Angle d'inclinaison de la glissière par rapport au vecteur directeur x"""
        u = self.vector_director
        return atan2(u[1], u[0])

    @property
    def centre(self) -> Point2D:
        """Point central représentant la liaison glissière"""
        return self.point_coulissant

    def equations(self) -> list[float]:

        # 1. Décollement normal : distance projetée sur la normal v
        pt_rail = self.segment_directeur.point1
        dx = self.point_coulissant.x - pt_rail.x
        dy = self.point_coulissant.y - pt_rail.y
        v = self.vector_normal
        decollement = dx * v[0] + dy * v[1]  # Si 0 pas de décollement

        # 2. Orientation relative bloquée
        u_rail = self.segment_directeur.vector_director
        u_guide = self._ref_segment_guide.vector_director
        dot = u_rail @ u_guide
        det = u_rail[0] * u_guide[1] - u_rail[1] * u_guide[0]
        actual_theta = atan2(det, dot)
        d_theta = (actual_theta - self._initial_theta + pi) % (2 * pi) - pi

        return [decollement, d_theta]


# TODO : faire avec la nouvelle methode
class LiaisonPivotGlissant2D(LiaisonGlissiere2D):
    def __init__(
        self,
        segment1: Segment2D,
        segment2: Segment2D,
        segment_directeur: Segment2D,
        point_coulissant: Point2D,
    ) -> None:

        super().__init__(segment1, segment2, segment_directeur, point_coulissant)

        self.ddl = np.array(
            [
                [1, 0],  # Tx, Rx
                [0, 0],  # Ty, Ry
                [0, 1],
            ]
        )  # Tz, Rz

    def equations(self) -> list[float]:

        pt_rail = self._segment_directeur.point1
        dx = self._point_coulissant.x - pt_rail.x
        dy = self._point_coulissant.y - pt_rail.y
        v = self.vector_normal

        return [dx * v[0] + dy * v[1]]


# TODO : faire avec la nouvelle methode
class LiaisonEncastrement2D(Liaison):
    def __init__(self, segment1: Segment2D, segment2: Segment2D) -> None:
        """Liaison encastrement 2D reliant deux segments

        Args:
            segment1 (Segment2D): 1er segment relié à la liaison encastrement
            segment2 (Segment2D): 2eme segment relié à la liaison encastrement
        """

        ddl = np.array(
            [
                [0, 0],  # Tx, Rx
                [0, 0],  # Ty, Ry
                [0, 0],
            ]
        )  # Tz, Rz

        super().__init__(segment1, segment2, ddl)

        if segment1.point1 == segment2.point1:
            self._pt1, self._pt2 = segment1.point1, segment2.point1
        elif segment1.point1 == segment2.point2:
            self._pt1, self._pt2 = segment1.point1, segment2.point2
        elif segment1.point2 == segment2.point1:
            self._pt1, self._pt2 = segment1.point2, segment2.point1
        elif segment1.point2 == segment2.point2:
            self._pt1, self._pt2 = segment1.point2, segment2.point2
        elif segment1.is_include(segment2.point1):
            self._pt1, self._pt2 = segment1.point1, segment2.point1
        elif segment1.is_include(segment2.point2):
            self._pt1, self._pt2 = segment1.point1, segment2.point2
        elif segment2.is_include(segment1.point1):
            self._pt1, self._pt2 = segment1.point1, segment2.point1
        elif segment2.is_include(segment1.point2):
            self._pt1, self._pt2 = segment1.point2, segment2.point1
        else:
            raise JonctionError(
                "Un point des segments doit être situé sur l'autre segment"
            )

        self._point_jonction = self._pt2

        # Define initial angle betwenn segment1 et segment2
        u1 = self.segment1.vector_director
        u2 = self.segment2.vector_director
        self._initial_cos = float(u1 @ u2)
        self._initial_sin = float(u1[0] * u2[1] - u1[1] * u2[0])

    @property
    def centre(self) -> Point2D:
        return self._point_jonction

    def equations(self) -> list[float]:

        u1 = self.segment1.vector_director
        u2 = self.segment2.vector_director
        actuel_sin = u1[0] * u2[1] - u1[1] * u2[0]

        return [
            self._pt1.x - self._pt2.x,
            self._pt1.y - self._pt2.y,
            actuel_sin - self._initial_sin,
        ]


# TODO : faire avec la nouvelle methode
class LiaisonBati2D(LiaisonEncastrement2D):
    @overload
    def __init__(
        self, segment: Segment2D, point: Point2D, axe: Literal["x", "y"] = "x"
    ) -> None: ...

    @overload
    def __init__(
        self, segment: Segment2D, point: Point2D, angle: float = 0.0
    ) -> None: ...

    def __init__(
        self,
        segment: Segment2D,
        point: Point2D,
        orientation: Literal["x", "y"] | float = "x",
    ):

        # LiaisonBati2D(segment, point, axe)
        axe = orientation
        if axe in {"x", "X"}:
            u = np.array([1.0, 0.0])
        elif axe in {"y", "Y"}:
            u = np.array([0.0, 1.0])
        # LiaisonBati2D(segment, point, angle)
        elif isinstance(orientation, (float, int)):
            angle = orientation
            u = np.array([cos(angle), sin(angle)])
        else:
            raise ValueError("L'axe doit être 'x', 'y' ou un angle en radians")

        if not segment.is_include(point):
            raise JonctionError("Le point d'ancrage du bâti doit appartenir au segment")

        point1 = point.array - 0.5 * u
        point2 = point.array + 0.5 * u

        bati = Segment2D(Point2D(point1), Point2D(point2))
        super().__init__(segment, bati)

        self._point_jonction = point


# TODO : Créer liaison ponctuelle si utile (à voir)
class LiaisonPonctuelle2D(Liaison):
    pass


if __name__ == "__main__":
    pass
