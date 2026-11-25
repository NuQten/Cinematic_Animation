import math  
from abc import ABC, abstractmethod
from math import atan2, cos, isclose, pi, sin, degrees
from typing import Literal, overload

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from .errors import JonctionError
from .constants import Zorder, Colors, RAYON_LAISON
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
        self, 
        structure1: Structure2D | Segment2D, 
        structure2: Structure2D | Segment2D, 
        ddl: np.ndarray
    ) -> None:

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

    @property
    @abstractmethod
    def centre(self) -> Point2D | None:
        """Renvoi le centre de la liaison"""

    @abstractmethod
    def equations(self) -> list[float]:
        """Retourne les contraintes scalaires de la liaison (doivent valoir 0.0)"""

    @abstractmethod
    def draw(self, ax: plt.Axes) -> plt.Axes: ...


class LiaisonPivot2D(Liaison):

    def __init__(
        self,
        structure1: Structure2D | Segment2D,
        structure2: Structure2D | Segment2D,
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
    def centre(self) -> Point2D:
        return self.point1

    def equations(self) -> list[float]:
        return [self.point1.x - self.point2.x, self.point1.y - self.point2.y]

    def draw(self, ax: plt.Axes, rayon: float = RAYON_LAISON) -> plt.Axes:

        color1 = self.structure1.color
        color2 = self.structure2.color
        centre = self.centre 

        # 1. Cercle exterieur (structure 1)
        cercle_ext = patches.Circle(
                    (centre.array), 
                    radius=rayon, 
                    facecolor=color1, 
                    zorder=Zorder.LIAISON)
        ax.add_patch(cercle_ext)

        # 2. Blanc entre les deux segments
        cercle_blanc = patches.Circle(
            (centre.array), 
            radius=rayon * 0.8, 
            facecolor=Colors.WHITE, 
            zorder=Zorder.LIAISON)
        ax.add_patch(cercle_blanc)

        # 3. Cercle interieur (structure 2) + segment lié au pivot
        cercle_int = patches.Circle(
            (centre.array), 
            radius=rayon*0.6, 
            facecolor=color2, 
            zorder=Zorder.LIAISON)
        ax.add_patch(cercle_int)

        # 4. Segment allant jusqu'au point central
        vectors_pt = []
        for segment in self.structure2.segments:
            if centre in segment : 
                if centre == segment.point1:
                    vectors_pt.append((segment.vector_director, False))
                else : 
                    vectors_pt.append((segment.vector_director, True))

        for u, minus in vectors_pt :
            if minus : 
                x = [centre.x - u[0]*rayon, centre.x]
                y = [centre.y - u[1]*rayon, centre.y]
            else :  
                x = [centre.x + u[0]*rayon, centre.x]
                y = [centre.y + u[1]*rayon, centre.y]

            ax.plot(x, y, color= color2, linewidth = 2, zorder = Zorder.SEGMENT_LIAISON)
        
        return ax


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

        self.structure_rail = self.structure1
        self.structure_mobile = self.structure2

        if segment_directeur not in self.structure_rail:
            raise ValueError("Le segment directeur doit être à la structure de guidage")
        if point_coulissant not in self.structure_mobile:
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

    def __str__(self):
        return f"Liaison glissiere entre {self.structure1} et {self.structure2} avec le rail {self.segment_directeur} et le point coulissant {self.point_coulissant}"

    def equations(self) -> list[float]:

        # 1. Décollement normal : distance projetée sur la normal v
        pt_rail = self.segment_directeur.point1
        dx = self.point_coulissant.x - pt_rail.x
        dy = self.point_coulissant.y - pt_rail.y
        v = self.vector_normal
        decollement = dx * v[0] + dy * v[1]  # Si 0 pas de décollement

        # 2. Orientation relative bloquée
        u_rail = self.segment_directeur.vector_director
        u_guide = self._ref_segment_mobile.vector_director
        dot = u_rail @ u_guide
        det = det_2_vector2d(u_rail, u_guide)
        actual_theta = atan2(det, dot)
        d_theta = (actual_theta - self._initial_theta + pi) % (2 * pi) - pi

        return [decollement, d_theta]

    def draw(
            self, 
            ax: plt.Axes, 
            longeur: float = 4*RAYON_LAISON, 
            largeur: float = 2*RAYON_LAISON
    ) -> plt.Axes: 

        centre = self.centre
        angle_x_rad = self.angle_from_ux
        angle_x_deg = degrees(angle_x_rad)
        color2 = self.structure_mobile.color

        # 1. Rotation
        rotation = transforms.Affine2D().rotate_deg_around(centre.x, centre.y, angle_x_deg) + ax.transData

        # 2. Rectangle couleur
        bas_gauche = (centre.x - longeur/2, centre.y - largeur/2)
        rect = patches.Rectangle(
            bas_gauche, 
            width=longeur, 
            height=largeur, 
            facecolor=color2,
            zorder=Zorder.LIAISON,   
            transform=rotation)
        ax.add_patch(rect)

        # 3. Interieur rectangle blanc
        d_side = 0.2*min(longeur, largeur)
        longeur -= d_side
        largeur -= d_side
        bas_gauche = (centre.x - longeur/2, centre.y - largeur/2)
        rect = patches.Rectangle(
            bas_gauche, 
            width=longeur, 
            height=largeur, 
            facecolor= Colors.WHITE,
            zorder=Zorder.LIAISON, 
            transform=rotation)
        ax.add_patch(rect)

        return ax


class LiaisonPivotGlissant2D(LiaisonGlissiere2D):
    def __init__(
        self,
        structure_rail: Structure2D | Segment2D,
        structure_mobile: Structure2D | Segment2D,
        segment_directeur: Segment2D,
        point_coulissant: Point2D,
    ) -> None:

        super().__init__(structure_rail, structure_mobile, segment_directeur, point_coulissant)

        self.ddl = np.array(
            [
                [1, 0],  # Tx, Rx
                [0, 0],  # Ty, Ry
                [0, 1],  # Tz, Rz
            ]
        )  

    def equations(self) -> list[float]:

        pt_rail = self.segment_directeur.point1
        dx = self.point_coulissant.x - pt_rail.x
        dy = self.point_coulissant.y - pt_rail.y
        v = self.vector_normal

        return [dx * v[0] + dy * v[1]]

    def draw(
            self, 
            ax: plt.Axes, 
            longeur: float = 4*RAYON_LAISON, 
            largeur: float = 2*RAYON_LAISON,
            rayon: float = RAYON_LAISON
    ) -> plt.Axes: 

        centre = self.centre
        angle_x_rad = self.angle_from_ux
        angle_x_deg = degrees(angle_x_rad)
        color1 = self.structure_rail.color
        color2 = self.structure_mobile.color

        # 1. Rotation
        rotation = transforms.Affine2D().rotate_deg_around(centre.x, centre.y, angle_x_deg) + ax.transData

        # 2. Rectangle couleur
        bas_gauche = (centre.x - longeur/2, centre.y - largeur/2)
        rect = patches.Rectangle(
            bas_gauche, 
            width=longeur, 
            height=largeur, 
            facecolor=color2,
            zorder=Zorder.LIAISON,   
            transform=rotation)
        ax.add_patch(rect)

        # 3. Interieur rectangle blanc
        d_side = 0.2*min(longeur, largeur)
        longeur -= d_side
        largeur -= d_side
        bas_gauche = (centre.x - longeur/2, centre.y - largeur/2)
        rect = patches.Rectangle(
            bas_gauche, 
            width=longeur, 
            height=largeur, 
            facecolor= Colors.WHITE,
            zorder=Zorder.LIAISON, 
            transform=rotation)
        ax.add_patch(rect)

        # 4. Cercle exterieur (structure 1)
        cercle_ext = patches.Circle(
                    (centre.array), 
                    radius=rayon, 
                    facecolor=color2, 
                    zorder=Zorder.LIAISON)
        ax.add_patch(cercle_ext)

        # 5. Blanc entre les deux segments
        cercle_blanc = patches.Circle(
            (centre.array), 
            radius=rayon * 0.8, 
            facecolor= Colors.WHITE, 
            zorder=Zorder.LIAISON)
        ax.add_patch(cercle_blanc)

        # 6. Cercle interieur (structure 2) + segment lié au pivot
        cercle_int = patches.Circle(
            (centre.array), 
            radius=rayon*0.6, 
            facecolor=color1, 
            zorder=Zorder.LIAISON)
        ax.add_patch(cercle_int)

        # 7. Segment allant jusqu'au point central
        vectors_pt = []
        for segment in self.structure_mobile.segments:
            if self.point_coulissant in segment : 
                if self.point_coulissant == segment.point1:
                    vectors_pt.append((segment.vector_director, False))
                else : 
                    vectors_pt.append((segment.vector_director, True))
        
        distance = (largeur**2 + longeur**2)**0.5
        for u, minus in vectors_pt :
            # u est le vector_director du segment (normalisé, repère monde).
            # Le point sur le bord du cercle est simplement centre ± rayon*u.
            # Si minus=True  → le point coulissant est point2 du segment, donc
            #   le segment part vers -u depuis le centre : border est à -u.
            # Si minus=False → le segment part vers +u depuis le centre : border est à +u.
            if minus:
                border_x = centre.x - rayon * u[0]
                border_y = centre.y - rayon * u[1]
                x = [centre.x - u[0]*distance, border_x]
                y = [centre.y - u[1]*distance, border_y]
            else:
                border_x = centre.x + rayon * u[0]
                border_y = centre.y + rayon * u[1]
                x = [centre.x + u[0]*distance, border_x]
                y = [centre.y + u[1]*distance, border_y]
            ax.plot(x, y, color=color2, linewidth=2, zorder=Zorder.SEGMENT_LIAISON)
        
        return ax

# TODO : Ajuste fonction draw
class LiaisonEncastrement2D(Liaison):
    def __init__(
            self, 
            structure1: Structure2D | Segment2D, 
            structure2: Structure2D | Segment2D,
            point1:Point2D,
            point2: Point2D | None = None
    ) -> None:
        """Liaison encastrement 2D reliant deux segments

        Args:
            segment1 (Segment2D): 1er segment relié à la liaison encastrement
            segment2 (Segment2D): 2eme segment relié à la liaison encastrement
        """

        ddl = np.array(
            [
                [0, 0],  # Tx, Rx
                [0, 0],  # Ty, Ry
                [0, 0],  # Tz, Rz
            ]
        )  

        super().__init__(structure1, structure2, ddl)

        self.point1 = point1
        self.point2 = point2 if point2 is not None else point1

        if self.point1 not in self.structure1:
            raise JonctionError(f"Le point {self.point1} n'appartient pas à la structure 1")
        if self.point2 not in self.structure2:
            raise JonctionError(f"Le point {self.point2} n'appartient pas à la structure 2")
        if self.point1 != self.point2:
            raise JonctionError("Les points d'encastrement doivent être initialement superposés")

        self._ref_seg1 = self.structure1.segments[0]
        self._ref_seg2 = self.structure2.segments[0]

        # Define initial angle betwenn segment1 et segment2
        u1 = self._ref_seg1.vector_director
        u2 = self._ref_seg2.vector_director
        dot = u1 @ u2
        det = det_2_vector2d(u1, u2)
        self._initial_theta = atan2(det, dot)

    @property
    def centre(self) -> Point2D:
        return self.point1

    def equations(self) -> list[float]:

        u1 = self._ref_seg1.vector_director
        u2 = self._ref_seg2.vector_director
        dot = u1 @ u2
        det = det_2_vector2d(u1, u2)
        actuel_theta = atan2(det, dot)
        rotation = (actuel_theta - self._initial_theta + pi)%(2*pi) - pi

        return [
            self.point1.x - self.point2.x,
            self.point1.y - self.point2.y,
            rotation,
        ]

    def draw(self, ax: plt.Axes, largeur: float = 2*RAYON_LAISON) -> plt.Axes:

        centre = self.centre
        bas_gauche = (centre.x - largeur/2, centre.y - largeur/2)
        carre = patches.Rectangle(
            bas_gauche, 
            width= largeur, 
            height= largeur, 
            facecolor= Colors.BLACK,
            zorder=Zorder.LIAISON
        )

        ax.add_patch(carre)
        


# TODO : Faire fonction draw
class LiaisonBati2D(LiaisonEncastrement2D):
    @overload
    def __init__(
        self, structure: Structure2D |Segment2D, point: Point2D, axe: Literal["x", "y"] = "x"
    ) -> None: ...

    @overload
    def __init__(
        self, structure: Structure2D |Segment2D, point: Point2D, angle: float = 0.0
    ) -> None: ...

    def __init__(
        self,
        structure: Structure2D |Segment2D,
        point: Point2D,
        orientation: Literal["x", "y"] | float = "x",
    ):

        self.structure = to_structure2d(structure)

        if point not in self.structure : 
            raise JonctionError(f"Le point d'ancrage {point} doit appartenir à la structure à encastrer")
        

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


        sol_point1 = Point2D(point.array - RAYON_LAISON * u, name="Bati_1")
        sol_point2 = Point2D(point.array + RAYON_LAISON * u, name="Bati_2")
        sol_segment1 = Segment2D(sol_point1, point)
        sol_segment2 = Segment2D(sol_point2, point)
        structure_sol = Structure2D(sol_segment1, sol_segment2)

        super().__init__(structure, structure_sol, point, point)

        self.bati = self.structure2

    def draw(self, ax: plt.Axes, longeur: float = RAYON_LAISON, angle: float = pi/4) -> plt.Axes : 

        color = self.structure.color
        u = self.bati.segments[0].vector_director
        v = [u[0]*cos(angle) - u[1]*sin(angle),
                         u[0]*sin(angle) + u[1]*cos(angle)]
        p1 = self.bati.points[1]
        p2 = self.bati.points[2]

        # 1. Segment de base
        ax.plot(
            [p1.x, p2.x], 
            [p1.y, p2.y], 
            color= color, 
            linewidth=2, 
            zorder=Zorder.SEGMENT_LIAISON
            )

        # 2. 3 Tirets
        for pt in self.bati.points : 
            ax.plot(
                [pt.x, pt.x + longeur*v[0]],
                [pt.y, pt.y + longeur*v[1]],
                color= color,
                linewidth = 2, 
                zorder = Zorder.LIAISON
            )


# TODO : Créer liaison ponctuelle si utile (à voir)
class LiaisonPonctuelle2D(Liaison):
    pass


if __name__ == "__main__":
    pass
