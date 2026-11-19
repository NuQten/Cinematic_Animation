import math
from abc import ABC
from math import atan2, cos, isclose, pi, sin
from typing import Literal, overload

import numpy as np

from .errors import JonctionError
from .geometry import Droite2D, Point2D, Segment2D, Structure2D


class Liaison(ABC): 
    def __init__(self, segment1: Segment2D, segment2: Segment2D, ddl: np.ndarray):

        if segment1 == segment2 :
            raise ValueError("Les deux segments doivent être différent")
        
        self.segment1 = segment1
        self.segment2 = segment2

        self.ddl = ddl
        # 0: Pas de liberté / 1: Liberté
        # (Tx, Rx)
        # (Ty, Ry)
        # (Tz, Rz)

    @property
    def nb_ddl(self) -> int: 
        return np.sum(self.ddl)

    @property
    def centre(self) -> Point2D | None:
        """Point de contact entre les deux segments"""
        if self.segment1.point1 == self.segment2.point1 or self.segment1.point1 == self.segment2.point2 :
            return self.segment1.point1
        elif self.segment1.point2 == self.segment2.point1 or self.segment1.point2 == self.segment2.point2: 
            return self.segment1.point2
        else :
            return None

    @property
    def angle(self) -> float:
        """Donne l'angle entre les deux segments"""
        dot = self.segment1.vector_director @ self.segment2.vector_director
        det = self.segment1.vector_director[0] * self.segment2.vector_director[1] - self.segment1.vector_director[1] * self.segment2.vector_director[0]

        return atan2(det, dot)


class LiaisonPivot2D(Liaison):
    def __init__(self, segment1: Segment2D, segment2: Segment2D) -> None:
        """Liaison pivot 2D reliant deux segment

        Args:
            segment1 (Segment2D): 1er Segment relié à la liaison pivot
            segment2 (Segment2D): 2eme Segment relié à la liaison pivot
        """

        # En 2D : Bloque Tx, Ty; Autorise : Rz
        ddl = np.array([[0, 0],  # Tx, Rx
                        [0, 0],  # Ty, Ry
                        [0, 1]]) # Tz, Rz 

        super().__init__(segment1, segment2, ddl)

        if self.centre is None : 
            raise JonctionError("Les deux segment n'ont pas de point en commun")


class LiaisonGlissiere2D(Liaison):
    def __init__(
            self, 
            segment1: Segment2D, 
            segment2: Segment2D, 
            segment_directeur: Segment2D,
            point_coulissant: Point2D
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
        ddl = np.array([[1, 0],  # Tx, Rx
                        [0, 0],  # Ty, Ry
                        [0, 0]]) # Tz, Rz
                               
        super().__init__(segment1, segment2, ddl)

        # Vérification si le segment directeur appartient à la liaison
        if segment_directeur != segment1 and segment_directeur != segment2:
            raise ValueError("Le segment directeur doit être un segment appartenant à la liason")
        self._segment_directeur = segment_directeur

        if segment_directeur == segment1 : 
            self._segment_guide = segment2 
        else :
            self._segment_guide = segment1

        # Vérification si le point coulissant appartient au segment guide
        if self._segment_guide.point1 != point_coulissant and self._segment_guide.point2 != point_coulissant:
            raise ValueError("Le point coulisant doit appartenir au segment guide")
        self._point_coulissant = point_coulissant

       # Vérifie que le point coulissant est bien situé sur la droite directrise.
        if not self._segment_directeur.droite2D.is_include(self._point_coulissant):
            raise JonctionError("Le point coulisant doit être situé sur la droite défini par le segment directeur")


    @property
    def vector_director(self) -> np.ndarray:
        """Vecteur unitaire directeur u de la translation"""
        return self._segment_directeur.vector_director

    @property
    def vector_normal(self) -> np.ndarray:
        """Vecteur normal v de la translation (+90° à la translation)"""
        return self._segment_directeur.vector_normal

    @property
    def angle_from_ux(self) -> float:
        """Angle d'inclinaison de la glissière par rapport au vecteur directeur x"""
        u = self.vector_director
        return atan2(u[1], u[0])

    @property
    def centre(self) -> Point2D:
        """Point central représentant la liaison glissière"""
        return self._point_coulissant


class LiaisonPivotGlissant2D(LiaisonGlissiere2D):
    def __init__(
            self, 
            segment1: Segment2D, 
            segment2: Segment2D, 
            segment_directeur: Segment2D,
            point_coulissant: Point2D
    ) -> None:

        super().__init__(segment1, segment2, segment_directeur, point_coulissant)

        self.ddl = np.array([[1, 0],  # Tx, Rx
                             [0, 0],  # Ty, Ry
                             [0, 1]])  # Tz, Rz


class LiaisonEncastrement2D(Liaison):

    def __init__(self, segment1: Segment2D, segment2: Segment2D) -> None:
        """Liaison encastrement 2D reliant deux segments

        Args:
            segment1 (Segment2D): 1er segment relié à la liaison encastrement
            segment2 (Segment2D): 2eme segment relié à la liaison encastrement
        """
    
        ddl = np.array([[0, 0],  # Tx, Rx
                        [0, 0],  # Ty, Ry
                        [0, 0]])  # Tz, Rz
        
        super().__init__(segment1, segment2, ddl)

        if segment1.is_include(segment2.point1) : 
            self._point_jonction = segment2.point1
        elif segment1.is_include(segment2.point2) :
            self._point_jonction = segment2.point2
        elif segment2.is_include(segment1.point1) : 
            self._point_jonction = segment1.point1
        elif segment2.is_include(segment1.point2) : 
            self._point_jonction = segment1.point2
        else : 
            raise JonctionError("Un point des segments doit être situé l'autre segment")

    @property
    def centre(self) -> Point2D: 
        return self._point_jonction


class LiaisonBati2D(LiaisonEncastrement2D):

    @overload
    def __init__(self, segment: Segment2D, point: Point2D, axe: Literal['x', 'y'] = 'x') -> None: ...

    @overload
    def __init__(self, segment: Segment2D, point: Point2D, angle: float = 0.0) -> None: ...
    
    def __init__(self, segment: Segment2D, point: Point2D, orientation: Literal['x', 'y'] | float = 'x'):

        # LiaisonBati2D(segment, point, axe)
        axe = orientation
        if axe in {'x', 'X'}:
            u = np.array([1.0, 0.0])
        elif axe in {'y', 'Y'}:
            u = np.array([0.0, 1.0])
        # LiaisonBati2D(segment, point, angle)
        elif isinstance(orientation, (float, int)):
            angle = orientation
            u = np.array([cos(angle), sin(angle)])
        else : 
            raise ValueError("L'axe doit être \'x\', \'y\' ou un angle en radians")

        if not segment.is_include(point):
            raise JonctionError("Le point d'ancrage du bâti doit appartenir au segment")

        point1 = point.array - 0.5*u
        point2 = point.array + 0.5*u

        bati = Segment2D(Point2D(point1), Point2D(point2))
        super().__init__(segment, bati)

        self._point_jonction = point


# TODO : Créer liaison ponctuelle si utile (à voir)
class LiaisonPonctuelle2D(Liaison):
    pass