from __future__ import annotations

import math
from math import atan2, cos, pi, sin
from typing import overload

import numpy as np

from .constants import TOLERANCE


class Point2D:

    @overload
    def __init__(self, x:float = 0.0, y:float = 0.0, name:str| None = None) -> None: ...

    @overload
    def __init__(self, array: np.ndarray | tuple | list = (0.0, 0.0), name:str| None = None) -> None: ...

    def __init__(self, *args, name:str| None = None) -> None:

        self.name = name

        # Default : Point2D() -> (0, 0)
        if len(args) == 0: 
            self.x, self.y = 0.0, 0.0

        # 1 Arg : Point2D(array)
        elif len(args) == 1:
            self.x = float(args[0][0])
            self.y = float(args[0][1])

        # 2 Args :
        elif len(args) == 2: 

            # Point2D(x, y)
            if isinstance(args[0], (float, int, np.number)):
                self.x = args[0]
                self.y = args[1]

            # Point2D(array, name)
            else : 
                self.x = args[0][0]
                self.y = args[0][1]
                self.name = args[1]

        # 3 Args : Point2D(x, y, name)
        elif len(args) == 3:
            self.x = args[0]
            self.y = args[1]
            self.name = args[2]

        else : 
            raise TypeError('Arguments invalide pour initialiser un Point2D')
    
    @property
    def distance_from_origin(self) -> float: 
        return (self.x**2 + self.y**2)**0.5

    @property
    def array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def __str__(self) -> str:
        return f"Point : {self.name}({self.x}, {self.y})"

    def __eq__(self, other:Point2D) -> bool:
        if not isinstance(other, Point2D):
            return False
        
        x = math.isclose(self.x, other.x, abs_tol = TOLERANCE)
        y = math.isclose(self.y, other.y, abs_tol = TOLERANCE)
        return x and y

    def distance(self, other:Point2D) -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

class Droite2D:

    def __init__(self, point1: Point2D, point2: Point2D):

            n1 = point1.name or "?"
            n2 = point2.name or "?"
            self.name = f"({n1} {n2})"
    
            self.point1 = point1 
            self.point2 = point2

    @property
    def vector_director(self) -> np.ndarray:
        """Vecteur directeur unitaire u du segment allant du Point1 au Point2"""

        dx = self.point2.x - self.point1.x
        dy = self.point2.y - self.point1.y
        L = self.point1.distance(self.point2)

        if math.isclose(L, 0, abs_tol = TOLERANCE):
            return np.array([1.0, 0.0])
        
        return np.array([dx / L, dy / L])

    @property
    def vector_normal(self) -> np.ndarray:
        """Vecteur normal unitaire v du segment (+90° par rapport à u)"""
        u = self.vector_director
        return np.array([-u[1], u[0]])

    def __str__(self) -> str:
        return f"Droite : {self.name} entre {self.point1} et {self.point2}"
 
    ### Methode ###

    def point_croisement(self, other: Droite2D) -> Point2D | None: 
        """Donne le point de croisement I de deux droites"""
        u = self.vector_director
        v = other.vector_director
        det_uv = u[0] * v[1] - u[1] * v[0]
        if abs(det_uv) < TOLERANCE :
            return None
        
        AC = other.point1.array - self.point1.array
        det_AC_v = AC[0] * v[1] - AC[1] * v[0]
        t = det_AC_v / det_uv
        I = self.point1.array + t*u
        return Point2D(I)

    def projete_orthogonal(self, P: Point2D) -> Point2D:
        """Donne le projeté orthogonal H de P sur la droite AB"""
        A = self.point1
        AP = P.array - self.point1.array
        u = self.vector_director
        dot_AP_u = np.dot(AP, u)
        H = A.array + dot_AP_u * u
        return Point2D(H)

    def distance_orthogonal(self, P: Point2D) -> float:
        """Donne la distance orthogonal entre le point P et la droite AB"""
        H = Droite2D.projete_orthogonal(self, P)
        return P.distance(H)

    def is_include(self, P: Point2D) -> bool:
        """Dit si le point P appartient à la droite"""
        H = Droite2D.projete_orthogonal(self, P)
        return P.distance(H) < TOLERANCE

class Segment2D(Droite2D):
    def __init__(self, point1: Point2D, point2: Point2D):

        if point1 == point2 : 
            raise ValueError("Les deux points doivent être différent")

        super().__init__(point1, point2)

        n1 = point1.name or "?"
        n2 = point2.name or "?"
        self.name = f"[{n1} {n2}]"

    @property
    def longeur(self):
        """Longeur du segment"""
        dx = self.point1.x - self.point2.x
        dy = self.point1.y - self.point2.y
        return (dx**2 + dy**2)**0.5

    @property
    def center(self) -> tuple[float, float]:
        """Centre du segment"""
        return ((self.point1.x + self.point2.x)/2, # X
                (self.point1.y + self.point2.y)/2) # Y

    @property
    def droite2D(self) -> Droite2D:
        return Droite2D(self.point1, self.point2)

    def __str__(self) -> str:
        return f"Segment : {self.name} entre {self.point1} et {self.point2}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment2D):
            return False

        p1 = self.point1 == other.point1 and self.point2 == other.point2 
        p2 = self.point2 == other.point1 and self.point1 == other.point2 
        return p1 or p2

    def is_include(self, P):
        """Dit si le point P appartient au segment"""
        if super().is_include(P):
            A = self.point1
            B = self.point2
            AB = B.array - A.array
            AP = P.array - A.array
            dot_AP_AB = np.dot(AP, AB)
            dot_AB2 = np.dot(AB, AB)
            return dot_AP_AB >= 0 and dot_AP_AB <= dot_AB2 
        else :
            return False

    def point_croisement(self, other: Droite2D) -> Point2D | None: 
        """Donne le point de croisement I de deux segments"""
        I = super().point_croisement(other)
        if I is None: 
            return None

        if self.is_include(I) and other.is_include(I):
            return I
        else :
            return None 

    def projete_orthogonal(self, P: Point2D) -> Point2D | None:
        """Donne le projeté orthogonal H de P sur le segment AB"""
        H = super().projete_orthogonal(P)
        if self.is_include(H):
            return H
        else :
            return None

    def distance_orthogonal(self, P: Point2D) -> float | None:
        """Donne la distance orthogonal entre le point P et le segment AB"""
        H = super().projete_orthogonal(P)
    
        if self.is_include(H):
            return P.distance(H)
        else :
            return None

    def distance_point(self, P: Point2D) -> float:
        H = super().projete_orthogonal(P)
        if self.is_include(H) : 
            return P.distance(H)
        else : 
            return min(P.distance(self.point1), P.distance(self.point2))

class Structure2D: 

    def __init__(self):
        pass

