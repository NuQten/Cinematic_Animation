from abc import ABC, abstractmethod
from math import atan2, pi
from .geometry import Point2D, Segment2D, Structure2D, to_structure2d, det_2_vector2d

class Motor(ABC):

    def __init__(self, vitesse: float = 1.0) -> None:
        self.vitesse = vitesse

    @abstractmethod
    def set_target(self, value: float) -> None: ...

    @abstractmethod
    def update(self, dt: float) -> None: ...

    @abstractmethod
    def equations(self) -> list[float]: ...


class MotorRotatif2D(Motor): 

    def __init__(
            self, 
            segment: Segment2D, 
            angle_initial: float = 0.0,    # [rad]
            vitesse_angulaire: float = pi/4 # [rad/s]
    ) -> None:

        super().__init__(vitesse_angulaire)
        self.segment = segment
        self.target_angle = angle_initial

    def set_target(self, angle: float) -> None:
        self.target_angle = angle

    def update(self, dt: float) -> None:
        self.target_angle += self.vitesse*dt

    def equations(self) -> list[float]:
        u = self.segment.vector_director
        actuel_angle = atan2(u[1], u[0])

        d_theta = (actuel_angle - self.target_angle + pi)%(2*pi) - pi

        return [d_theta]


class MotorLinear2D(Motor):

    def __init__(
            self, 
            point1: Point2D, 
            point2: Point2D, 
            course_initial: float | None = None,
            vitesse_lineaire: float = 1.0,
            course_min: float | None = None,
            course_max: float | None = None,
    ) -> None:

        super().__init__(vitesse_lineaire)
        self.point1 = point1
        self.point2 = point2
        self.course_min = course_min
        self.course_max = course_max

        self.target_length = course_initial if course_initial is not None else point1.distance(point2)


    def set_target(self, longeur: float) -> None:
        self.target_length = longeur

    def update(self, dt: float) -> None:
        nouvelle_longueur = self.target_length + self.vitesse * dt
        # Inverse le sens automatiquement si des bornes min/max sont définies (aller-retour)
        if self.course_max is not None and nouvelle_longueur >= self.course_max:
            nouvelle_longueur = self.course_max
            self.vitesse = -abs(self.vitesse)
        elif self.course_min is not None and nouvelle_longueur <= self.course_min:
            nouvelle_longueur = self.course_min
            self.vitesse = abs(self.vitesse)

        self.target_length = nouvelle_longueur

    def equations(self) -> list[float]:

        dx = self.point2.x - self.point1.x
        dy = self.point2.y - self.point1.y
        return [(dx**2 + dy**2) - self.target_length**2]

