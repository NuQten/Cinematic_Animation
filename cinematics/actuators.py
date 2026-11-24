from abc import ABC, abstractmethod
from math import atan2, pi
from .geometry import Point2D, Segment2D, Structure2D, to_structure2d, det_2_vector2d

class Actuator(ABC):

    @abstractmethod
    def set_target(self, value: float) -> None: ...

    @abstractmethod
    def equations(self) -> list[float]: ...


class MotorRotatif2D(Actuator): 

    def __init__(self, segment: Segment2D, angle_initial: float = 0.0) -> None:

        self.segment = segment
        self.target_angle = angle_initial

    def set_target(self, angle: float) -> None:
        self.target_angle = angle

    def equations(self) -> list[float]:
        u = self.segment.vector_director
        actuel_angle = atan2(u[1], u[0])

        d_theta = (actuel_angle - self.target_angle + pi)%(2*pi) - pi

        return d_theta

class MotorLinear2D(Actuator):

    def __init__(
            self, 
            point1: Point2D, 
            point2: Point2D, 
            course_initial: float | None = None
    ) -> None:

        self.point1 = point1
        self.point2 = point2

        self.target_length = course_initial if course_initial is not None else point1.distance(point2)


    def set_target(self, longeur: float) -> None:
        self.target_length = longeur

    def equations(self) -> list[float]:

        dx = self.point2.x - self.point1.x
        dy = self.point2.y - self.point1.y
        return [(dx**2 + dy**2) - self.target_length**2]

