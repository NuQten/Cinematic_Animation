import numpy as np
import matplotlib.pyplot as plt
from math import pi

from cinematics.geometry import Point2D, Segment2D, Structure2D
from cinematics.joints import LiaisonBati2D, LiaisonGlissiere2D, LiaisonPivot2D, LiaisonPivotGlissant2D
from cinematics.mechanism import Mechanism
from cinematics.motors import MotorRotatif2D

# 1. Définition des points (configuration initiale géométriquement cohérente)
O = Point2D(0.0, 0.0, name="O")   # Pivot fixe de la manivelle
A = Point2D(0.0, 2.0, name="A")   # Bout de la manivelle / pied de bielle
B = Point2D(4.0, 0.0, name="B")   # Centre du coulisseau (glissière)

# Points pour définir le rail horizontal (bâti)
R = Point2D(8.0, 0.0, name="R2")

# 2. Création des solides (Structures)
# Note : pour que O appartienne à bati_rail, découpons le rail en [R1, O] et [O, R2] :
bati_rail = Structure2D(Segment2D(O, R, color="black"), name="Bati", color="black")
manivelle = Structure2D(Segment2D(O, A, color="royalblue"), name="Manivelle", color="royalblue")
bielle = Structure2D(Segment2D(A, B, color="crimson"), name="Bielle", color="crimson")


# 3. Définition des liaisons
liaisons = [
    LiaisonBati2D(bati_rail, R, orientation=pi/4),               # Ancrage du rail au sol
    LiaisonPivot2D(bati_rail, manivelle, point1=O),              # Pivot en O (nécessite O dans bati_rail, voir note ci-dessous)
    LiaisonPivot2D(manivelle, bielle, point1=A),                 # Pivot en A
    LiaisonPivotGlissant2D(
        bati_rail, 
        bielle, 
        bati_rail.segments[0], 
        point_coulissant=B
    )
]

# 4. Définition des moteurs
motors = [
    MotorRotatif2D(manivelle.segments[0], pi/2)
] 

# 5. Création et affichage du mécanisme
meca = Mechanism(
    structures=[bati_rail, manivelle, bielle],
    liaisons=liaisons,
    motors=motors
)


fig, ax = plt.subplots(figsize=(7, 7))

anim = meca.animate(ax)

plt.show()