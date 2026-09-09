"""What the rod is made of, and which of these numbers is a guess.

PULTRUDED UNIDIRECTIONAL CARBON is not isotropic and must not be given an
isotropic modulus.  Along the fibre it is stiff; across it, and in shear,
it is the epoxy that answers.  The single number a datasheet quotes is the
axial one, and using E / 2(1+nu) for the shear modulus overstates the
torsional stiffness of a rod like this by roughly an order of magnitude.
That matters here because a three-chord truss with no vertical members
carries torque through its diagonals, and a tube carries it in shear flow:
the comparison this package exists to make turns on G.

EVERY VALUE BELOW IS A TYPICAL VALUE FOR THE CLASS, NOT A MEASUREMENT of
the stock in the shop.  `Material.measured` says which ones have been
checked against the real rod; none, at the time of writing.  The sweep is
therefore a first iteration: it ranks designs, and the ranking is only as
trustworthy as the ordering these numbers imply.  `check_material`
reports the sensitivity of the answer to each of them, which says which
one to measure first.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Material:
    """An orthotropic rod, reduced to what a frame element needs."""
    name:    str
    E:       float          # Pa, axial (fibre direction)
    G:       float          # Pa, shear -- resin dominated in a UD rod
    rho:     float          # kg/m^3
    alpha:   float          # 1/K, axial coefficient of thermal expansion
    measured: tuple = ()    # which fields came from a rig rather than a class

    def is_guess(self, field_name):
        return field_name not in self.measured


# Pultruded UD carbon rod: 230 GPa fibre at ~65% by volume gives ~150 GPa
# axial, which the brief's own arithmetic uses and sim/truss inherits.
# G12 for a UD carbon/epoxy laminate runs 4 to 6 GPa; 5 is the middle of
# that and is the value most likely to be wrong here.  The axial CTE of UD
# carbon is small and NEGATIVE, a few tenths of a ppm; exactly zero is the
# conventional simplification and it is the wrong one to make here,
# because it silently answers every thermal question with "nothing
# happens".  [VERIFY: E by 3-point bend, G by torsion, alpha by
# dilatometer -- and alpha is the one whose sign nobody guesses right]
CARBON_UD = Material(
    name="pultruded UD carbon rod",
    E=150e9, G=5.0e9, rho=1600.0, alpha=-0.3e-6,
)

# The camera bracket and the mount collar: stiff, and MASSLESS because
# their mass is already counted in the camera head.  Not a material, a
# modelling device -- it closes the end triangle the way a bolted plate
# does, so the face has a pose rather than a shape.
BRACKET = Material(name="bracket (massless, stiff)", E=69e9, G=26e9,
                   rho=0.0, alpha=0.0)

# The tube the brief rejects, for the comparison: roll-wrapped cloth at
# +-45 degrees is a very different material from the same fibre -- about
# 70 GPa axial, and much better in shear, which is the whole point of
# +-45 plies.
CARBON_WRAP = Material(
    name="roll-wrapped +-45 carbon tube",
    E=70e9, G=25e9, rho=1550.0, alpha=2.0e-6,
)

ALUMINIUM = Material(name="6061-T6", E=69e9, G=26e9, rho=2700.0, alpha=23.6e-6)

CATALOGUE = {m.name: m for m in (CARBON_UD, CARBON_WRAP, ALUMINIUM)}
