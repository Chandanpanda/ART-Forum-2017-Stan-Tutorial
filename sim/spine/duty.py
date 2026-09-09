"""What the spine must hold calibration through, and what "held" means.

Separate from the structure on purpose: a survey mast and a quadrotor
share every equation in this package and agree on none of these numbers.
Changing the duty is how you ask whether a design that fails on a drone
is excellent on a tripod, which is the question that decides how wide the
product's market is.

CRASH IS NOT HERE.  Nor creep, moisture or fatigue.  They are real, they
are application-specific, and folding them into the general case would
make every answer a drone answer.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Duty:
    name:        str = "quadrotor"
    tip_mass_g:  float = 50.0      # one camera head
    manoeuvre_g: float = 3.0       # sustained, any direction
    g:           float = 9.81
    range_m:     float = 100.0     # the range the accuracy claim is made at
    yaw_budget_deg: float = 0.005  # relative, between the camera faces
    soak_K:      float = 30.0      # 20 to 50 C
    gradient_K:  float = 10.0      # sun on one side, across the section
    ang_accel:   float = 20.0      # rad/s^2 about the spine's own axis
                                   # [VERIFY from flight logs: an agile
                                   # quadrotor rolls harder than this]
    band_hz:     tuple = (200.0, 330.0)   # propeller blade pass
    damping:     float = 0.01      # [VERIFY by ring-down; joints dominate]
    base_g:      float = 0.5       # broadband base excitation in the band

    @property
    def tip_force(self):
        """N at one camera head under the manoeuvre load."""
        return self.tip_mass_g * 1e-3 * self.g * self.manoeuvre_g


QUADROTOR = Duty()
# A mast, a boat, a vehicle roof: no crash case, gentler manoeuvre, wider
# temperature swing, and vibration from an engine rather than a propeller.
SURVEY_MAST = Duty(name="survey mast", manoeuvre_g=0.5, soak_K=50.0,
                   gradient_K=20.0, ang_accel=1.0, band_hz=(20.0, 120.0), base_g=0.1)
CATALOGUE = {d.name: d for d in (QUADROTOR, SURVEY_MAST)}
