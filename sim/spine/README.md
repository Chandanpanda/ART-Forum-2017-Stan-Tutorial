# `sim/spine` — is the truss worth building?

`sim/truss` answers whether the cell can build a spine. This package
answers whether the spine is worth building, by the only measure the
product has: **how much range accuracy the stereo pair loses when the
structure moves.** It models a spine as a 3-D frame, solves it under
acceleration and temperature, finds its modes, and converts every
compliance into metres of range error at a stated range, so a lattice and
a carbon tube can be compared in one currency inside one solver.

```
python3 sim/scripts/spine/check_all.py        # 47 checks, under two seconds
python3 sim/scripts/spine/sweep_spine.py      # the design sweep
python3 sim/scripts/spine/sweep_spine.py --wide --f1-min 200
```

Nothing here imports the automation package, and nothing here knows about
rings, fixtures or winding. Units are SI throughout; millimetres stop at
the builders.

## The conversion everything rests on

A stereo pair reads range from disparity, `Z = f B / d`. Rotate one camera
by `dθ` about the vertical and every disparity shifts by `f dθ`, so

```
dZ = Z² dθ / B          the focal length cancels
dZ = Z  dB / B          a stretched baseline is a pure scale error
```

Both are metres at a stated range. So a stiffness, a temperature and a
resonance are added up in one number, and that number is what a customer
buys. `check_spine` verifies the formula against the disparity arithmetic
it comes from, at two focal lengths.

**Only the relative pose matters.** Rigid motion of the whole rig changes
nothing. That single observation produces most of the findings below.

## What the model found

**A centre-mounted spine is twice as sensitive as its tip slope suggests.**
The two cantilevers point opposite ways, so their camera faces rotate in
opposite senses and the errors add. The brief compares one tip's slope
against a budget written between the faces. Measured: the faces rotate
±4.83e-5 rad, so the relative error is 9.66e-5, and the 1 m truss as first
specified uses 1.42 of its yaw budget rather than the 0.5 the hand
calculation suggests.

**The camera bracket is a structural member.** A three-chord Warren
lattice with no transverse members is thirty mechanisms short of rigidity
by Maxwell's count — 60 members and 9 supports against 99 freedoms — and
stands up only by bending 2 mm rods. Left open, the end triangle breathes,
the tip mass rides the mechanism, and the first mode is 35 Hz instead of
195. Closing the two end triangles fixes it, and **the strap that closes
them need not be stiff, only present**: a 6 × 1 mm bar gives 194 Hz where
a 20 × 3 mm plate gives 195. This is a design requirement the brief does
not state, and a flimsy bracket bolted to one chord would forfeit six
times the stiffness.

**Once the ends are closed, the single-diagonal web is as good as an X.**
203 Hz at 74.9 g for the X-braced version against 195 Hz at 54.4 g for the
specified one. The 20 g and the thirty extra joints buy nothing.

**Aircraft rotation costs nothing at all.** Angular acceleration about any
axis moves both faces together, so the relative pose does not change. The
open lattice's torsional softness, which is where a closed tube should
have won, never gets to matter. Only *linear* acceleration disturbs the
pair, and fore-aft is the direction that costs depth: vertical costs
relative roll instead, which breaks rectification rather than biasing
range.

**The truss beats the tube by six times at a quarter of the mass**, and
the tube's problem is thermal, not stiffness. A monolithic tube bows under
a one-sided gradient at `α × grad`; the lattice's open section and its
smaller expansion coefficient give it 1572 µdeg against the tube's 26,649.
Aluminium is hopeless for the same reason.

**The design is section-limited.** Range error falls monotonically with
section depth, so what stops the sweep is packaging, not physics. Inside
the brief's own 100 mm envelope, no design in the sweep holds the 0.005°
budget at 3 g, in any stock, in either web; the best is 1.19. At 115 mm it
holds at 0.94. **That is the trade to put to whoever owns the airframe.**

**...and, once the cell's bore is a constraint, web-limited.** The bore a
joint needs scales as `tan α`, so the winding head limits the **web angle**
— and the web angle is what buys accuracy. Adding that constraint
(`sweep_spine.bore_margin`, supplied to `sweep.run`; the package stays
independent of any one cell) changes the answer, and the check that the
sweep carries it is in `check_spine`:

| relax nothing | 0 designs |
|---|---|
| relax the **bore** — open the ring 1.5 mm | **28 designs** |
| relax f1 — soft-mount harder | 124 designs |
| relax the budget | 0 designs |
| relax mass | 0 designs |

Swept over sides 80–180 mm and webs 35–55°, **not one design meets mass,
f1, the budget and the bore together.** Every design that holds the budget
wants a 45° web or steeper; none of those fits a 20 mm bore. Without the
constraint the sweep named `120/45/3.0/1.5` — 28 designs met the budget and
**none of them could be made**, which is a sweep answering a different
question.

So the bore is the cheapest constraint to give, and the only one that gives
a design meeting everything else: `warren 120/45/3.0/1.5`, 45.7 g, 0.88 of
the budget, 0.77 m at 100 m, f1 200 Hz — and 0.76 mm short of bore.
`Ring.ID` 20.0 → 21.5 buys it. What that costs is a re-check of the head:
`EXIT_R` sits at 11.0 and would have 0.25 mm over a 10.75 mm bore, which is
too little, so the exit guide moves too. **That is a decision, not a
derivation** — it trades the machine's head against the product's accuracy
— and `chosen.py` records it rather than making it.

**Measure E first.** A 20% error in the axial modulus moves the yaw budget
by 19%; the shear modulus, which is the least certain number in
`material.py`, moves it by nothing at all, because no load case that
matters puts the spine in torsion.

## The chosen design

`chosen.py` records the sweep's answer with its provenance: a 115 mm
triangle, 40° diagonals, 3 mm chords, 1.5 mm web. Against the geometry the
cell was first built around it is 16% lighter, 34% truer, inside the yaw
budget rather than 42% over it, and 24 joints instead of 27, so the cell
builds it in 21.6 minutes instead of 24.1. It costs a 115 mm section
instead of 92. `sim/scripts/truss/demo_optimal.py` builds it.

Two caveats the re-run added, both recorded in `chosen.py`: it now misses
the brief's 200 Hz first mode by **1 Hz** — 0.5 %, against a modulus that
is a typical value and moves f1 by 10 % per 20 % of itself, so measuring E
is what settles it; and it fits the head as built with **+0.45 mm** of bore
to spare, which is why it is the choice and a 45° web is not.

## What this model does not do

Crash, creep, moisture and fatigue are deliberately absent. They are real
and they are application-specific, and folding them in would make every
answer a drone answer when the same spine on a survey mast has no crash
case at all. `duty.py` is where an application states what it needs.

Resonant amplitude needs a damping figure, and joint damping — the thing
the brief claims as an advantage of sixty bonded joints — has to come from
a bench ring-down. The joints themselves are modelled as continuous
material; a wound and bonded lashing is somewhere between a pin and a
moment connection, and `pinned_diagonals` brackets that.

## The files

```
material.py   what the rod is made of, and which numbers are guesses
duty.py       what the spine must hold calibration through
frame.py      one 3-D frame element: stiffness, consistent mass, thermal
model.py      assembly, supports, static and modal solves, camera faces
build.py      the candidates: a three-chord lattice in three webs, and a tube
metrics.py    relative pose, range error, modal participation, transmissibility
sweep.py      the grid, the constraints, the Pareto front
chosen.py     the answer, with what it was chosen for and what it cost
```
