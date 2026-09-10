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
±1.14e-5 rad, so the relative error is 2.29e-5 — twice what a one-tip
calculation reports, whatever the head weighs.

**The camera bracket is a structural member.** A three-chord Warren
lattice with no transverse members is thirty mechanisms short of rigidity
by Maxwell's count — 60 members and 9 supports against 99 freedoms — and
stands up only by bending 2 mm rods. Left open, the end triangle breathes,
the tip mass rides the mechanism, and the first mode is 96 Hz instead of
290. Closing the two end triangles fixes it, and **the strap that closes
them need not be stiff, only present**: a 6 × 1 mm bar recovers 97% of what
a 20 × 3 mm plate does (283 Hz against 290, from 96 open). This is a design
requirement the brief does not state, and a flimsy bracket bolted to one
chord would forfeit three times the stiffness. With the mount modelled the
question goes away entirely — the nose's own end battens close the triangle,
and an "open" truss with a nose on it reads identically to a closed one.

**Once the ends are closed, the single-diagonal web beats an X.** 403 Hz at
74.9 g for the X-braced version against 290 Hz at 54.4 g for the specified
one — and the X is *worse* on the angular budget, 0.64 against 0.58. With a
3 g camera the truss carries mostly itself (54 g of structure against 8 g
of cameras), so 20 g of extra carbon is 20 g of extra manoeuvre load. It
buys frequency nobody asked for and pays for it in accuracy.

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

## What the head mass was worth

Every finding below used to read differently, and one unmeasured number is
why: the camera head was carried at **50 g**, a placeholder inherited from
the brief. A Camera Module 2 weighs **3 g** on the vendor's own comparison
table. More than ten times too much tip mass had been setting the tip force,
the Rayleigh mass, the first mode and therefore the whole answer.

(The part changed too, and for a separate reason: Module 3's motorised lens
moves ±50 µm with orientation, which is 1.06% of its image distance and so
1.06% of every range it reports — larger, from inside one camera, than the
whole inter-camera budget the truss holds. Module 2 has no lens actuator.
`sim/truss/README.md` has the trade.)

With the real head, on a sweep of 1470 candidates over sections 60–160 mm
and webs 30–60° with the mount modelled and every rule applied:

| what it was said to be | what it is |
|---|---|
| the budget cannot be met inside a 100 mm section | met at **85 mm** |
| the ring's **bore** is the binding constraint on accuracy | the chosen design clears it by **+0.37 mm** |
| the design is **section-limited** | it is **member-mode**-limited |
| 0 designs meet mass, f1, budget and bore together | **7** break no rule at all |

**What binds now is the diagonals' own first mode.** 1062 of the 1470
candidates hold the accuracy budget. What refuses them most often is a
1.0 mm web whose members ring inside the propellers' 200–330 Hz band —
831 of the 1062. Give up that one rule and the sweep reaches
`100/35/3.0/1.0`: 46.5 g and **0.358 m**, six grams lighter and 36% truer
than the chosen design. `MEMBER_F_MIN` is a rule about fatigue at a joint,
not about the cameras, and **nobody has pulled a joint with a 1.0 mm
diagonal at blade-pass.** That is the next bench test, and it is worth more
than any remaining geometric choice.

Opening the ring is still worth something and no longer decisive: give up
the bore and `100/45/3.0/2.0` reaches 0.480 m — 15% truer for 13 g.

**Four rules that bind nothing.** Giving up the section envelope, the first
mode, the mass ceiling or the loader's Y reach, one at a time, leaves the
answer exactly where it is.

**Measure E first.** A 20% error in the axial modulus moves the yaw budget
by 19%; the shear modulus, which is the least certain number in
`material.py`, moves it by nothing at all, because no load case that
matters puts the spine in torsion.

## Every rule, and who owns it

The sweep applies four kinds of rule, and only the first is its own:

- **This model's**: the angular budget, the first mode, mass, the section
  envelope. A frame solver is the best tool for these.
- **The members'** and **the cell's**, handed over whole by
  `sweep_spine.cell_rules` from the package that owns them: a diagonal's
  own mode against the propeller band, the ring's fit round a joint, the
  qualified joint angle, chord buckling, the cycle time, and how far the
  gantry must reach to lay the racks. The spine package never learns them.
- **The winding head's bore**, as a margin in millimetres
  (`sweep_spine.bore_margin`), so a near miss can be read rather than only
  rejected.

Each has rejected a design the sweep ranked first. Without the bore it
named a 45° web the head cannot enter; with the bore but without the
members' modes and the loader's reach it named a 1.0 mm web on a truss the
gantry cannot lay the racks for.

**And the angle in the name is a request, not a fact.** `_stations` rounds
the bay pitch to a whole number of bays, so at a 100 mm section 35° and 40°
are the *same truss*, built at 38.66°. The bore rule scales as `tan α`;
asked about the requested angle it passed **40** candidates of the wide grid
whose real joints the head cannot enter. `build.effective_alpha` is what the
rules are asked about now, and `check_spine` asserts the difference.

## The chosen design

`chosen.py` records the sweep's answer with its provenance: an **85 mm**
triangle, 40° diagonals, 3 mm chords, 1.5 mm web — 52.6 g, f1 278 Hz, 0.67
of the yaw budget, **0.56 m of range error at 100 m**, +0.37 mm of bore,
30 joints and 21.9 minutes in the cell. It is the truest design that breaks
no rule; the lightest that breaks none is `75/35/3.0/1.5` at 51.2 g and
0.68 m, so the feasible set spans 1.4 g and 0.12 m end to end.
`sim/scripts/truss/demo_optimal.py` builds it.

`check_spine` reads that design out of `chosen.py` and holds it to every
rule, so a record that stops being feasible cannot go on being the answer —
which is how a 115 mm section stayed on record 0.16 mm short of its bore.

## The mount is in the model

The camera stands **21.4 mm** off the chord ends, solved so nothing of the
truss is inside a 66 × 41° field (`truss.mount.fov_standoff`), and the six
struts of each nose bond to the module's own metal enclosure so its mass
sits on the spine's axis and in the platform's plane. The standoff is not a
constant: it falls out of the section, 23 mm at 92 mm and 36 mm at 140, and
it is a lever arm — so a deeper section buys stiffness and pays part of it
back, and the sweep charges each candidate its own.

**The massless rigid placeholder it replaces is not a bound in either
direction.** It charges nothing for the mount's own mass and compliance
(optimistic), and it hangs the payload on the chord ends 66 mm off the axis
(pessimistic). Between a 50 g head and a 4 g one the net error changes
*sign*. No margin on a placeholder could have covered both, which is the
argument for solving the mount rather than allowing for it.

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
