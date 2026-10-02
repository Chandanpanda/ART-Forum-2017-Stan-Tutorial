"""A month of the line in seconds: the fast line model (Tier 0).

Stations, the robot and the person are timed resources.  Trays move only
on the robot.  Orders arrive as a Poisson stream; the person visits on the
Line's schedule and does the plan's six manual tasks; faults and rejects
are drawn from spec.Faults, and one long robot outage can be injected.
Every time and every size is line.plan_line's -- this module chooses
neither, it only says whether they held:

    the takt     every order received in the month ships inside the
                 promise, and none is left behind
    the stores   no station waits for a part, a unit or a tray that the
                 line has run out of, and the person never finds a store
                 empty while a tray still wants it
    the racks    nothing that has to be set down finds its rack, shelf or
                 stretchers full

THE POLICY is the executive's, in one place (_next_*): downstream first,
so the line drains; a tray of frames is released only against a tray of
sleeves; station 1 collapses before it builds and builds only while the
frames on hand and in cure are short of the frame stock; parts trays are
swapped before they run out.  A bat that fails is marked and skipped and
goes to the reject shelf at station 5; one that fails before its sleeve
hands its sleeve to the next frame.

WHAT IS NOT HERE YET, so nobody mistakes it for a result: the robot's
charge (the plan's charging job), a second robot, faults a station cannot
clear itself, the person's own time limit, and anything a station does
inside a job -- those are its job time, from line.py.
"""
import heapq
from dataclasses import dataclass, field
from math import ceil

import numpy as np

from ..spec import Est, Faults, Person, Epoxy, Rod
from .. import line as L

DAY = 86400.0
TRAY_STATIONS = ("electronics", "sleeve", "calibration", "pack")


@dataclass
class Tray:
    id: int
    kind: str                # bat | mandrel | stretcher | tube | parts
    slots: list = None       # bat/tube: bat ids; mandrel: None, "cage" or (t_cured, ok);
                             # stretcher: order ids
    kits: int = 0            # parts trays
    station: str = ""        # parts trays: whose
    loc: str = ""
    bound: int = None        # a bat tray in flight: the stretcher tray its sleeves are on
    done: set = field(default_factory=set)
    read: set = field(default_factory=set)
    away: bool = False       # a stretcher tray out with a bat tray


@dataclass
class Bat:
    id: int
    ok: bool = True
    order: int = None
    done: set = field(default_factory=set)


@dataclass
class Order:
    id: int
    t_in: float
    sleeve: str = "waiting"  # waiting | ready | bound | fitted
    tray: int = None
    bat: int = None
    t_out: float = None
    remakes: int = 0         # sleeves lost with a bat that failed after its fitting


class Resource:
    def __init__(self, name):
        self.name = name
        self.busy = False
        self.down_until = 0.0
        self.token = 0
        self.t_end = 0.0
        self.busy_s = 0.0
        self.done = None


@dataclass
class Result:
    plan: object
    seed: int
    window_days: float
    orders: int
    shipped: int
    late: int                # shipped after the promise ...
    late_remade: int         # ... of which, inside it plus a visit interval per re-made sleeve
    unshipped: int
    lead_days: list
    stockouts: list          # (what, t): once per visit interval it lasts
    overflows: list          # (what, t): once per visit interval it lasts
    busy: dict               # resource -> fraction of line hours
    busy_per_bat: dict       # resource -> seconds busy per bat shipped, over the whole run
    person_min: dict         # task -> minutes over the run
    person_per_bat: float
    built: int
    frame_rejects: int
    rejects: int
    nozzles: int
    purged_ml: float
    resin_ml: float
    plates: int
    plates_failed: int
    peak: dict               # what -> the most it ever held
    low: dict                # what -> the least it ever held

    @property
    def takt_held(self):
        """Every order shipped inside the promise.  A sleeve is made only
        at a visit, so an order whose bat failed after its sleeve was fitted
        waits a visit interval for another: that lateness is the promise's,
        not the line's, and is reported (late_remade) rather than failed."""
        return self.late == self.late_remade and self.unshipped == 0

    def kinds(self, events):
        """{what: (times, first day)} for stockouts or overflows."""
        out = {}
        for what, t in events:
            n, first = out.get(what, (0, t / DAY))
            out[what] = (n + 1, first)
        return out

    def ok(self):
        return self.takt_held and not self.stockouts and not self.overflows


class LineSim:
    def __init__(self, plan, seed=0, faults=True, outage_day=None, days=None):
        self.P = P = plan
        self.ln = P.line
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.faults = faults
        self.k = self.ln.bats_per_tray
        self.window = self.ln.days_month
        self.horizon = (days if days is not None
                        else self.ln.days_month + self.ln.promise_days + 1.0) * DAY
        self.q, self.seq, self.t = [], 0, 0.0
        S = P.sizes
        self.res = {n: Resource(n) for n in L.STATIONS + ("robot",)}
        self.robot_at = "charger"
        self.cap = dict(S["racks"])
        self.racks = {name: [] for name in self.cap}
        self.docks = {"%s/%s" % (n, k): None for n, st in P.sts.items() for k in st.docks}
        self.incoming = {}
        self.trays, self.bats, self.orders = {}, {}, {}
        self.by_state = {"waiting": set(), "ready": set(), "bound": set(), "fitted": set()}
        self.stockouts, self.overflows = [], []
        self._dry, self._full = {}, {}
        self.person = {}
        self.peak, self.low = {}, {}
        self.built = self.frame_rejects = self.rejects = 0
        self.nozzles, self.purged, self.resin_used = 0, 0.0, 0.0
        self.plates = self.plates_failed = 0
        self.last_dose = -1e18
        self.golden_day = -1
        self.reject_shelf = self.reject_bin = 0
        self._build()

    # ============================================================ engine
    def at(self, t, fn, *args):
        heapq.heappush(self.q, (t, self.seq, fn, args))
        self.seq += 1

    def run(self):
        while self.q and self.q[0][0] <= self.horizon:
            t, _, fn, args = heapq.heappop(self.q)
            self.t = t
            fn(*args)
            self.dispatch()
        return self.result()

    def _noop(self):
        pass

    def open(self, t):
        h = (t % DAY) / 3600.0
        return self.ln.day_start_h <= h < self.ln.day_start_h + self.ln.hours_day

    def next_open(self, t):
        d = floor_day(t)
        start = d * DAY + self.ln.day_start_h * 3600.0
        if t < start:
            return start
        if self.open(t):
            return t
        return start + DAY

    # ========================================================= the start
    def _new_tray(self, kind, loc, **kw):
        tr = Tray(len(self.trays), kind, **kw)
        self.trays[tr.id] = tr
        self._place(tr, loc)
        return tr

    def _new_bat(self):
        b = Bat(len(self.bats))
        self.bats[b.id] = b
        return b.id

    def _build(self):
        """The line as it starts the month: primed to its sizes."""
        S, k = self.P.sizes, self.k
        # bat trays: the frame stock full of frames, one empty at station 1
        n_frames = S["frame_stock"]["frames"]
        for i in range(S["trays"]["bat"]):
            loc = "frame/frames" if i == 0 else "stock"
            slots = [None] * k
            if i > 0:
                for j in range(k):
                    if n_frames > 0:
                        slots[j] = self._new_bat()
                        n_frames -= 1
            self._new_tray("bat", loc, slots=slots)
        for i in range(S["trays"]["mandrel"]):
            loc = ("frame/mandrels A", "frame/mandrels B")[i] if i < 2 else "cure"
            self._new_tray("mandrel", loc, slots=[None] * k)
        for i in range(S["trays"]["stretcher"]):
            self._new_tray("stretcher", "sleeves", slots=[None] * k)
        for i in range(S["trays"]["tube"]):
            self._new_tray("tube", "pack/tubes" if i == 0 else "outbound", slots=[None] * k)
        # the bulk stores: bought at a restock's worth, printed at the farm's level
        self.bulk = dict(S["bought"])
        for part, v in S["printers"].items():
            self.bulk[part] = v["order_up_to"]
        for st, v in S["parts_trays"].items():
            dock = "%s/%s" % (st, "cores" if st == "frame" else "parts")
            for i in range(v["trays"]):
                tr = self._new_tray("parts", dock if i == 0 else "inbound", station=st, kits=0)
                self._fill_tray(tr, count=False)
        # consumables at the stations, full
        self.cons = {k: v["units"] * v["per"] for k, v in S["consumables"].items()}
        self.cons_cap = dict(self.cons)
        self.printers = []
        for part, v in S["printers"].items():
            self.printers += [{"part": part, "t_done": None, "good": False} for _ in range(v["printers"])]
        # orders: a Poisson stream over the whole run
        lam = self.ln.demand_month / (self.ln.days_month * DAY)
        t = 0.0
        while True:
            t += self.rng.exponential(1.0 / lam)
            if t >= self.horizon:
                break
            self.at(t, self._order)
        # the days and the visits
        for d in range(int(ceil(self.horizon / DAY)) + 1):
            self.at(d * DAY + self.ln.day_start_h * 3600.0, self._noop)
            if d % self.ln.visit_every_days == 0:
                self.at(d * DAY + self.ln.visit_at_h * 3600.0, self._visit, d)
        if self.faults:
            for r in self.res.values():
                self._next_fault(r, 0.0)

    # ======================================================== places
    def _place(self, tr, loc):
        tr.loc = loc
        if loc in self.racks:
            self.racks[loc].append(tr.id)
            self._mark("rack " + loc, len(self.racks[loc]))
        else:
            self.docks[loc] = tr.id

    def _take(self, tr):
        if tr.loc in self.racks:
            self.racks[tr.loc].remove(tr.id)
        else:
            self.docks[tr.loc] = None
        tr.loc = "robot"

    def _free(self, loc):
        if loc in self.racks:
            return len(self.racks[loc]) + self.incoming.get(loc, 0) < self.cap[loc]
        return self.docks[loc] is None and self.incoming.get(loc, 0) == 0

    def dock(self, loc):
        i = self.docks[loc]
        return None if i is None else self.trays[i]

    def rack(self, loc):
        return [self.trays[i] for i in self.racks[loc]]

    def _mark(self, what, v):
        self.peak[what] = max(self.peak.get(what, v), v)
        self.low[what] = min(self.low.get(what, v), v)

    # ============================================== stores and shortages
    def _short(self, what):
        """A consumer found `what` gone.  Logged once until it comes back, or
        until the next visit if it is still gone then."""
        if what not in self._dry:
            self._dry[what] = self.t
            self.stockouts.append((what, self.t))

    def _back(self, what):
        self._dry.pop(what, None)

    def _overflow(self, what):
        if what not in self._full:
            self._full[what] = self.t
            self.overflows.append((what, self.t))

    def _room(self, what):
        self._full.pop(what, None)

    # =========================================================== faults
    def _next_fault(self, r, t):
        mtbf = Faults.ROBOT_MTBF_H if r.name == "robot" else Faults.STATION_MTBF_H
        # running hours to wall-clock: the line runs hours_day of every 24
        dt = self.rng.exponential(mtbf * 3600.0 * 24.0 / self.ln.hours_day)
        self.at(t + dt, self._fault, r)

    def _fault(self, r):
        if not self.open(self.t):
            self.at(self.next_open(self.t), self._fault, r)
            return
        mttr = (Faults.ROBOT_MTTR_MIN if r.name == "robot" else Faults.STATION_MTTR_MIN) * 60.0
        self._down(r, mttr)
        self._next_fault(r, self.t)

    def _down(self, r, dt):
        r.down_until = max(r.down_until, self.t) + dt
        if r.busy:
            r.token += 1
            r.t_end += dt
            self.at(r.t_end, self._finish, r, r.token)
        self.at(r.down_until, self._noop)

    def outage(self, t, hours):
        """The robot out for `hours` from t: the longest the line rides through."""
        self.at(t, lambda: self._down(self.res["robot"], hours * 3600.0))

    # ============================================================== jobs
    def _start(self, name, dt, done, *args):
        r = self.res[name]
        r.busy, r.token = True, r.token + 1
        r.t_end = self.t + dt
        r.done = (done, args)
        r.busy_s += dt
        self.at(r.t_end, self._finish, r, r.token)
        return True

    def _finish(self, r, token):
        if token != r.token or not r.busy:
            return
        r.busy = False
        fn, args = r.done
        fn(*args)

    def dispatch(self):
        if not self.open(self.t):
            return
        for n in L.STATIONS + ("robot",):
            r = self.res[n]
            if not r.busy and r.down_until <= self.t:
                getattr(self, "_next_" + ("tray" if n in TRAY_STATIONS else n))(*(
                    (n,) if n in TRAY_STATIONS else ()))

    # ========================================================= station 1
    def frames_on_hand(self):
        """Good frames not yet promised to an order: in trays anywhere,
        on mandrels curing or cured, and the one in the cage."""
        n = 0
        for tr in self.trays.values():
            if tr.kind == "bat":
                n += sum(1 for b in tr.slots if b is not None and self.bats[b].ok
                         and self.bats[b].order is None)
            elif tr.kind == "mandrel":
                n += sum(1 for m in tr.slots if isinstance(m, tuple) and m[1])
        r = self.res["frame"]
        if r.busy and r.done[0] == self._built:
            n += 1
        return n

    def _next_frame(self):
        st = self.P.sts["frame"]
        ft = self.dock("frame/frames")
        docks = [self.dock("frame/mandrels A"), self.dock("frame/mandrels B")]
        # collapse whatever has cured, first
        for mt in docks:
            if mt is None:
                continue
            for i, m in enumerate(mt.slots):
                if isinstance(m, tuple) and m[0] <= self.t:
                    if m[1] and (ft is None or None not in ft.slots):
                        continue
                    mt.slots[i] = "cage"
                    return self._start("frame", st.job_s("collapse"), self._collapsed, mt, i, m[1])
        # then build, while the frames on hand are short of the stock
        if self.frames_on_hand() >= self.P.sizes["frame_stock"]["frames"]:
            return
        slot = next(((mt, i) for mt in docks if mt is not None
                     for i, m in enumerate(mt.slots) if m is None), None)
        if slot is None:
            return
        f = self.P.bom["frame"]
        ct = self.dock("frame/cores")
        if ct is None or ct.kits < 1:
            if not any(t.kits > 0 for t in self.rack("inbound") if t.station == "frame"):
                self._short("cores")
            return
        new_nozzle = self.t - self.last_dose > Epoxy.MIXER_LIFE_MIN * 60.0
        need = {"rods": f["rods"], "thread_m": f["thread_m"],
                "resin_ml": f["resin_ml"] + (Epoxy.PURGE_ML if new_nozzle else 0.0),
                "nozzles": 1 if new_nozzle else 0}
        for k, v in need.items():
            if self.cons[k] < v - 1e-9:
                self._short(k)
                return
        for k, v in need.items():
            self.cons[k] -= v
            self._mark(k, self.cons[k])
        self.resin_used += f["resin_ml"]
        if new_nozzle:
            self.nozzles += 1
            self.purged += Epoxy.PURGE_ML
        ct.kits -= 1
        mt, i = slot
        mt.slots[i] = "cage"
        dt = st.job_s("build") + (st.job_s("session") if new_nozzle else 0.0)
        return self._start("frame", dt, self._built, mt, i)

    def _built(self, mt, i):
        ok = self.rng.random() >= Faults.REJECT["frame"]
        mt.slots[i] = (self.t + Epoxy.HANDLING_H * 3600.0, ok)
        self.last_dose = self.t
        self.built += 1
        self.at(mt.slots[i][0], self._noop)

    def _collapsed(self, mt, i, ok):
        mt.slots[i] = None
        if not ok:
            self.frame_rejects += 1
            self.reject_bin += 1
            self._mark("reject bin", self.reject_bin)
            if self.reject_bin > self.P.sizes["shelves"]["reject bin"]:
                self._overflow("reject bin")
            return
        ft = self.dock("frame/frames")
        ft.slots[ft.slots.index(None)] = self._new_bat()

    # =================================================== stations 2 to 5
    def _parts(self, name):
        return self.dock("%s/parts" % name)

    def _next_tray(self, name):
        bt = self.dock(name + "/bats")
        if bt is None or name in bt.done:
            return
        st = self.P.sts[name]
        if name not in bt.read:
            bt.read.add(name)
            return self._start(name, st.job_s("tray"), self._noop)
        if name == "calibration" and self.golden_day != floor_day(self.t):
            self.golden_day = floor_day(self.t)
            return self._start(name, st.job_s("day"), self._noop)
        for i, b in enumerate(bt.slots):
            if b is None:
                continue
            bat = self.bats[b]
            if name in bat.done:
                continue
            if bat.order is None and bat.ok:          # a frame riding along unpromised
                bat.done.add(name)
                continue
            if not bat.ok:
                if name != "pack":
                    bat.done.add(name)
                    continue
                if self.reject_shelf >= self.P.sizes["shelves"]["reject shelf"]:
                    self._overflow("reject shelf")
                    return
                return self._start(name, st.job_s("reject"), self._shelved, bt, i)
            # what the job needs
            pt = self._parts(name) if name in ("electronics", "sleeve", "pack") else None
            if pt is not None and pt.kits < 1:
                if not any(t.kits > 0 for t in self.rack("inbound") if t.station == name):
                    self._short(name + " parts")
                return
            if name == "sleeve":
                xt = self.dock("sleeve/stretchers")
                if xt is None or xt.id != bt.bound or bat.order not in xt.slots:
                    return
            if name == "pack":
                tt = self.dock("pack/tubes")
                if tt is None or None not in tt.slots:
                    return
                for k in ("tube", "tube_cap"):
                    if self.bulk[k] < 1:
                        self._short(k)
                        return
                if self.cons["labels"] < 1:
                    self._short("labels")
                    return
            if pt is not None:
                pt.kits -= 1
            dt = st.job_s("bat")
            if name == "calibration" and self.rng.random() < 1.0 / Est.SWING_EVERY:
                dt += st.job_s("swing")
            return self._start(name, dt, self._did, name, bt, i)
        bt.done.add(name)

    def _did(self, name, bt, i):
        bat = self.bats[bt.slots[i]]
        bat.done.add(name)
        order = self.orders[bat.order]
        if name == "sleeve":
            xt = self.trays[bt.bound]
            xt.slots[xt.slots.index(order.id)] = None
            self._sleeve(order, "fitted")
        if name == "pack":
            for k in ("tube", "tube_cap"):
                self.bulk[k] -= 1
                self._mark("store " + k, self.bulk[k])
            self.cons["labels"] -= 1
        if self.rng.random() < Faults.REJECT[name]:
            bat.ok = False
            self.rejects += 1
            order.bat, bat.order = None, None
            # before its sleeve is fitted, the order keeps its sleeve
            self._sleeve(order, "ready" if name == "electronics" else "waiting")
            if name != "electronics":
                order.tray = None
                order.remakes += 1
            if name == "pack":
                self._shelf_one()
                bt.slots[i] = None
            return
        if name == "pack":
            tt = self.dock("pack/tubes")
            tt.slots[tt.slots.index(None)] = bat.id
            bt.slots[i] = None

    def _shelved(self, bt, i):
        self._shelf_one()
        self.bats[bt.slots[i]].done.add("pack")
        bt.slots[i] = None

    def _shelf_one(self):
        self.reject_shelf += 1
        self._mark("reject shelf", self.reject_shelf)
        if self.reject_shelf > self.P.sizes["shelves"]["reject shelf"]:
            self._overflow("reject shelf")

    # ============================================================ robot
    def _move(self, tr, dest):
        src = tr.loc
        fl = self.P.floor
        dt = (fl.trip_s(self.robot_at, src) + Est.DOCK_S + fl.trip_s(src, dest) + Est.DOCK_S)
        self._take(tr)
        self.incoming[dest] = self.incoming.get(dest, 0) + 1
        self.robot_at = dest
        return self._start("robot", dt, self._moved, tr, dest)

    def _moved(self, tr, dest):
        self.incoming[dest] -= 1
        self._place(tr, dest)
        if tr.kind == "bat" and dest == "stock":        # back from its pass, or from station 1
            tr.bound, tr.done, tr.read = None, set(), set()
            for s in tr.slots:
                if s is not None:
                    self.bats[s].done = set()
        if tr.kind == "stretcher" and dest == "sleeves":
            tr.away = False

    def _bound_trays(self, xt):
        return [t for t in self.trays.values() if t.kind == "bat" and t.bound == xt.id]

    def _next_robot(self):
        for tr, dest in self._wants():
            if self._free(dest):
                return self._move(tr, dest)

    def _wants(self):
        """Every move the line wants, most urgent first: downstream before
        upstream, so the line drains before it fills."""
        k = self.k
        # 1. packed tubes out when the tray is full or no more are coming;
        #    an empty tube tray in
        tt = self.dock("pack/tubes")
        pb = self.dock("pack/bats")
        coming = any(t.kind == "bat" and t.bound is not None and "pack" not in t.done
                     for t in self.trays.values())
        if tt is not None and any(tt.slots) and (None not in tt.slots or not coming):
            if not self._free("outbound"):
                self._overflow("outbound")
            yield tt, "outbound"
        if tt is None and self._free("pack/tubes"):
            empty = [t for t in self.rack("outbound") if not any(t.slots)]
            if empty:
                self._room("outbound")
                yield empty[0], "pack/tubes"
            elif pb is not None and "pack" not in pb.done:
                self._overflow("outbound")
        # 2. bat trays onward
        for a, b in (("pack/bats", "stock"), ("calibration/bats", "pack/bats"),
                     ("sleeve/bats", "calibration/bats")):
            bt = self.dock(a)
            if bt is not None and a.split("/")[0] in bt.done:
                yield bt, b
        # 3. a tray of sleeves to station 3 before its frames, and back after
        xt = self.dock("sleeve/stretchers")
        if xt is not None and not any("sleeve" not in t.done for t in self._bound_trays(xt)):
            yield xt, "sleeves"
        eb = self.dock("electronics/bats")
        for bt in [t for t in (eb, self.dock("sleeve/bats")) if t is not None and t.bound is not None]:
            want = self.trays[bt.bound]
            if want.loc == "sleeves":
                yield want, "sleeve/stretchers"
        if eb is not None and "electronics" in eb.done:
            xs = self.dock("sleeve/stretchers")
            if xs is not None and xs.id == eb.bound:
                yield eb, "sleeve/bats"
        # 4. parts trays swapped before they run out
        for st in ("frame", "electronics", "sleeve", "pack"):
            dock = "%s/%s" % (st, "cores" if st == "frame" else "parts")
            low = 1 if st == "frame" else k
            pt = self.dock(dock)
            full = sorted((t for t in self.rack("inbound") if t.station == st), key=lambda t: -t.kits)
            if pt is not None and pt.kits < low and full and full[0].kits > pt.kits:
                yield pt, "inbound"
            if pt is None and full and full[0].kits > 0:
                yield full[0], dock
        # 5. a tray of frames released against a tray of sleeves
        if self._free("electronics/bats"):
            rel = self._release()
            if rel is not None:
                yield rel, "electronics/bats"
        # 6. station 1: frames to stock, a tray with room back; mandrels to cure and back
        ft = self.dock("frame/frames")
        r1 = self.res["frame"]
        if ft is not None and not (r1.busy and r1.done[0] == self._collapsed):
            frames = sum(1 for b in ft.slots if b is not None)
            starving = frames and not self._stock_frames() and self.by_state["ready"]
            if None not in ft.slots or starving:
                yield ft, "stock"
        elif ft is None and self._free("frame/frames"):
            room = [t for t in self.rack("stock") if None in t.slots and t.bound is None]
            if room:
                yield max(room, key=lambda t: t.slots.count(None)), "frame/frames"
        cured_waiting = any(isinstance(m, tuple) and m[0] <= self.t
                            for t in self.rack("cure") for m in t.slots)
        for dk in ("frame/mandrels A", "frame/mandrels B"):
            mt = self.dock(dk)
            if mt is None or "cage" in mt.slots:
                pass
            elif all(isinstance(m, tuple) and m[0] > self.t for m in mt.slots):
                yield mt, "cure"                     # all wound: cure out of the way
            elif cured_waiting and not any(isinstance(m, tuple) and m[0] <= self.t for m in mt.slots):
                yield mt, "cure"                     # make room for frames that are ready
            if mt is None and self._free(dk):
                cured = [t for t in self.rack("cure")
                         if any(isinstance(m, tuple) and m[0] <= self.t for m in t.slots)]
                spare = [t for t in self.rack("cure") if None in t.slots]
                pick = (min(cured, key=lambda t: min(m[0] for m in t.slots if isinstance(m, tuple)))
                        if cured else (spare[0] if spare else None))
                if pick is not None:
                    yield pick, dk

    def _stock_frames(self):
        return sum(1 for t in self.rack("stock") for b in t.slots
                   if b is not None and self.bats[b].ok and self.bats[b].order is None)

    def _release(self):
        """Bind a tray of stock frames to the oldest tray of sleeves; the
        tray to send, or None."""
        ready = [self.orders[i] for i in self.by_state["ready"]]
        ready = [o for o in ready if not self.trays[o.tray].away and self.trays[o.tray].loc == "sleeves"]
        if not ready:
            return None
        oldest = min(ready, key=lambda o: o.t_in)
        xt = self.trays[oldest.tray]
        mine = sorted((o for o in ready if o.tray == xt.id), key=lambda o: o.t_in)
        trays = [t for t in self.rack("stock") if t.bound is None]
        free = {t.id: [b for b in t.slots if b is not None and self.bats[b].ok
                       and self.bats[b].order is None] for t in trays}
        trays = [t for t in trays if free[t.id]]
        if not trays:
            if self._stock_frames() == 0:
                self._short("frame stock")
            return None
        self._back("frame stock")
        # the tray that leaves fewest frames riding along unpromised: the
        # smallest that covers the sleeves, else the fullest
        cover = [t for t in trays if len(free[t.id]) >= len(mine)]
        bt = (min(cover, key=lambda t: len(free[t.id])) if cover
              else max(trays, key=lambda t: len(free[t.id])))
        for o, b in zip(mine, free[bt.id]):
            self._sleeve(o, "bound")
            o.bat = b
            self.bats[b].order = o.id
        bt.bound, bt.done, bt.read = xt.id, set(), set()
        for s in bt.slots:
            if s is not None:
                self.bats[s].done = set()
        xt.away = True
        return bt

    # =========================================================== orders
    def _order(self):
        o = Order(len(self.orders), self.t)
        self.orders[o.id] = o
        self.by_state["waiting"].add(o.id)

    def _sleeve(self, o, state):
        self.by_state[o.sleeve].discard(o.id)
        o.sleeve = state
        self.by_state[state].add(o.id)

    # ======================================================= the person
    def _visit(self, day):
        P, ln = self.P, self.ln
        mins = self.person
        add = lambda k, v: mins.__setitem__(k, mins.get(k, 0.0) + v)  # noqa: E731
        # 5. dispatch: every packed tube goes; the tray stays, empty
        for tr in self.rack("outbound"):
            for i, b in enumerate(tr.slots):
                if b is not None:
                    o = self.orders[self.bats[b].order]
                    o.t_out = self.t
                    tr.slots[i] = None
                    add("dispatch", Person.DISPATCH_MIN)
        self._room("outbound")
        # 6. rejects cleared, boards recovered
        n = self.reject_shelf + self.reject_bin
        add("rejects", Person.REJECT_MIN * n)
        self.reject_shelf = self.reject_bin = 0
        self._room("reject shelf"), self._room("reject bin")
        # 3. the printers unloaded
        for pr in self.printers:
            if pr["t_done"] is not None and pr["t_done"] <= self.t:
                add("printers", Person.UNLOAD_MIN)
                if pr["good"]:
                    self.bulk[pr["part"]] += P.sizes["printers"][pr["part"]]["per_plate"]
                pr["t_done"] = None
        # 1. bought parts restocked to a restock's worth
        if day % ln.restock_every_days == 0:
            for k, v in P.sizes["bought"].items():
                self.bulk[k] = max(self.bulk[k], v)
                self._back(k)
        # ... and the parts trays in the inbound rack filled from the stores
        for tr in self.rack("inbound"):
            add("parts", Person.STOCK_MIN * self._fill_tray(tr))
        # 2. consumables: a unit that will not last to the next visit is changed
        S = P.sizes["consumables"]
        for k, v in S.items():
            if k == "rods":
                cut = self.cons_cap[k] - self.cons[k]
                if cut > 0:
                    add("rods", (Person.CUT_ROD_MIN * cut if ln.rods_by_hand else 0.0)
                        + Person.CONSUMABLE_MIN * ceil(cut / Rod.MAGAZINE))
                    self.cons[k] = self.cons_cap[k]
            elif k == "nozzles":
                if self.cons[k] < self.cons_cap[k]:
                    add("consumables", Person.CONSUMABLE_MIN)
                    self.cons[k] = self.cons_cap[k]
            elif self.cons[k] < v["use"]:
                units = int(ceil((self.cons_cap[k] - self.cons[k]) / v["per"] - 1e-9))
                add("consumables", Person.CONSUMABLE_MIN * units)
                self.cons[k] = self.cons_cap[k]
            self._back(k)
        # 4. sleeves for every order waiting for one
        for o in sorted((self.orders[i] for i in self.by_state["waiting"]), key=lambda o: o.t_in):
            xt = next((t for t in self.rack("sleeves") if not t.away and None in t.slots), None)
            if xt is None:
                self._overflow("stretchers")
                break
            xt.slots[xt.slots.index(None)] = o.id
            self._sleeve(o, "ready")
            o.tray = xt.id
            add("sleeves", Person.SLEEVE_MIN)
        # 3. the printers restarted: each part printed back up to its level
        for part, v in P.sizes["printers"].items():
            pos = self.bulk[part] + self._in_trays(part) + sum(
                v["per_plate"] for pr in self.printers if pr["part"] == part and pr["t_done"] is not None)
            want = int(ceil(max(0, v["order_up_to"] - pos) / v["per_plate"]))
            for pr in self.printers:
                if want <= 0:
                    break
                if pr["part"] == part and pr["t_done"] is None:
                    pr["t_done"] = self.t + v["plate_h"] * 3600.0
                    pr["good"] = self.rng.random() >= Faults.PRINT_FAIL
                    self.plates += 1
                    self.plates_failed += 0 if pr["good"] else 1
                    want -= 1
        for k in self.bulk:
            self._mark("store " + k, self.bulk[k])
        # whatever is still short after the visit is logged again when it bites
        self._dry.clear()

    def _in_trays(self, part):
        return sum(t.kits for t in self.trays.values()
                   if t.kind == "parts" and part in self.P.sizes["parts_trays"][t.station]["parts"])

    def _fill_tray(self, tr, count=True):
        """Top a parts tray up from the stores; the kits it took."""
        spec = self.P.sizes["parts_trays"][tr.station]
        room = spec["kits"] - tr.kits
        n = min([room] + [self.bulk[p] for p in spec["parts"]])
        for p in spec["parts"]:
            self.bulk[p] -= n
            self._mark("store " + p, self.bulk[p])
        tr.kits += n
        return n

    # ========================================================== results
    def result(self):
        ln = self.ln
        win = [o for o in self.orders.values() if o.t_in < self.window * DAY]
        shipped = [o for o in win if o.t_out is not None]
        lead = sorted((o.t_out - o.t_in) / DAY for o in shipped)
        late = [o for o in shipped if (o.t_out - o.t_in) / DAY > ln.promise_days]
        remade = sum(1 for o in late if (o.t_out - o.t_in) / DAY
                     <= ln.promise_days + o.remakes * ln.visit_every_days)
        open_s = self.horizon / DAY * ln.hours_day * 3600.0
        busy = {n: r.busy_s / open_s for n, r in self.res.items()}
        n_ship = len([o for o in self.orders.values() if o.t_out is not None])
        total = sum(self.person.values())
        per_bat = {n: r.busy_s / max(n_ship, 1) for n, r in self.res.items()}
        return Result(self.P, self.seed, self.window, len(win), len(shipped), len(late), remade,
                      len(win) - len(shipped), lead, self.stockouts, self.overflows, busy, per_bat,
                      dict(self.person), total / max(n_ship, 1), self.built, self.frame_rejects,
                      self.rejects, self.nozzles, self.purged, self.resin_used, self.plates,
                      self.plates_failed, dict(self.peak), dict(self.low))


def floor_day(t):
    return int(t // DAY)


def run_month(plan, seed=0, faults=True, outage_day=None, days=None):
    """One month of the line; the robot's longest outage injected on
    `outage_day` at the visit, when the most trays want moving."""
    sim = LineSim(plan, seed=seed, faults=faults, days=days)
    if outage_day is not None:
        sim.outage(outage_day * DAY + plan.line.visit_at_h * 3600.0, plan.line.outage_h)
    return sim.run()


def summary(r):
    ln = r.plan.line
    lead = r.lead_days
    out = []
    w = out.append
    w("month at %.0f, seed %d: %d orders, %d shipped, %d late (%d for a re-made sleeve), %d unshipped; "
      "lead time p50 %.2f p99 %.2f max %.2f days"
      % (ln.demand_month, r.seed, r.orders, r.shipped, r.late, r.late_remade, r.unshipped,
         np.percentile(lead, 50) if lead else 0, np.percentile(lead, 99) if lead else 0,
         max(lead) if lead else 0))
    w("  busy: " + ", ".join("%s %.1f%%" % (k, 100 * v) for k, v in r.busy.items()))
    w("  frames built %d (%d rejected), bats rejected %d; nozzles %d, purged %.0f of %.0f ml"
      % (r.built, r.frame_rejects, r.rejects, r.nozzles, r.purged_ml, r.resin_ml + r.purged_ml))
    w("  plates %d (%d failed); person %.1f min a bat (%s)"
      % (r.plates, r.plates_failed, r.person_per_bat,
         ", ".join("%s %.0f" % kv for kv in sorted(r.person_min.items()))))
    for name, ev in (("stockouts", r.stockouts), ("overflows", r.overflows)):
        w("  %s: %s" % (name, ", ".join("%s x%d from day %.1f" % (k, n, d)
                                         for k, (n, d) in sorted(r.kinds(ev).items())) or "none"))
    return "\n".join(out)
