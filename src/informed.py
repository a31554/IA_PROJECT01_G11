import heapq
import itertools
import math
from dataclasses import dataclass
from time import time as now
from model import State

@dataclass
class SearchResult:
    routes: dict
    cost: float
    expanded: int
    seconds: float
    status: str

def h_admissible(p, vi, loc, picked, delivered, stored):
    km, depot = (lambda a, b: p.travel(a, b)[0]), p.depot
    best = km(loc, depot)
    later = vi < len(p.vehicles) - 1
    for r in p.requests:
        i = r.item_id
        if i in delivered:
            continue
        if i in stored:
            b = km(loc, r.delivery_loc) + km(r.delivery_loc, depot)
        elif i in picked:
            b = km(loc, depot) + km(depot, r.delivery_loc) + km(r.delivery_loc, depot)
        else:
            b = km(loc, r.pickup_loc) + km(r.pickup_loc, depot) \
                + km(depot, r.delivery_loc) + km(r.delivery_loc, depot)
            if later:
                b = min(b, km(depot, r.pickup_loc) + km(r.pickup_loc, depot)
                        + km(depot, r.delivery_loc) + km(r.delivery_loc, depot))
        best = max(best, b)
    return best

def h_informative(p, vi, loc, picked, delivered, stored):
    km = lambda a, b: p.travel(a, b)[0]
    depot = p.depot
    total, nearest = 0.0, math.inf
    for r in p.requests:
        i = r.item_id
        if i in delivered:
            continue
        if i in stored:
            total += km(loc, r.delivery_loc)
            nearest = min(nearest, km(loc, r.delivery_loc))
        elif i in picked:
            total += km(depot, r.delivery_loc)
            nearest = min(nearest, km(loc, depot))
        else:
            total += km(r.pickup_loc, depot) + km(depot, r.delivery_loc)
            nearest = min(nearest, km(loc, r.pickup_loc))
    return km(loc, depot) if nearest == math.inf else total + nearest

def _dead(p, s):
    if s.vi >= len(p.vehicles):
        return False
    depot = p.vehicles[s.vi].depot_loc
    for i in s.picked - s.delivered:
        r = p.req_by_id[i]
        if i in s.stored:
            m = p.travel(s.loc, r.delivery_loc)[1]
        else:
            m = p.travel(s.loc, depot)[1] + p.travel(depot, r.delivery_loc)[1]
        if max(s.time + m, r.delivery_window[0]) > r.delivery_window[1]:
            return True
    if s.vi == len(p.vehicles) - 1:
        for r in p.requests:
            if r.item_id not in s.picked and \
               s.time + p.travel(s.loc, r.pickup_loc)[1] > r.availability_window[1]:
                return True
    return False

def _successors(p, s):
    v = p.vehicles[s.vi]
    onboard = s.picked - s.delivered
    out = []
    for r in p.requests:
        i = r.item_id
        if i in s.picked:
            continue
        mv = p.move(v, s.loc, s.time, s.load, onboard, s.stored, "P", i)
        if mv:
            loc, t, load, d, stored = mv
            out.append(State(s.vi, loc, t, load, s.picked | {i}, s.delivered, stored,
                             s.cost + d, s, ("P", i)))
    for i in onboard:
        mv = p.move(v, s.loc, s.time, s.load, onboard, s.stored, "D", i)
        if mv:
            loc, t, load, d, stored = mv
            out.append(State(s.vi, loc, t, load, s.picked, s.delivered | {i}, stored,
                             s.cost + d, s, ("D", i)))
    if not onboard:
        d = p.travel(s.loc, v.depot_loc)[0]
        if len(s.delivered) == len(p.requests):
            out.append(State(len(p.vehicles), v.depot_loc, 0.0, 0.0, s.picked,
                             s.delivered, frozenset(), s.cost + d, s, ("CLOSE", None)))
        elif s.vi + 1 < len(p.vehicles):
            nxt = p.vehicles[s.vi + 1]
            out.append(State(s.vi + 1, nxt.depot_loc, 0.0, 0.0, s.picked,
                             s.delivered, frozenset(), s.cost + d, s, ("CLOSE", None)))
    return [c for c in out if not _dead(p, c)]

def _routes_from(p, goal):
    routes = {v.vehicle_id: [] for v in p.vehicles}
    s = goal
    while s.parent is not None:
        kind, item = s.action
        if kind != "CLOSE":
            routes[p.vehicles[s.parent.vi].vehicle_id].insert(0, (kind, item))
        s = s.parent
    return routes

def informed_search(p, algorithm="astar", heuristic="admissible", w=1.0,
                    max_expansions=50_000):
    h = h_admissible if heuristic == "admissible" else h_informative
    n, t_start, tie = len(p.vehicles), now(), itertools.count()

    def f_h(s):
        hv = h(p, s.vi, s.loc, s.picked, s.delivered, s.stored) if s.vi < n else 0.0
        return (hv if algorithm == "greedy" else s.cost + w * hv), hv

    root = State(0, p.vehicles[0].depot_loc, 0.0, 0.0, frozenset(), frozenset(),
                 frozenset(), 0.0)
    frontier = [(f_h(root)[0], 0, next(tie), root)]
    seen, expanded = {}, 0

    while frontier:
        if expanded >= max_expansions:
            return SearchResult({}, math.inf, expanded, now() - t_start, "limit")
        _, _, _, s = heapq.heappop(frontier)
        if s.vi == n:
            routes = _routes_from(p, s)
            return SearchResult(routes, p.solution_cost(routes), expanded,
                                now() - t_start, "solved")
        key = (s.vi, s.loc, s.picked, s.delivered, s.stored)
        if key in seen and seen[key][0] <= s.cost and seen[key][1] <= s.time:
            continue
        seen[key] = (s.cost, s.time)
        expanded += 1
        for c in _successors(p, s):
            heapq.heappush(frontier, (f_h(c)[0], -(len(c.picked) + len(c.delivered)),
                                      next(tie), c))
    return SearchResult({}, math.inf, expanded, now() - t_start, "no_solution")