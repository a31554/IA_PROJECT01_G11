import math
from dataclasses import dataclass
from time import time as now

@dataclass
class CSPResult:
    assignment: dict
    routes: dict
    cost: float
    nodes: int
    backtracks: int
    seconds: float
    status: str

def find_sequence(p, vehicle, ids, cap=20_000):
    budget, n_stops = [cap], 2 * len(ids)

    def rec(route, loc, t, load, picked, delivered, stored):
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        if len(route) == n_stops:
            return list(route)
        onboard = picked - delivered
        cands = []
        for i in ids:
            r = p.req_by_id[i]
            if i not in picked:
                kind, prazo = "P", r.availability_window[1]
            elif i not in delivered:
                kind, prazo = "D", r.delivery_window[1]
            else:
                continue
            mv = p.move(vehicle, loc, t, load, onboard, stored, kind, i)
            if mv:
                cands.append((prazo, mv[3], kind, i, mv))
        for _, _, kind, i, mv in sorted(cands, key=lambda c: c[:2]):
            nloc, nt, nload, _, nstored = mv
            route.append((kind, i))
            out = rec(route, nloc, nt, nload,
                      picked | {i} if kind == "P" else picked,
                      delivered | {i} if kind == "D" else delivered, nstored)
            if out is not None:
                return out
            route.pop()
        return None

    return rec([], vehicle.depot_loc, 0.0, 0.0, frozenset(), frozenset(), frozenset())

def solve_csp(p, use_mrv=True, use_forward_checking=True, use_lcv=True,
              max_backtracks=20_000):
    t_start, cache = now(), {}
    stats = {"nodes": 0, "bt": 0}

    def seq(vid, ids):
        key = (vid, frozenset(ids))
        if key not in cache:
            cache[key] = find_sequence(p, p.veh_by_id[vid], list(ids)) if ids else []
        return cache[key]

    def km(vid, ids):
        s = seq(vid, ids)
        return None if s is None else p.cost_function(s, p.veh_by_id[vid])

    domains = {r.item_id: [v.vehicle_id for v in p.vehicles if seq(v.vehicle_id, [r.item_id])]
               for r in p.requests}

    def members(assign, vid):
        return [i for i, v in assign.items() if v == vid]

    def backtrack(assign, doms):
        if len(assign) == len(doms):
            return dict(assign)
        free = [i for i in doms if i not in assign]
        if use_mrv:
            i = min(free, key=lambda x: (len(doms[x]), p.req_by_id[x].delivery_window[1]
                                         - p.req_by_id[x].delivery_window[0]))
        else:
            i = free[0]
        values = list(doms[i])
        if use_lcv:
            def extra(v):
                m = members(assign, v)
                a, b = km(v, m + [i]), km(v, m)
                return math.inf if a is None else a - (b or 0)
            values.sort(key=extra)
        for v in values:
            stats["nodes"] += 1
            group = members(assign, v) + [i]
            if seq(v, group) is None:
                stats["bt"] += 1
                continue
            assign[i] = v
            new, ok = doms, True
            if use_forward_checking:
                new = {k: list(x) for k, x in doms.items()}
                for u in new:
                    if u not in assign and v in new[u] and seq(v, group + [u]) is None:
                        new[u].remove(v)
                        if not new[u]:
                            ok = False
                            break
            if ok:
                out = backtrack(assign, new)
                if out is not None:
                    return out
            del assign[i]
            stats["bt"] += 1
            if stats["bt"] > max_backtracks:
                return None
        return None

    if any(not d for d in domains.values()):
        return CSPResult({}, {}, math.inf, 0, 0, now() - t_start, "no_solution")
    sol = backtrack({}, domains)
    if sol is None:
        status = "limit" if stats["bt"] > max_backtracks else "no_solution"
        return CSPResult({}, {}, math.inf, stats["nodes"], stats["bt"],
                         now() - t_start, status)
    routes = {v.vehicle_id: seq(v.vehicle_id, members(sol, v.vehicle_id))
              for v in p.vehicles}
    return CSPResult(sol, routes, p.solution_cost(routes), stats["nodes"],
                     stats["bt"], now() - t_start, "solved")