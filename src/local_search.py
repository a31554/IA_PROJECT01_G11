"""Pesquisa local: Hill Climbing (best improvement) com vizinhança relocate."""
from collections import Counter

def _insertions(route, item):
    """Todas as rotas que resultam de inserir P(item) e D(item) (P antes de D)."""
    for i in range(len(route) + 1):
        for j in range(i, len(route) + 1):
            yield route[:i] + [("P", item)] + route[i:j] + [("D", item)] + route[j:]

def hill_climbing(p, routes, max_iter=500):
    routes = {v: list(r) for v, r in routes.items()}
    veh = p.veh_by_id
    assigned = [(kind, item) for route in routes.values() for kind, item in route]
    expected = Counter(r.item_id for r in p.requests)
    expected_events = Counter(
        event for item in expected for event in (("P", item), ("D", item))
    )
    if Counter(assigned) != expected_events:
        raise ValueError(
            "Initial routes must contain exactly one pickup and one delivery "
            "for every request"
        )
    cost = {v: p.cost_function(r, veh[v]) for v, r in routes.items()}
    history = [sum(cost.values())]

    for _ in range(max_iter):
        best_gain, best_move = 1e-9, None
        for r in p.requests:
            i = r.item_id
            src = next(v for v, rt in routes.items()
                       if any(x == i for _, x in rt))
            src_wo = [s for s in routes[src] if s[1] != i]
            src_wo_cost = p.cost_function(src_wo, veh[src])
            for dst in routes:
                base = src_wo if dst == src else routes[dst]
                old = cost[src] + (cost[dst] if dst != src else 0.0)
                for cand in _insertions(base, i):
                    c = p.cost_function(cand, veh[dst])
                    new = c + (src_wo_cost if dst != src else 0.0)
                    if old - new > best_gain:
                        best_gain, best_move = old - new, (src, dst, src_wo, cand,
                                                           src_wo_cost, c)
        if best_move is None:
            break
        src, dst, src_wo, cand, c_src, c_dst = best_move
        if src != dst:
            routes[src], cost[src] = src_wo, c_src
        routes[dst], cost[dst] = cand, c_dst
        history.append(sum(cost.values()))
    return routes, history