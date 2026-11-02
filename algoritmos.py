"""
algoritmos.py  -  Agente de rotas (pickup & delivery com janelas temporais)

Conteúdo:
    1. Modelação        Vehicle, Request, State, Problem (+ viabilidade e custo)
    2. Pesquisa informada   Greedy e A* com 2 heurísticas
    3. Pesquisa local       Hill Climbing
    4. CSP                  Backtracking + MRV + forward checking + LCV

CONVENÇÕES
  * Tempos em MINUTOS desde t0 (primeira hora de disponibilidade dos pedidos).
  * Um pedido (Request) = um item de encomenda: recolher no produtor e entregar no cliente.
  * Uma ROTA é uma lista de paragens [("P", item_id), ("D", item_id), ...]
    ("P" = recolha no produtor, "D" = entrega ao cliente).
    O veículo parte e regressa ao centro de recolha.
  * Uma SOLUÇÃO é um dicionário {vehicle_id: rota}.
  * REGRA DO ARMAZÉM: o veículo NÃO entrega diretamente o que recolheu. Tem de
    passar pelo armazém (centro de recolha) antes de entregar. Na prática, ao ir
    entregar um item que ainda não passou pelo armazém, a viagem é
    local atual -> armazém -> cliente; ao passar no armazém, tudo o que vai a
    bordo fica "armazenado" e já pode ser entregue diretamente.
  * Chegar antes da janela abrir => o veículo espera (sem custo).
    Chegar depois da janela fechar => rota inviável.
  * Custo = km totais percorridos.
"""
import heapq
import itertools
import math
from dataclasses import dataclass
from time import time as now

import pandas as pd


# =============================================================================
# 1. MODELAÇÃO
# =============================================================================
class Vehicle:
    """Veículo de entrega associado a um centro de recolha."""
    def __init__(self, vehicle_id, capacity_kg, speed_kmh, depot_loc, refrigerated=False):
        self.vehicle_id = vehicle_id
        self.capacity = capacity_kg
        self.speed = speed_kmh
        self.depot_loc = depot_loc        # location_id do centro (NÃO o center_id!)
        self.refrigerated = refrigerated


class Request:
    """Item de encomenda: recolher num produtor e entregar num cliente."""
    def __init__(self, item_id, order_id, product_id, weight,
                 pickup_loc, delivery_loc,
                 availability_start, availability_end,
                 delivery_start, delivery_end):
        self.item_id = item_id
        self.order_id = order_id
        self.product_id = product_id
        self.weight = weight
        self.pickup_loc = pickup_loc
        self.delivery_loc = delivery_loc
        self.availability_window = (availability_start, availability_end)  # recolha
        self.delivery_window = (delivery_start, delivery_end)              # entrega


class State:
    """
    Estado da pesquisa informada. Os veículos são construídos UM DE CADA VEZ
    (primeiro o 0, depois o 1...), o que evita explorar a mesma solução em
    milhares de ordens diferentes.
        vi        índice do veículo que está a ser construído
        loc, time, load   onde está, a que horas e quantos kg leva
        picked / delivered    itens já recolhidos / já entregues
        stored    itens que já passaram pelo armazém (a bordo e prontos a entregar)
        cost      km acumulados
        parent, action    para reconstruir as rotas no fim
    """
    def __init__(self, vi, loc, time, load, picked, delivered, stored, cost,
                 parent=None, action=None):
        self.vi, self.loc, self.time, self.load = vi, loc, time, load
        self.picked, self.delivered, self.stored = picked, delivered, stored
        self.cost, self.parent, self.action = cost, parent, action


class Problem:
    """Carrega os dados e define as regras do problema."""
    def __init__(self, data_path="data_tiny/", exclude_status=("delivered",),
                 drop_infeasible=True, verbose=True):
        p = data_path
        self.locations = pd.read_csv(f"{p}locations.csv")
        self.products = pd.read_csv(f"{p}products.csv")
        self.orders = pd.read_csv(f"{p}orders.csv")
        self.items = pd.read_csv(
            f"{p}order_items.csv",
            parse_dates=["availability_start", "availability_end",
                         "delivery_start", "delivery_end"])
        self.vehicles_df = pd.read_csv(f"{p}vehicles.csv")
        dm = pd.read_csv(f"{p}distance_matrix.csv")

        # (origem, destino) -> (km, minutos)
        self.dist = {(r.origin_location_id, r.destination_location_id):
                     (r.distance_km, r.duration_min) for r in dm.itertuples()}
        # (tipo, ref_id) -> location_id   (os ids da matriz são os location_id)
        self.loc_id = {(r.location_type, r.ref_id): r.location_id
                       for r in self.locations.itertuples()}

        self.vehicles = self._load_vehicles()
        self.requests = self._load_requests(exclude_status)
        self.req_by_id = {r.item_id: r for r in self.requests}
        self.veh_by_id = {v.vehicle_id: v for v in self.vehicles}
        self.depot = self.vehicles[0].depot_loc      # todos partem do mesmo centro

        # Pedidos IMPOSSÍVEIS mesmo sozinhos (ex.: com a regra do armazém a janela
        # de entrega fecha antes de ser possível chegar). Não há algoritmo que os
        # consiga servir, por isso são retirados do planeamento e ficam em
        # self.dropped para serem reportados.
        self.dropped = []
        if drop_infeasible:
            ok = []
            for r in self.requests:
                alone = [("P", r.item_id), ("D", r.item_id)]
                if any(self.route_feasibility_checker(alone, v) for v in self.vehicles):
                    ok.append(r)
                else:
                    self.dropped.append(r)
            if self.dropped and verbose:
                print(f"[Problem] AVISO: pedido(s) impossíveis de cumprir mesmo sozinhos "
                      f"e retirados do planeamento: {[r.item_id for r in self.dropped]}")
            self.requests = ok
            self.req_by_id = {r.item_id: r for r in ok}

    def _load_vehicles(self):
        return [Vehicle(r.vehicle_id, r.capacity_kg, r.avg_speed_kmh,
                        self.loc_id[("collection_center", r.center_id)],
                        bool(r.refrigerated))
                for r in self.vehicles_df.itertuples()]

    def _load_requests(self, exclude_status):
        df = (self.items
              .merge(self.products[["product_id", "producer_id", "unit_weight_kg"]],
                     on="product_id")
              .merge(self.orders[["order_id", "delivery_location_id", "status"]],
                     on="order_id"))
        df = df[~df["status"].isin(exclude_status)]   # ignora encomendas já entregues

        # TEMPO: datas -> minutos desde t0 (comparar janelas = comparar números)
        self.t0 = df["availability_start"].min()
        mins = lambda ts: (ts - self.t0).total_seconds() / 60

        reqs = []
        for r in df.itertuples():
            reqs.append(Request(
                item_id=r.order_item_id, order_id=r.order_id, product_id=r.product_id,
                weight=r.quantity * r.unit_weight_kg,
                pickup_loc=self.loc_id[("producer", r.producer_id)],
                delivery_loc=self.loc_id[("delivery_location", r.delivery_location_id)],
                availability_start=mins(r.availability_start),
                availability_end=mins(r.availability_end),
                delivery_start=mins(r.delivery_start),
                delivery_end=mins(r.delivery_end)))
        return reqs

    def travel(self, a, b):
        """(km, minutos) entre dois locais; 0 se for o mesmo."""
        return (0.0, 0.0) if a == b else self.dist[(a, b)]

    def move(self, vehicle, loc, t, load, onboard, stored, kind, item):
        """
        Executa UMA paragem a partir da situação atual. Devolve
        (novo_local, nova_hora, nova_carga, km, novo_stored) ou None se for inviável.
          "P": recolher no produtor  (janela de recolha + capacidade)
          "D": entregar ao cliente   (janela de entrega). Se o item ainda não passou
               pelo armazém, a viagem faz-se por lá: local -> armazém -> cliente, e
               tudo o que vai a bordo fica armazenado.
        Esta função é partilhada pelo verificador, pela pesquisa informada e pelo
        CSP, por isso as regras do problema estão num só sítio.
        """
        r = self.req_by_id[item]
        if kind == "P":
            d, m = self.travel(loc, r.pickup_loc)
            t2 = max(t + m, r.availability_window[0])        # chega cedo -> espera
            if t2 > r.availability_window[1]:                # chega tarde -> inviável
                return None
            if load + r.weight > vehicle.capacity:           # capacidade excedida
                return None
            return r.pickup_loc, t2, load + r.weight, d, stored
        if item not in onboard:                              # entrega sem recolha
            return None
        km = 0.0
        if item not in stored:                               # TEM de passar pelo armazém
            d1, m1 = self.travel(loc, vehicle.depot_loc)
            km, t, loc = d1, t + m1, vehicle.depot_loc
            stored = frozenset(onboard)                      # tudo o que vai a bordo
        d, m = self.travel(loc, r.delivery_loc)
        t2 = max(t + m, r.delivery_window[0])
        if t2 > r.delivery_window[1]:
            return None
        return r.delivery_loc, t2, load - r.weight, km + d, stored - {item}

    def _simulate(self, route, vehicle):
        """
        Percorre a rota e devolve (viável?, km totais). Verifica:
          - janela de recolha e janela de entrega (espera se chegar cedo)
          - capacidade em cada instante (não só no total)
          - recolha antes da entrega, passagem pelo armazém antes de entregar
          - nada fica por entregar; o veículo regressa ao centro
        """
        loc, t, load, km = vehicle.depot_loc, 0.0, 0.0, 0.0
        picked, delivered, stored = frozenset(), frozenset(), frozenset()
        for kind, item in route:
            if (kind == "P" and item in picked) or (kind == "D" and item in delivered):
                return False, km                              # repetido
            mv = self.move(vehicle, loc, t, load, picked - delivered, stored, kind, item)
            if mv is None:
                return False, km
            loc, t, load, d, stored = mv
            km += d
            if kind == "P":
                picked = picked | {item}
            else:
                delivered = delivered | {item}
        if picked != delivered:                               # ficou algo por entregar
            return False, km
        if route:
            km += self.travel(loc, vehicle.depot_loc)[0]      # regresso ao centro
        return True, km

    def route_feasibility_checker(self, route, vehicle):
        """True se a rota cumpre capacidade, janelas e precedências."""
        return self._simulate(route, vehicle)[0]

    def cost_function(self, route, vehicle):
        """km da rota; infinito se for inviável (assim nunca é preferida)."""
        ok, km = self._simulate(route, vehicle)
        return km if ok else math.inf

    def solution_cost(self, routes):
        """Custo total de uma solução {vehicle_id: rota}."""
        return sum(self.cost_function(r, self.veh_by_id[v]) for v, r in routes.items())

    def is_complete(self, routes):
        """Todos os pedidos foram entregues exatamente uma vez?"""
        done = sorted(i for r in routes.values() for k, i in r if k == "D")
        return done == sorted(self.req_by_id)

    # ---------------------------- heurísticas ----------------------------
    def h_admissible(self, vi, loc, picked, delivered, stored):
        """
        HEURÍSTICA 1 (admissível; a matriz cumpre a desigualdade triangular).
        Para cada pedido por entregar calcula-se o mínimo que alguém tem de andar
        para o servir, tendo em conta a regra do armazém:
            já armazenado:     local -> cliente -> centro
            recolhido:         local -> armazém -> cliente -> centro
            por recolher:      local -> produtor -> armazém -> cliente -> centro
        Toma-se o MÁXIMO e não a soma, porque uma viagem pode servir vários
        pedidos (a soma sobrestimaria).
        """
        km, depot = (lambda a, b: self.travel(a, b)[0]), self.depot
        best = km(loc, depot)
        later = vi < len(self.vehicles) - 1      # ainda há veículos por usar?
        for r in self.requests:
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
                if later:   # um veículo seguinte pode servi-lo a partir do centro
                    b = min(b, km(depot, r.pickup_loc) + km(r.pickup_loc, depot)
                            + km(depot, r.delivery_loc) + km(r.delivery_loc, depot))
            best = max(best, b)
        return best

    def h_informative(self, vi, loc, picked, delivered, stored):
        """
        HEURÍSTICA 2 (mais informada, NÃO admissível): soma, para cada pedido por
        servir, a viagem que lhe falta (produtor -> armazém -> cliente se ainda não
        foi recolhido; armazém -> cliente se já vai a bordo) e junta a distância à
        paragem mais próxima. Pode sobrestimar, mas guia muito melhor a pesquisa.
        """
        km = lambda a, b: self.travel(a, b)[0]
        depot = self.depot
        total, nearest = 0.0, math.inf
        for r in self.requests:
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


# =============================================================================
# 2. PESQUISA INFORMADA (Greedy e A*)
# =============================================================================
@dataclass
class SearchResult:
    routes: dict      # {vehicle_id: rota}
    cost: float
    expanded: int     # nós expandidos (métrica pedida no enunciado)
    seconds: float
    status: str       # "solved" | "no_solution" | "limit"


def _dead(p, s):
    """Poda: estados sem saída (alguma janela já não pode ser cumprida)."""
    if s.vi >= len(p.vehicles):
        return False
    depot = p.vehicles[s.vi].depot_loc
    for i in s.picked - s.delivered:                  # item a bordo já não chega a tempo
        r = p.req_by_id[i]
        if i in s.stored:
            m = p.travel(s.loc, r.delivery_loc)[1]
        else:                                         # tem de passar pelo armazém
            m = p.travel(s.loc, depot)[1] + p.travel(depot, r.delivery_loc)[1]
        if max(s.time + m, r.delivery_window[0]) > r.delivery_window[1]:
            return True
    if s.vi == len(p.vehicles) - 1:                   # último veículo: ninguém mais serve
        for r in p.requests:                          # os pedidos já fechados
            if r.item_id not in s.picked and \
               s.time + p.travel(s.loc, r.pickup_loc)[1] > r.availability_window[1]:
                return True
    return False


def _successors(p, s):
    """Ações: recolher um item, entregar um item a bordo, ou fechar a rota."""
    v = p.vehicles[s.vi]
    onboard = s.picked - s.delivered
    out = []
    for r in p.requests:                                          # (a) recolher
        i = r.item_id
        if i in s.picked:
            continue
        mv = p.move(v, s.loc, s.time, s.load, onboard, s.stored, "P", i)
        if mv:
            loc, t, load, d, stored = mv
            out.append(State(s.vi, loc, t, load, s.picked | {i}, s.delivered, stored,
                             s.cost + d, s, ("P", i)))
    for i in onboard:                                             # (b) entregar
        mv = p.move(v, s.loc, s.time, s.load, onboard, s.stored, "D", i)
        if mv:
            loc, t, load, d, stored = mv
            out.append(State(s.vi, loc, t, load, s.picked, s.delivered | {i}, stored,
                             s.cost + d, s, ("D", i)))
    if not onboard:                                               # (c) fechar a rota
        d = p.travel(s.loc, v.depot_loc)[0]
        if len(s.delivered) == len(p.requests):                   # objetivo
            out.append(State(len(p.vehicles), v.depot_loc, 0.0, 0.0, s.picked,
                             s.delivered, frozenset(), s.cost + d, s, ("CLOSE", None)))
        elif s.vi + 1 < len(p.vehicles):                          # próximo veículo
            nxt = p.vehicles[s.vi + 1]
            out.append(State(s.vi + 1, nxt.depot_loc, 0.0, 0.0, s.picked,
                             s.delivered, frozenset(), s.cost + d, s, ("CLOSE", None)))
    return [c for c in out if not _dead(p, c)]


def _routes_from(p, goal):
    """Sobe pelos pais do nó objetivo e reconstrói {vehicle_id: rota}."""
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
    """
    algorithm  "greedy" -> f = h        |  "astar" -> f = g + w*h
    heuristic  "admissible" | "informative"
    w          peso da heurística no A*: w=1 é o A* clássico; w>1 é o A* ponderado
               (mais rápido, já não garante a solução ótima).
    O A* clássico pode não terminar em instâncias difíceis: o limite
    max_expansions devolve status "limit".
    """
    h = p.h_admissible if heuristic == "admissible" else p.h_informative
    n, t_start, tie = len(p.vehicles), now(), itertools.count()

    def f_h(s):
        hv = h(s.vi, s.loc, s.picked, s.delivered, s.stored) if s.vi < n else 0.0
        return (hv if algorithm == "greedy" else s.cost + w * hv), hv

    root = State(0, p.vehicles[0].depot_loc, 0.0, 0.0, frozenset(), frozenset(),
                 frozenset(), 0.0)
    frontier = [(f_h(root)[0], 0, next(tie), root)]
    seen, expanded = {}, 0

    while frontier:
        if expanded >= max_expansions:
            return SearchResult({}, math.inf, expanded, now() - t_start, "limit")
        _, _, _, s = heapq.heappop(frontier)
        if s.vi == n:                                    # todos os itens entregues
            routes = _routes_from(p, s)
            return SearchResult(routes, p.solution_cost(routes), expanded,
                                now() - t_start, "solved")
        key = (s.vi, s.loc, s.picked, s.delivered, s.stored)   # poda de estados repetidos
        if key in seen and seen[key][0] <= s.cost and seen[key][1] <= s.time:
            continue
        seen[key] = (s.cost, s.time)
        expanded += 1
        for c in _successors(p, s):
            # desempate: a igualdade de f prefere nós mais avançados
            heapq.heappush(frontier, (f_h(c)[0], -(len(c.picked) + len(c.delivered)),
                                      next(tie), c))
    return SearchResult({}, math.inf, expanded, now() - t_start, "no_solution")


# =============================================================================
# 3. PESQUISA LOCAL (Hill Climbing)
# =============================================================================
def _insertions(route, item):
    """Todas as rotas que resultam de inserir P(item) e D(item) (P antes de D)."""
    for i in range(len(route) + 1):
        for j in range(i, len(route) + 1):
            yield route[:i] + [("P", item)] + route[i:j] + [("D", item)] + route[j:]


def hill_climbing(p, routes, max_iter=500):
    """
    Hill Climbing de MELHOR melhoria. Vizinhança "relocate": tira um pedido
    (recolha + entrega) da rota e reinsere-o noutra posição, no mesmo veículo
    ou noutro. Em cada iteração aplica o movimento que mais reduz o custo; pára
    num ótimo local. Devolve (rotas, historico) com o custo após cada iteração
    (historico[0] = custo inicial) -> curva custo vs. iteração.
    """
    routes = {v: list(r) for v, r in routes.items()}
    veh = p.veh_by_id
    cost = {v: p.cost_function(r, veh[v]) for v, r in routes.items()}
    history = [sum(cost.values())]

    for _ in range(max_iter):
        best_gain, best_move = 1e-9, None
        for r in p.requests:
            i = r.item_id
            src = next(v for v, rt in routes.items() if any(x == i for _, x in rt))
            src_wo = [s for s in routes[src] if s[1] != i]       # rota sem o pedido
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
        if best_move is None:                                    # ótimo local
            break
        src, dst, src_wo, cand, c_src, c_dst = best_move
        if src != dst:
            routes[src], cost[src] = src_wo, c_src
        routes[dst], cost[dst] = cand, c_dst
        history.append(sum(cost.values()))
    return routes, history


# =============================================================================
# 4. CSP (Constraint Satisfaction Problem)
# =============================================================================
"""
Formulação
  Variáveis   X_i = veículo que serve o item i (uma por item)
  Domínios    D(X_i) = veículos que conseguem servir o item i, pelo menos sozinhos
  Restrições  (duras) para cada veículo, o conjunto de itens que lhe foi
              atribuído tem de admitir UMA sequência de paragens viável:
                C1 capacidade em cada instante   C2 janela de recolha
                C3 janela de entrega             C4 recolha antes da entrega
              (verificadas por find_sequence, um DFS com poda)
  Restrição suave (soft): minimizar os km. Não invalida soluções; trata-se com
              a ordenação de valores (LCV): tenta-se primeiro o veículo que
              aumenta menos os km.
Heurísticas:  MRV (variável com menos valores possíveis, desempate pela janela de
              entrega mais apertada), forward checking e LCV.
"""
@dataclass
class CSPResult:
    assignment: dict   # {item_id: vehicle_id}
    routes: dict
    cost: float
    nodes: int         # atribuições tentadas
    backtracks: int    # retrocessos (métrica pedida no enunciado)
    seconds: float
    status: str


def find_sequence(p, vehicle, ids, cap=20_000):
    """Uma sequência viável para o conjunto `ids` neste veículo (ou None).
    DFS que tenta primeiro a paragem cujo prazo fecha mais cedo (EDF).
    Usa Problem.move, por isso respeita todas as regras (incluindo o armazém)."""
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
    """Backtracking. Os 3 booleanos permitem comparar versões no relatório.
    LIMITAÇÃO: a sequência de cada veículo é a primeira viável (não a ótima);
    pode melhorar-se depois com hill_climbing(p, resultado.routes)."""
    t_start, cache = now(), {}
    stats = {"nodes": 0, "bt": 0}

    def seq(vid, ids):                       # find_sequence com cache
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
        if use_mrv:                                  # MRV
            i = min(free, key=lambda x: (len(doms[x]), p.req_by_id[x].delivery_window[1]
                                         - p.req_by_id[x].delivery_window[0]))
        else:
            i = free[0]
        values = list(doms[i])
        if use_lcv:                                  # LCV: menos km extra primeiro
            def extra(v):
                m = members(assign, v)
                a, b = km(v, m + [i]), km(v, m)
                return math.inf if a is None else a - (b or 0)
            values.sort(key=extra)
        for v in values:
            stats["nodes"] += 1
            group = members(assign, v) + [i]
            if seq(v, group) is None:                # viola C1-C4
                stats["bt"] += 1
                continue
            assign[i] = v
            new, ok = doms, True
            if use_forward_checking:                 # forward checking
                new = {k: list(x) for k, x in doms.items()}
                for u in new:
                    if u not in assign and v in new[u] and seq(v, group + [u]) is None:
                        new[u].remove(v)
                        if not new[u]:               # domínio vazio -> falha cedo
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


# =============================================================================
# Demonstração:  python algoritmos.py
# =============================================================================
if __name__ == "__main__":
    prob = Problem("data_tiny/")
    print(f"{len(prob.requests)} pedidos, {len(prob.vehicles)} veículos")

    best = None
    for algo, hname, w in [("greedy", "admissible", 1), ("greedy", "informative", 1),
                           ("astar", "admissible", 1), ("astar", "informative", 1),
                           ("astar", "informative", 1.5), ("astar", "informative", 2)]:
        r = informed_search(prob, algo, hname, w=w, max_expansions=30_000)
        print(f"{algo:6s} {hname:12s} w={w}: {r.status:8s} custo={r.cost:8.2f} "
              f"expandidos={r.expanded:6d} ({r.seconds:.1f}s)")
        if r.status == "solved" and best is None:
            best = r

    if best:
        _, hist = hill_climbing(prob, best.routes)
        print(f"Hill climbing: {hist[0]:.2f} -> {hist[-1]:.2f} km em {len(hist)-1} iterações")

    c = solve_csp(prob)
    print(f"CSP: {c.status}, custo={c.cost:.2f}, retrocessos={c.backtracks}, nós={c.nodes}")
