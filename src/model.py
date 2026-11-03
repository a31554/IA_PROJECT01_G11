"""Modelação do problema: entidades, custo e verificação de viabilidade."""
import pandas as pd
import math


class Vehicle:
    """Veículo de entrega associado a um centro de recolha."""
    def __init__(self, vehicle_id, capacity_kg, speed_kmh, depot_loc, refrigerated=False):
        self.vehicle_id = vehicle_id
        self.capacity = capacity_kg
        self.speed = speed_kmh
        self.depot_loc = depot_loc
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
        self.availability_window = (availability_start, availability_end)
        self.delivery_window = (delivery_start, delivery_end)


class State:
    """Estado da pesquisa informada."""
    def __init__(self, vi, loc, time, load, picked, delivered, stored, cost,
                 parent=None, action=None):
        self.vi, self.loc, self.time, self.load = vi, loc, time, load
        self.picked, self.delivered, self.stored = picked, delivered, stored
        self.cost, self.parent, self.action = cost, parent, action


class Problem:
    """Carrega os dados e define as regras do problema.

    via_depot=True  -> o produto tem de passar pelo armazém entre a recolha e a entrega
                       (regra de modelação do grupo).
    via_depot=False -> entrega direta produtor -> cliente (sem passagem obrigatória).
    """
    def __init__(self, data_path="data_tiny/", exclude_status=("delivered",),
                 drop_infeasible=True, verbose=True, via_depot=True):
        p = data_path.rstrip("/") + "/"
        self.via_depot = via_depot
        self.locations = pd.read_csv(f"{p}locations.csv")
        self.products = pd.read_csv(f"{p}products.csv")
        self.orders = pd.read_csv(f"{p}orders.csv")
        self.items = pd.read_csv(
            f"{p}order_items.csv",
            parse_dates=["availability_start", "availability_end",
                         "delivery_start", "delivery_end"])
        self.vehicles_df = pd.read_csv(f"{p}vehicles.csv")
        dm = pd.read_csv(f"{p}distance_matrix.csv")

        self.dist = {(r.origin_location_id, r.destination_location_id):
                     (r.distance_km, r.duration_min) for r in dm.itertuples()}
        self.loc_id = {(r.location_type, r.ref_id): r.location_id
                       for r in self.locations.itertuples()}

        self.vehicles = self._load_vehicles()
        self.requests = self._load_requests(exclude_status)
        self.req_by_id = {r.item_id: r for r in self.requests}
        self.veh_by_id = {v.vehicle_id: v for v in self.vehicles}
        # depósito "principal" (usado só para apresentação); as heurísticas e as regras
        # usam sempre o depósito de cada veículo, por isso vários centros são suportados
        self.depot = self.vehicles[0].depot_loc
        self.single_depot = len({v.depot_loc for v in self.vehicles}) == 1

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
        df = df[~df["status"].isin(exclude_status)]

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
        return (0.0, 0.0) if a == b else self.dist[(a, b)]

    def move(self, vehicle, loc, t, load, onboard, stored, kind, item):
        """Aplica uma paragem (P = recolha, D = entrega) e devolve o novo estado
        (loc, t, carga, km, armazenados) ou None se violar alguma restrição."""
        r = self.req_by_id[item]
        if kind == "P":
            d, m = self.travel(loc, r.pickup_loc)
            t2 = max(t + m, r.availability_window[0])
            if t2 > r.availability_window[1]:
                return None
            if load + r.weight > vehicle.capacity:
                return None
            return r.pickup_loc, t2, load + r.weight, d, stored
        if item not in onboard:
            return None
        km = 0.0
        if self.via_depot and item not in stored:
            d1, m1 = self.travel(loc, vehicle.depot_loc)
            km, t, loc = d1, t + m1, vehicle.depot_loc
            stored = frozenset(onboard)
        d, m = self.travel(loc, r.delivery_loc)
        t2 = max(t + m, r.delivery_window[0])
        if t2 > r.delivery_window[1]:
            return None
        return r.delivery_loc, t2, load - r.weight, km + d, stored - {item}

    def _simulate(self, route, vehicle):
        loc, t, load, km = vehicle.depot_loc, 0.0, 0.0, 0.0
        picked, delivered, stored = frozenset(), frozenset(), frozenset()
        for kind, item in route:
            if (kind == "P" and item in picked) or (kind == "D" and item in delivered):
                return False, km
            mv = self.move(vehicle, loc, t, load, picked - delivered, stored, kind, item)
            if mv is None:
                return False, km
            loc, t, load, d, stored = mv
            km += d
            if kind == "P":
                picked = picked | {item}
            else:
                delivered = delivered | {item}
        if picked != delivered:
            return False, km
        if route:
            km += self.travel(loc, vehicle.depot_loc)[0]
        return True, km

    def route_feasibility_checker(self, route, vehicle):
        return self._simulate(route, vehicle)[0]

    def cost_function(self, route, vehicle):
        ok, km = self._simulate(route, vehicle)
        return km if ok else math.inf

    def solution_cost(self, routes):
        return sum(self.cost_function(r, self.veh_by_id[v]) for v, r in routes.items())

    def is_complete(self, routes):
        done = sorted(i for r in routes.values() for k, i in r if k == "D")
        return done == sorted(self.req_by_id)
