"""Modelação do problema: entidades, custo e verificação de viabilidade."""
import pandas as pd
import math


class Vehicle:
    def __init__(self, vehicle_id, capacity_kg, speed_kmh, depot_loc, refrigerated=False):
        self.vehicle_id = vehicle_id
        self.capacity = capacity_kg
        self.speed = speed_kmh
        self.depot_loc = depot_loc
        self.refrigerated = refrigerated


class Request:
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
    def __init__(self, vi, loc, time, load, picked, delivered, stored, cost,
                 parent=None, action=None):
        self.vi, self.loc, self.time, self.load = vi, loc, time, load
        self.picked, self.delivered, self.stored = picked, delivered, stored
        self.cost, self.parent, self.action = cost, parent, action


# --- helpers para o novo tipo de `stored` ---------------------------------
def _is_stored(stored, item):
    """True se `item` já passou pelo armazém (independente da representação)."""
    return any(x == item for x, _ in stored)


def _get_arrival(stored, item):
    """Instante de chegada ao armazém de `item`, ou None."""
    for x, at in stored:
        if x == item:
            return at
    return None


class Problem:
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

        # --- NOVO: carregar as regras do dataset ---
        rules_df = pd.read_csv(f"{p}rules.csv")
        rules = dict(zip(rules_df.rule, rules_df.value))
        self.stop_service_min = float(rules["stop_service_min"])
        self.warehouse_handling_min = float(rules["warehouse_handling_min"])
        self.max_warehouse_dwell_min = float(rules["max_warehouse_dwell_h"]) * 60.0

        self.dist = {(r.origin_location_id, r.destination_location_id):
                     (r.distance_km, r.duration_min) for r in dm.itertuples()}
        self.loc_id = {(r.location_type, r.ref_id): r.location_id
                       for r in self.locations.itertuples()}

        self.vehicles = self._load_vehicles()
        self.requests = self._load_requests(exclude_status)
        self.req_by_id = {r.item_id: r for r in self.requests}
        self.veh_by_id = {v.vehicle_id: v for v in self.vehicles}
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
        (loc, t, carga, km, armazenados) ou None se violar alguma restrição.

        `stored` é um frozenset de (item_id, arrival_time_em_minutos).
        """
        r = self.req_by_id[item]

        # ---------- RECOLHA ----------
        if kind == "P":
            d, m = self.travel(loc, r.pickup_loc)
            t_service = max(t + m, r.availability_window[0])   # início do serviço
            if t_service > r.availability_window[1]:
                return None
            if load + r.weight > vehicle.capacity:
                return None
            # + tempo de serviço no produtor
            return r.pickup_loc, t_service + self.stop_service_min, load + r.weight, d, stored

        # ---------- ENTREGA ----------
        if item not in onboard:
            return None

        km = 0.0
        if self.via_depot and not _is_stored(stored, item):
            # 1) ir ao armazém
            d1, m1 = self.travel(loc, vehicle.depot_loc)
            km, loc = d1, vehicle.depot_loc
            t_arrival = t + m1                                    # chegada ao armazém
            t = t_arrival + self.warehouse_handling_min           # + handling
            # guardar apenas os itens que ainda não estavam armazenados
            new_items = [(x, t_arrival) for x in onboard if not _is_stored(stored, x)]
            stored = frozenset(list(stored) + new_items)

        # 2) verificar dwell no armazém (arrival → início do loading)
        arrival = _get_arrival(stored, item)
        if arrival is not None and (t - arrival) > self.max_warehouse_dwell_min + 1e-9:
            return None

        # 3) entregar
        d, m = self.travel(loc, r.delivery_loc)
        t_service = max(t + m, r.delivery_window[0])              # início do serviço
        if t_service > r.delivery_window[1]:
            return None
        stored = frozenset((x, at) for x, at in stored if x != item)
        return r.delivery_loc, t_service + self.stop_service_min, \
               load - r.weight, km + d, stored

    def _simulate(self, route, vehicle):
        loc, t, load, km = vehicle.depot_loc, 0.0, 0.0, 0.0
        picked, delivered = frozenset(), frozenset()
        stored = frozenset()                                       # (item, arrival)
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