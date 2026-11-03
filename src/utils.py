"""Funções auxiliares para APRESENTAR e VERIFICAR soluções (não fazem parte do modelo)."""
import pandas as pd
import matplotlib.pyplot as plt


def detalhar_rota(p, vid, route):
    """Mostra uma rota perna a perna, incluindo as passagens implícitas pelo armazém."""
    v = p.veh_by_id[vid]
    loc, onboard, armazenado, linhas = v.depot_loc, set(), set(), []

    def perna(de, para, nota):
        dk, dm = p.travel(de, para)
        linhas.append({"de": de, "para": para, "km": dk, "min": round(dm), "nota": nota})

    for kind, i in route:
        r = p.req_by_id[i]
        if kind == "P":
            perna(loc, r.pickup_loc, f"recolher pedido {i}")
            onboard.add(i)
            loc = r.pickup_loc
        else:
            if i not in armazenado:
                perna(loc, v.depot_loc, "passagem obrigatória pelo armazém")
                loc = v.depot_loc
                armazenado |= onboard
            perna(loc, r.delivery_loc, f"entregar pedido {i}")
            onboard.discard(i)
            armazenado.discard(i)
            loc = r.delivery_loc
    perna(loc, v.depot_loc, "regresso ao centro")
    return pd.DataFrame(linhas)


def sequencia_locais(p, vid, route):
    """Locais visitados por um veículo, incluindo as passagens implícitas pelo armazém."""
    v = p.veh_by_id[vid]
    seq, onboard, arm = [v.depot_loc], set(), set()
    for kind, i in route:
        r = p.req_by_id[i]
        if kind == "P":
            seq.append(r.pickup_loc)
            onboard.add(i)
        else:
            if i not in arm:
                seq.append(v.depot_loc)
                arm |= onboard
            seq.append(r.delivery_loc)
            onboard.discard(i)
            arm.discard(i)
    seq.append(v.depot_loc)
    return seq


def mostrar_rotas(p, routes):
    """Tabela com o horário de cada veículo, paragem a paragem."""
    rows = []
    for vid, route in routes.items():
        v = p.veh_by_id[vid]
        loc, t, carga, onboard, arm = v.depot_loc, 0.0, 0.0, set(), set()

        def linha(rotulo, local, chegada, servico):
            rows.append({"veículo": vid, "paragem": rotulo, "local": local,
                         "chegada": (p.t0 + pd.Timedelta(minutes=chegada)).round("min"),
                         "serviço": (p.t0 + pd.Timedelta(minutes=servico)).round("min"),
                         "espera (min)": round(servico - chegada),
                         "carga (kg)": round(max(carga, 0.0), 1)})

        for kind, i in route:
            r = p.req_by_id[i]
            if kind == "P":
                t += p.travel(loc, r.pickup_loc)[1]
                cheg = t
                t = max(t, r.availability_window[0])
                carga += r.weight
                onboard.add(i)
                linha(f"P{i}", r.pickup_loc, cheg, t)
                loc = r.pickup_loc
            else:
                if i not in arm:
                    t += p.travel(loc, v.depot_loc)[1]
                    loc = v.depot_loc
                    arm |= onboard
                    linha("armazém", loc, t, t)
                t += p.travel(loc, r.delivery_loc)[1]
                cheg = t
                t = max(t, r.delivery_window[0])
                carga -= r.weight
                onboard.discard(i)
                arm.discard(i)
                linha(f"D{i}", r.delivery_loc, cheg, t)
                loc = r.delivery_loc
    return pd.DataFrame(rows)


def desenhar_rotas(p, routes, titulo):
    """Desenha no mapa a sequência de locais visitada por cada veículo."""
    L = p.locations.set_index("location_id")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(L.longitude, L.latitude, c="lightgray", edgecolor="k", zorder=3)
    for lid, row in L.iterrows():
        ax.annotate(str(lid), (row.longitude, row.latitude), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    cores = ["tab:red", "tab:blue", "tab:green"]
    for k, (vid, route) in enumerate(routes.items()):
        if not route:
            continue
        seq = sequencia_locais(p, vid, route)
        ax.plot([L.loc[s, "longitude"] for s in seq], [L.loc[s, "latitude"] for s in seq],
                color=cores[k % 3], alpha=.5, lw=1.5, label=f"veículo {vid}", zorder=2)
    ax.scatter(L.loc[p.depot, "longitude"], L.loc[p.depot, "latitude"], marker="*",
               s=300, c="gold", edgecolor="k", zorder=4, label="armazém")
    ax.set_title(titulo)
    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.show()


def resumo_rotas(p, routes):
    """km e nº de pedidos por veículo."""
    return pd.DataFrame([{"veículo": vid,
                          "pedidos": sum(1 for k, _ in rt if k == "P"),
                          "km": round(p.cost_function(rt, p.veh_by_id[vid]), 1)}
                         for vid, rt in routes.items()])


def auditar(p, routes):
    """Verificação independente (não usa Problem.move). Devolve (tabela, lista de erros)."""
    erros, linhas, entregues, dep = [], [], set(), p.depot
    for vid, rt in routes.items():
        v = p.veh_by_id[vid]
        loc, t, km, carga, pico, espera, idas = v.depot_loc, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        onb, arm = set(), set()
        for k, i in rt:
            r = p.req_by_id[i]
            if k == "P":
                dk, dm = p.travel(loc, r.pickup_loc)
                km += dk
                t += dm
                a0, a1 = r.availability_window
                if t < a0:
                    espera += a0 - t
                    t = a0
                if t > a1 + 1e-9:
                    erros.append(f"recolha do pedido {i} fora da janela")
                carga += r.weight
                onb.add(i)
                loc = r.pickup_loc
                pico = max(pico, carga)
                if carga > v.capacity:
                    erros.append(f"capacidade excedida no pedido {i}")
            else:
                if i not in onb:
                    erros.append(f"pedido {i} entregue sem recolha")
                if i not in arm:                       # tem de passar pelo armazém
                    dk, dm = p.travel(loc, dep)
                    km += dk
                    t += dm
                    loc = dep
                    arm |= onb
                    idas += 1
                dk, dm = p.travel(loc, r.delivery_loc)
                km += dk
                t += dm
                b0, b1 = r.delivery_window
                if t < b0:
                    espera += b0 - t
                    t = b0
                if t > b1 + 1e-9:
                    erros.append(f"entrega do pedido {i} fora da janela")
                carga -= r.weight
                onb.discard(i)
                arm.discard(i)
                entregues.add(i)
                loc = r.delivery_loc
        if rt:
            dk, dm = p.travel(loc, dep)
            km += dk
            t += dm
        if onb:
            erros.append(f"veículo {vid}: pedidos {sorted(onb)} nunca entregues")
        if rt and abs(km - p.cost_function(rt, v)) > 1e-6:
            erros.append(f"veículo {vid}: custo diverge do verificador")
        linhas.append({"veículo": vid, "pedidos": len(rt) // 2, "km": round(km, 1),
                       "pico de carga (kg)": round(pico, 1), "capacidade (kg)": v.capacity,
                       "idas forçadas ao armazém": idas,
                       "espera total (dias)": round(espera / 1440, 1),
                       "regressa ao centro (dia)": round(t / 1440, 1)})
    if entregues != set(p.req_by_id):
        erros.append(f"pedidos por entregar: {sorted(set(p.req_by_id) - entregues)}")
    return pd.DataFrame(linhas), erros
