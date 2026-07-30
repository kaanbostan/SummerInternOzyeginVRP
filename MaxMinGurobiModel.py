"""
=============================================================================
COK TURLU ARAC ROTALAMA PROBLEMI (Multi-Trip VRP) - IS YUKU DENGELEME
MAX-MIN YAKLASIMI
=============================================================================
Bu script, projede tanimlanan matematik modeli (Set 1-3, Parametreler,
Karar Degiskenleri, Amac Fonksiyonu ve 26 kisit) birebir Gurobi/Python
(gurobipy) kodu olarak uygular.

Model ozeti:
    - 1 depo, N musteri, K homojen arac, her arac en fazla R_max tur yapabilir
    - Amac: toplam tasima maliyetini + is yuku dengesizligini (D_max, Q_max,
      C_max) minimize etmek
    - Zaman penceresi, kapasite, tur suresi ve calisma suresi kisitlari var
    - Alt tur (subtour) engelleme MTZ ile yapiliyor

VERI AKISI (ONEMLI):
    1) generate_instance_json(...)  -> rastgele bir instance uretir ve
       bir JSON dosyasina (varsayilan: instance.json) yazar. seed=None
       verilirse her calistirmada FARKLI rastgele veri uretilir.
    2) load_instance_from_json(...) -> o JSON dosyasini okuyup modelin
       kullanacagi 'data' sozlugunu olusturur.
    3) build_model(data, ...)       -> data'yi kullanarak Gurobi modelini kurar.

    TUM PARAMETRE DEGISIKLIKLERI (n_customers, n_vehicles, r_max, Q,
    T_max_trip, T_max_work, alpha, beta, theta, seed) en alttaki
    `if __name__ == "__main__":` blogundaki generate_instance_json(...)
    CAGRISINDAN yapilir. Bu dosyada baska bir yerde degistirmeniz gerekmez.

Calistirmak icin:
    pip install gurobipy
    python multi_trip_vrp.py

=============================================================================
"""
import time
import json
import math
import random
import gurobipy as gp
from gurobipy import GRB

# =============================================================================
# 0) VERI / INSTANCE URETIMI  (kendi verinizle degistirin)
# =============================================================================

def _finalize_instance(N, V, K, R, n, m, r_max, coords, d, s, a, b, Q,
                        T_max_trip, T_max_work, alpha, beta, theta):
    """
    coords/d/s/a/b gibi ham veriler hazir oldugunda, c/t mesafe matrislerini
    hesaplayip tam "data" sozlugunu olusturan ortak yardimci fonksiyon.
    load_instance_from_json tarafindan kullanilir (kod tekrarini onlemek icin).
    """
    c = {}
    t = {}
    for i in V:
        for j in V:
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist = math.hypot(dx, dy)
                c[i, j] = dist
                t[i, j] = dist

    return dict(N=N, V=V, K=K, R=R, n=n, m=m, r_max=r_max,
                c=c, t=t, d=d, s=s, a=a, b=b, Q=Q,
                T_max_trip=T_max_trip, T_max_work=T_max_work,
                alpha=alpha, beta=beta, theta=theta, coords=coords)


# =============================================================================
# 0b) JSON TABANLI VERI URETIMI / OKUMA
# =============================================================================

def generate_instance_json(json_path="instance.json", n_customers=9,
                            n_vehicles=3, r_max=2, Q=20, seed=None,
                            T_max_trip=150.0, T_max_work=400.0,
                            alpha=1.0, beta=1.0, theta=1.0):
    """
    Rastgele bir instance uretir ve JSON dosyasina yazar.

    seed=None verilirse (varsayilan), her cagirildiginda gercekten farkli
    / tekrarlanamaz bir rastgele instance uretilir (sistem entropisinden
    beslenen bir seed secilir). Ayni instance'i daha sonra tekrar uretmek
    isterseniz donen seed degerini not edip seed=<o_deger> olarak
    verebilirsiniz.

    Donus: kullanilan seed (int)
    """
    if seed is None:
        seed = random.SystemRandom().randint(0, 2_000_000_000)

    rnd = random.Random(seed)

    N = list(range(1, n_customers + 1))

    coords = {0: (20.0, 20.0)}
    for i in N:
        coords[i] = (round(rnd.uniform(0, 40), 3), round(rnd.uniform(0, 40), 3))

    d = {i: rnd.randint(5, 20) for i in N}
    s = {i: round(rnd.uniform(3, 8), 3) for i in N}

    horizon = 400.0
    a, b = {}, {}
    for i in N:
        start = round(rnd.uniform(0, horizon * 0.3), 3)
        a[i] = start
        b[i] = round(start + rnd.uniform(150, 250), 3)

    json_data = {
        "seed": seed,
        "n_customers": n_customers,
        "n_vehicles": n_vehicles,
        "r_max": r_max,
        "Q": Q,
        "T_max_trip": T_max_trip,
        "T_max_work": T_max_work,
        "alpha": alpha,
        "beta": beta,
        "theta": theta,
        # JSON'da dict key'leri string olmak zorunda -> str(id) kullanildi
        "depot_xy": list(coords[0]),
        "customers": {
            str(i): {
                "x": coords[i][0],
                "y": coords[i][1],
                "demand": d[i],
                "service_time": s[i],
                "ready_time": a[i],
                "due_time": b[i],
            }
            for i in N
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"[Bilgi] Rastgele instance uretildi ve kaydedildi: {json_path} "
          f"(seed={seed})")
    return seed


def load_instance_from_json(json_path):
    """
    generate_instance_json (veya ayni formatta elle hazirlanmis bir dosya)
    tarafindan yazilmis JSON'u okuyup build_model'in bekledigi 'data'
    sozlugunu olusturur.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    depot_xy = tuple(j["depot_xy"])
    coords = {0: depot_xy}
    d, s, a, b = {}, {}, {}, {}

    # JSON key'leri string oldugu icin int'e geri ceviriyoruz
    N = sorted(int(k) for k in j["customers"].keys())
    for i in N:
        row = j["customers"][str(i)]
        coords[i] = (row["x"], row["y"])
        d[i] = row["demand"]
        s[i] = row["service_time"]
        a[i] = row["ready_time"]
        b[i] = row["due_time"]

    n_vehicles = j["n_vehicles"]
    r_max = j["r_max"]
    n = len(N)
    V = [0] + N
    K = list(range(1, n_vehicles + 1))
    R = list(range(1, r_max + 1))

    return _finalize_instance(
        N, V, K, R, n, n_vehicles, r_max, coords, d, s, a, b,
        Q=j["Q"], T_max_trip=j["T_max_trip"], T_max_work=j["T_max_work"],
        alpha=j["alpha"], beta=j["beta"], theta=j["theta"])


# =============================================================================
# 1) MODEL KURULUMU
# =============================================================================

def build_model(data, time_limit=120, mip_gap=0.02):
    N, V, K, R = data["N"], data["V"], data["K"], data["R"]
    c, t, d, s, a, b = data["c"], data["t"], data["d"], data["s"], data["a"], data["b"]
    Q = data["Q"]
    T_max_trip = data["T_max_trip"]
    T_max_work = data["T_max_work"]
    alpha, beta, theta = data["alpha"], data["beta"], data["theta"]
    n = data["n"]

    # Buyuk-M degerleri (constraint bazinda daha siki secildi -> daha hizli cozum)
    M_time = T_max_work + max(t.values()) + max(s.values()) + max(b.values()) + 10
    M_gen = 1e5

    model = gp.Model("MultiTrip_VRP_WorkloadBalance")

    # ------------------------------------------------------------------
    # 2) KARAR DEGISKENLERI  (4.3)
    # ------------------------------------------------------------------
    arcs = gp.tuplelist([(i, j) for i in V for j in V if i != j])
    x = model.addVars(arcs, K, R, vtype=GRB.BINARY, name="x")     # x_ijkr

    y = model.addVars(N, K, R, vtype=GRB.BINARY, name="y")       # y_ikr
    z = model.addVars(K, R, vtype=GRB.BINARY, name="z")          # z_kr

    Tserv = model.addVars(N, K, R, lb=0, name="T")               # T_ikr
    Tdep = model.addVars(K, R, lb=0, name="Tdep")                # T_kr^dep
    Tret = model.addVars(K, R, lb=0, name="Tret")                # T_kr^ret
    L = model.addVars(K, R, lb=0, name="L")                      # L_kr

    D = model.addVars(K, lb=0, name="D")                         # D_k
    Qk = model.addVars(K, lb=0, name="Qk")                       # Q_k
    Ck = model.addVars(K, lb=0, name="Ck")                       # C_k

    Dmax = model.addVar(lb=0, name="Dmax")                       # D^max
    Qmax = model.addVar(lb=0, name="Qmax")                       # Q^max
    Cmax = model.addVar(lb=0, name="Cmax")                       # C^max

    u = model.addVars(N, K, R, lb=1, ub=n, name="u")             # MTZ sira degiskeni

    model.update()

    # ------------------------------------------------------------------
    # 3) AMAC FONKSIYONU
    # ------------------------------------------------------------------
    travel_cost = gp.quicksum(c[i, j] * x[i, j, k, r]
                               for k in K for r in R
                               for i in V for j in V if i != j)
    model.setObjective(travel_cost + alpha * Dmax + beta * Qmax + theta * Cmax,
                        GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # 4) KISITLAR
    # ------------------------------------------------------------------

    model.addConstrs(
        (gp.quicksum(y[i, k, r] for k in K for r in R) == 1 for i in N),
        name="C1_customer_assignment")

    model.addConstrs(
        (gp.quicksum(d[i] * y[i, k, r] for i in N) <= Q * z[k, r]
         for k in K for r in R), name="C2_capacity")

    model.addConstrs(
        (gp.quicksum(x[0, j, k, r] for j in N) == z[k, r]
         for k in K for r in R), name="C3_depot_departure")

    model.addConstrs(
        (gp.quicksum(x[i, 0, k, r] for i in N) == z[k, r]
         for k in K for r in R), name="C4_depot_return")

    model.addConstrs(
        (gp.quicksum(x[j, i, k, r] for j in V if j != i) ==
         gp.quicksum(x[i, j, k, r] for j in V if j != i)
         for i in N for k in K for r in R), name="C5_flow_conservation")

    model.addConstrs(
        (gp.quicksum(x[i, j, k, r] for j in V if j != i) == y[i, k, r]
         for i in N for k in K for r in R), name="C6_outgoing_link")

    model.addConstrs(
        (gp.quicksum(x[j, i, k, r] for j in V if j != i) == y[i, k, r]
         for i in N for k in K for r in R), name="C7_incoming_link")

    model.addConstrs(
        (x[i, j, k, r] <= z[k, r]
         for i in V for j in V if i != j for k in K for r in R),
        name="C8_trip_activation")

    model.addConstrs(
        (z[k, r] <= z[k, r - 1] for k in K for r in R if r >= 2),
        name="C9_trip_sequence")

    model.addConstrs(
        (Tserv[i, k, r] >= a[i] - M_time * (1 - y[i, k, r])
         for i in N for k in K for r in R), name="C10_tw_lb")
    model.addConstrs(
        (Tserv[i, k, r] <= b[i] + M_time * (1 - y[i, k, r])
         for i in N for k in K for r in R), name="C10_tw_ub")

    model.addConstrs(
        (Tserv[i, k, r] <= M_time * y[i, k, r]
         for i in N for k in K for r in R), name="C10b_T_activation")

    model.addConstrs(
        (Tserv[j, k, r] >= Tserv[i, k, r] + s[i] + t[i, j]
         - M_time * (1 - x[i, j, k, r])
         for i in N for j in N if i != j for k in K for r in R),
        name="C11_time_propagation")

    model.addConstrs(
        (Tserv[j, k, r] >= Tdep[k, r] + t[0, j] - M_time * (1 - x[0, j, k, r])
         for j in N for k in K for r in R), name="C12_depot_departure_time")

    model.addConstrs(
        (Tret[k, r] >= Tserv[i, k, r] + s[i] + t[i, 0]
         - M_time * (1 - x[i, 0, k, r])
         for i in N for k in K for r in R), name="C13_depot_return_time")

    model.addConstrs(
        (L[k, r] == Tret[k, r] - Tdep[k, r] for k in K for r in R),
        name="C14_trip_duration_def")

    model.addConstrs(
        (Tdep[k, r + 1] >= Tret[k, r] - M_time * (1 - z[k, r + 1])
         for k in K for r in R if r <= len(R) - 1),
        name="C14b_between_trips_order")

    model.addConstrs(
        (L[k, r] <= T_max_trip for k in K for r in R),
        name="C15_max_trip_duration")

    model.addConstrs(
        (gp.quicksum(L[k, r] for r in R) <= T_max_work for k in K),
        name="C16_max_work_time")

    model.addConstrs(
        (D[k] == gp.quicksum(c[i, j] * x[i, j, k, r]
                              for r in R for i in V for j in V if i != j)
         for k in K), name="C17_distance_calc")

    model.addConstrs((D[k] <= Dmax for k in K), name="C18_max_distance")

    model.addConstrs(
        (Qk[k] == gp.quicksum(d[i] * y[i, k, r] for r in R for i in N)
         for k in K), name="C19_demand_calc")

    model.addConstrs((Qk[k] <= Qmax for k in K), name="C20_max_demand")

    model.addConstrs(
        (Ck[k] == gp.quicksum(y[i, k, r] for r in R for i in N)
         for k in K), name="C21_customer_count_calc")

    model.addConstrs((Ck[k] <= Cmax for k in K), name="C22_max_customers")

    model.addConstrs(
        (u[i, k, r] - u[j, k, r] + n * x[i, j, k, r] <= n - 1
         for i in N for j in N if i != j for k in K for r in R),
        name="C23_MTZ_subtour")

    model.addConstrs(
        (Tdep[k, r] <= M_gen * z[k, r] for k in K for r in R),
        name="C24_dep_time_activation")

    model.addConstrs(
        (Tret[k, r] <= M_gen * z[k, r] for k in K for r in R),
        name="C25_ret_time_activation")

    model.addConstrs(
        (L[k, r] <= M_gen * z[k, r] for k in K for r in R),
        name="C26_duration_activation")

    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 1

    variables = dict(x=x, y=y, z=z, Tserv=Tserv, Tdep=Tdep, Tret=Tret, L=L,
                      D=D, Qk=Qk, Ck=Ck, Dmax=Dmax, Qmax=Qmax, Cmax=Cmax, u=u)
    return model, variables


# =============================================================================
# 2) COZUM SONUCLARINI YAZDIRMA
# =============================================================================

def extract_route(data, variables, k, r):
    V, c = data["V"], data["c"]
    x = variables["x"]

    route = [0]
    current = 0
    visited = set()
    while True:
        nxt = None
        for j in V:
            if j != current and (current, j, k, r) in x and x[current, j, k, r].X > 0.5:
                nxt = j
                break
        if nxt is None or nxt == 0:
            route.append(0)
            break
        if nxt in visited:
            print(f"  [UYARI] Arac {k} Tur {r}: dongu tespit edildi, "
                  f"rota erken sonlandirildi.")
            break
        visited.add(nxt)
        route.append(nxt)
        current = nxt

    distance = sum(c[route[i], route[i + 1]] for i in range(len(route) - 1))
    return route, distance


def extract_route_timeline(data, variables, k, r, route=None):
    t, s, d, a, b = data["t"], data["s"], data["d"], data["a"], data["b"]
    Tserv, Tdep = variables["Tserv"], variables["Tdep"]

    if route is None:
        route, _ = extract_route(data, variables, k, r)

    timeline = []
    prev_node = 0
    prev_finish = Tdep[k, r].X
    for node in route[1:-1]:
        arrival = prev_finish + t[prev_node, node]
        service_start = Tserv[node, k, r].X
        wait = max(0.0, service_start - arrival)
        service_end = service_start + s[node]
        timeline.append(dict(node=node, arrival=arrival, wait=wait,
                              service_start=service_start,
                              service_end=service_end,
                              demand=d[node], tw_start=a[node], tw_end=b[node]))
        prev_node = node
        prev_finish = service_end

    return timeline


def print_solution(model, data, variables):
    if model.SolCount == 0:
        print("Fizibil cozum bulunamadi.")
        return

    K, R, N = data["K"], data["R"], data["N"]
    y, z = variables["y"], variables["z"]
    D, Qk, Ck = variables["D"], variables["Qk"], variables["Ck"]
    Tdep, Tret = variables["Tdep"], variables["Tret"]

    print("\n" + "=" * 60)
    print(f"Amac fonksiyonu (Obj) degeri : {model.ObjVal:.2f}")
    print(f"Dmax = {variables['Dmax'].X:.2f} | "
          f"Qmax = {variables['Qmax'].X:.2f} | "
          f"Cmax = {variables['Cmax'].X:.2f}")
    print("=" * 60)

    for k in K:
        for r in R:
            if z[k, r].X > 0.5:
                route, distance = extract_route(data, variables, k, r)
                visited_customers = [i for i in N if y[i, k, r].X > 0.5]
                timeline = extract_route_timeline(data, variables, k, r, route)

                print(f"\nArac {k} - Tur {r}:")
                print(f"  Rota            : {' -> '.join(map(str, route))}")
                print(f"  Musteriler      : {visited_customers}")
                print(f"  Tur suresi (L)  : {variables['L'][k, r].X:.2f}")
                print(f"  Rota mesafesi   : {distance:.2f}")
                print(f"  Depodan cikis   : {Tdep[k, r].X:.2f}")

                print(f"  {'Musteri':>8} | {'Varis':>8} | {'Bekleme':>8} | "
                      f"{'Servis Bas':>10} | {'Servis Bit':>10} | "
                      f"{'Talep':>6} | {'Zaman Penceresi':>16}")
                for stop in timeline:
                    tw = f"[{stop['tw_start']:.0f},{stop['tw_end']:.0f}]"
                    print(f"  {stop['node']:>8} | {stop['arrival']:>8.2f} | "
                          f"{stop['wait']:>8.2f} | {stop['service_start']:>10.2f} | "
                          f"{stop['service_end']:>10.2f} | {stop['demand']:>6.0f} | "
                          f"{tw:>16}")

                print(f"  Depoya donus    : {Tret[k, r].X:.2f}")

    print("\n--- Arac bazli ozet ---")
    for k in K:
        print(f"Arac {k}: Mesafe={D[k].X:.2f}  Talep={Qk[k].X:.2f}  "
              f"MusteriSayisi={Ck[k].X:.0f}")


# =============================================================================
# 3) GORSELLESTIRME
# =============================================================================

def plot_solution(model, data, variables, save_path="route_plot.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if model.SolCount == 0:
        print("Fizibil cozum bulunamadi, gorsellestirme yapilamiyor.")
        return None

    coords, N, K, R = data["coords"], data["N"], data["K"], data["R"]
    z = variables["z"]

    fig, ax = plt.subplots(figsize=(9, 8))

    for i in N:
        xi, yi = coords[i]
        ax.scatter(xi, yi, c="#333333", s=60, zorder=3)
        ax.annotate(str(i), (xi, yi), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    dx, dy = coords[0]
    ax.scatter(dx, dy, c="red", marker="s", s=180, zorder=4, label="Depo")
    ax.annotate("Depo", (dx, dy), textcoords="offset points",
                xytext=(8, -12), fontsize=10, fontweight="bold")

    cmap = plt.get_cmap("tab10")
    linestyles = ["-", "--", ":", "-."]

    for k in K:
        color = cmap((k - 1) % 10)
        for r in R:
            if z[k, r].X > 0.5:
                route, distance = extract_route(data, variables, k, r)
                xs = [coords[node][0] for node in route]
                ys = [coords[node][1] for node in route]
                ls = linestyles[(r - 1) % len(linestyles)]
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2,
                         marker="o", markersize=4, zorder=2,
                         label=f"Arac {k} - Tur {r} (mesafe={distance:.1f})")

    ax.set_title(f"Cok Turlu Arac Rotalama - Cozum (Obj={model.ObjVal:.2f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n[Bilgi] Rota haritasi kaydedildi: {save_path}")
    return save_path


# =============================================================================
# 4) CALISTIRMA (MAIN)
# =============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # ADIM 1: Her calistirmada YENI rastgele bir instance uret ve JSON'a yaz.
    # seed=None -> her calistirmada gercekten farkli veri.
    # Ayni veriyi tekrar kullanmak isterseniz, ekrana basilan seed degerini
    # kaydedip seed=<o_deger> olarak asagida sabitleyebilirsiniz.
    # ------------------------------------------------------------------
    used_seed = generate_instance_json(
        json_path="instance.json",
        n_customers=20,
        n_vehicles=3,
        r_max=3,
        Q=30,
        seed=None,
        T_max_trip=150,
        T_max_work=400,
        alpha=1,
        beta=2,
        theta=3,
    )

    # ------------------------------------------------------------------
    # ADIM 2: Modeli artik JSON dosyasindan oku.
    # ------------------------------------------------------------------
    data = load_instance_from_json("instance.json")

    # ~2000 degisken / ~2000 kisit sinirina dikkat (ucretsiz gurobipy lisansi)
    model, variables = build_model(data, time_limit=3000, mip_gap=0.02)
    model.update()
    print(f"[Bilgi] Degisken sayisi={model.NumVars}, "
          f"Kisit sayisi={model.NumConstrs} (limit: <2000 / <2000)")

    t0 = time.time()
    model.optimize()
    t1 = time.time()
    print(f"\n[CPU] Exact model suresi (build+solve): {t1 - t0:.3f} saniye")
    print(f"[CPU] Gurobi'nin kendi raporladigi solve suresi: {model.Runtime:.3f} saniye")

    print_solution(model, data, variables)
    plot_solution(model, data, variables, save_path="route_plot.png")
    
    
    print(f"\nInstance Seed = {used_seed}")
    
    
