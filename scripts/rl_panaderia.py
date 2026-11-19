import matplotlib; matplotlib.use("Agg")
"""Semana 14 - Reinforcement Learning: reposición de un producto perecedero (Q-Learning, SARSA, Value Iteration).
Script exportado del notebook notebooks/Semana14_RL_Brayan_Vega.ipynb. Ejecutar: python scripts/rl_panaderia.py"""

# ======================================================================
# imports
# ======================================================================
# Librerías base
import os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import warnings
warnings.filterwarnings("ignore")

# Opciones de visualización
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

# Reproducibilidad: una única semilla base para todo el cuaderno
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Carpetas de salida: el cuaderno vive en /notebooks; figuras y datos en la raíz del repositorio
BASE = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
FIG_DIR = os.path.join(BASE, "figures")
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Paleta fija para identificar los algoritmos en todas las gráficas
COLORES = ["teal", "salmon", "#8064A2", "goldenrod", "skyblue", "seagreen", "tomato", "slategray"]

# Presupuesto de cómputo. Con RAPIDO = True el cuaderno corre en ~1 minuto (resultados menos precisos).
RAPIDO = False
EPISODIOS         = 20_000 if RAPIDO else 100_000   # episodios de entrenamiento (comparación principal)
SEMILLAS          = [0, 1] if RAPIDO else [0, 1, 2, 3, 4]
EPISODIOS_BARRIDO = 10_000 if RAPIDO else 50_000    # episodios por configuración en los barridos
SEMILLAS_BARRIDO  = [0, 1] if RAPIDO else [0, 1, 2]
print(print(f"RAPIDO = {RAPIDO} | episodios = {EPISODIOS:,} | semillas = {SEMILLAS}"))

# ======================================================================
# funciones
# ======================================================================
def guardar_figura(nombre):
    """Guarda la figura activa en /figures (PNG, 200 dpi) para usarla en el informe."""
    plt.savefig(os.path.join(FIG_DIR, nombre + ".png"), dpi=200, bbox_inches="tight")


def pct_optimo(valor, optimo, referencia):
    """Porcentaje de la brecha 'referencia -> óptimo' que cubre un método (0 % = referencia, 100 % = óptimo)."""
    return 100 * (valor - referencia) / (optimo - referencia)

# ======================================================================
# params
# ======================================================================
# Parámetros del problema (supuestos del caso, en miles de pesos colombianos por unidad)
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
DEMANDA_MEDIA = np.array([2.0, 2.0, 2.5, 3.0, 4.0, 5.0, 3.0])   # unidades/día (Poisson)
CAP = 7                                  # máximo de unidades en cada "edad" y por pedido
PRECIO, COSTO, MERMA, FALTANTE = 4.0, 2.5, 0.5, 0.5
VIDA_UTIL = 3                            # días que dura el producto (llega fresco, queda a 1 y a 0 días de vencer)
H = 28                                   # días por episodio (4 semanas)
GAMMA = 0.95                             # factor de descuento
NA = CAP + 1                             # acciones: pedir 0..CAP unidades
N_ESTADOS = 7 * NA * NA                  # (día de la semana) x (unidades que vencen mañana) x (unidades que vencen hoy)

params = pd.DataFrame({
    "Demanda media (unid./día)": DEMANDA_MEDIA,
    "P(demanda > CAP)": poisson.sf(CAP, DEMANDA_MEDIA),
    "Cuantil 90 de la demanda": poisson.ppf(0.90, DEMANDA_MEDIA),
}, index=DIAS)
print("Margen por unidad vendida     :", PRECIO - COSTO)
print("Pérdida por unidad sobrante   :", COSTO + MERMA, "(costo no recuperado + disposición)")
print("Pérdida por unidad faltante   :", FALTANTE, "(+ margen no ganado de", PRECIO - COSTO, ")")
print("Estados:", N_ESTADOS, "| Acciones:", NA, "| Pares estado-acción:", N_ESTADOS * NA)

# Datos de ejemplo para el repositorio: parámetros y un año de demanda simulada
params.to_csv(os.path.join(DATA_DIR, "parametros_demanda.csv"), index_label="dia")
rng_demo = np.random.default_rng(SEED)
dias_365 = np.arange(365) % 7
pd.DataFrame({"dia_semana": [DIAS[i] for i in dias_365],
              "demanda": rng_demo.poisson(DEMANDA_MEDIA[dias_365])}).to_csv(
    os.path.join(DATA_DIR, "demanda_simulada_365d.csv"), index_label="dia_n")
print(params.round(3))

# ======================================================================
# demanda_fig
# ======================================================================
# Demanda semanal: media y rango entre los percentiles 10 y 90
p10 = poisson.ppf(0.10, DEMANDA_MEDIA)
p90 = poisson.ppf(0.90, DEMANDA_MEDIA)

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.bar(DIAS, DEMANDA_MEDIA, color="teal", alpha=0.9, label="Demanda media")
ax.errorbar(DIAS, DEMANDA_MEDIA, yerr=[DEMANDA_MEDIA - p10, p90 - DEMANDA_MEDIA], fmt="none",
            ecolor="black", capsize=4, label="Percentiles 10 a 90")
ax.axhline(CAP, color="salmon", ls="--", label=f"Capacidad (CAP = {CAP})")
ax.set_ylabel("Unidades demandadas por día")
ax.set_title("Panadería · Demanda diaria esperada (Poisson)")
ax.legend()
plt.tight_layout()
guardar_figura("fig1_demanda_semanal")
plt.show()

# ======================================================================
# newsvendor
# ======================================================================
# Referencia de un solo día ("problema del vendedor de periódicos"): ignora que lo no vendido se puede vender mañana
num = (PRECIO - COSTO) + FALTANTE          # costo de quedarse corto (cu)
den = num + (COSTO + MERMA)                # cu + costo de pasarse (co)
RC = num / den                             # razón crítica
q_crit = poisson.ppf(RC, DEMANDA_MEDIA).astype(int)
print(f"Razón crítica = {num:.1f} / {den:.1f} = {RC:.2f}")
print(pd.DataFrame({"Demanda media": DEMANDA_MEDIA, "Inventario objetivo (cuantil crítico)": q_crit}, index=DIAS))

# ======================================================================
# entorno
# ======================================================================
class PanaderiaEnv:
    """Entorno de reposición de un producto perecedero (API tipo Gymnasium, sin dependencias externas).

    Estado    : (día de la semana d, u1, u2) codificado como un entero 0..N_ESTADOS-1.
                u1 = unidades que vencen mañana, u2 = unidades que vencen hoy (se venden primero).
    Acción    : a in {0..CAP}, unidades que se piden al iniciar el día (llegan frescas de inmediato).
    Dinámica  : demanda D ~ Poisson(media del día); se vende primero lo más viejo (FIFO);
                al cerrar, lo que queda de u2 se desecha y el resto envejece un día.
    Recompensa: PRECIO*ventas - COSTO*pedido - MERMA*desechadas - FALTANTE*demanda_no_atendida.
    Episodio  : H días; termina por límite de tiempo (truncated = True), nunca por un estado terminal.
    """
    def __init__(self, seed=None, horizonte=H):
        self.rng = np.random.default_rng(seed)
        self.H = horizonte
        self.n_estados, self.n_acciones = N_ESTADOS, NA

    @staticmethod
    def codificar(d, u1, u2):
        return (d * NA + u1) * NA + u2

    @staticmethod
    def decodificar(s):
        return s // (NA * NA), (s // NA) % NA, s % NA

    def reset(self, dia=None):
        self.d = int(self.rng.integers(7)) if dia is None else dia
        self.u1 = self.u2 = 0                      # se arranca sin inventario
        self.t = 0
        # Demanda de todo el episodio, muestreada de una vez (más rápido que muestrear paso a paso)
        self._dem = self.rng.poisson(DEMANDA_MEDIA[(self.d + np.arange(self.H)) % 7]).tolist()
        return self.codificar(self.d, self.u1, self.u2)

    def step(self, a):
        D = self._dem[self.t]
        v2 = min(self.u2, D); r = D - v2           # se vende primero lo que vence hoy
        v1 = min(self.u1, r); r -= v1              # luego lo que vence mañana
        v0 = min(a, r);       falt = r - v0        # por último lo recién pedido
        merma = self.u2 - v2                       # lo que vence hoy sin vender se desecha
        recompensa = PRECIO * (v2 + v1 + v0) - COSTO * a - MERMA * merma - FALTANTE * falt
        self.u2, self.u1 = self.u1 - v1, a - v0    # el inventario envejece un día
        self.d = (self.d + 1) % 7
        self.t += 1
        info = {"demanda": D, "ventas": v2 + v1 + v0, "faltante": falt, "merma": merma}
        return self.codificar(self.d, self.u1, self.u2), recompensa, False, self.t >= self.H, info


env = PanaderiaEnv(seed=SEED)
print("Estados:", env.n_estados, "| Acciones:", env.n_acciones)
print(print("Estado inicial (lunes, sin inventario) =", env.reset(dia=0), "->", env.decodificar(env.reset(dia=0))))

# ======================================================================
# traza_aleatoria
# ======================================================================
# Prueba del entorno: una semana con una política aleatoria
env = PanaderiaEnv(seed=SEED)
s = env.reset(dia=0)
filas = []
for _ in range(7):
    d, u1, u2 = env.decodificar(s)
    a = int(env.rng.integers(NA))
    s, r, term, trunc, info = env.step(a)
    filas.append({"Día": DIAS[d], "Vencen mañana": u1, "Vencen hoy": u2, "Pedido": a, "Demanda": info["demanda"],
                  "Ventas": info["ventas"], "Faltante": info["faltante"], "Merma": info["merma"], "Recompensa": r})
traza = pd.DataFrame(filas)
print("Ganancia de la semana:", round(traza["Recompensa"].sum(), 2))
print(traza)

# ======================================================================
# modelo_vi
# ======================================================================
# Modelo exacto: P(s'|s,a) y R(s,a) calculados con la distribución de Poisson (la demanda se trunca en DMAX)
DMAX = 30
pmf = [poisson.pmf(np.arange(DMAX + 1), m) for m in DEMANDA_MEDIA]
for p in pmf:
    p[-1] += 1 - p.sum()                         # la cola de la distribución se agrega al último valor

T = np.zeros((N_ESTADOS, NA, N_ESTADOS))         # probabilidades de transición
R = np.zeros((N_ESTADOS, NA))                    # recompensa esperada
for d in range(7):
    for u1 in range(NA):
        for u2 in range(NA):
            s = PanaderiaEnv.codificar(d, u1, u2)
            for a in range(NA):
                for D in range(DMAX + 1):
                    v2 = min(u2, D); r_ = D - v2
                    v1 = min(u1, r_); r_ -= v1
                    v0 = min(a, r_);  falt = r_ - v0
                    rec = PRECIO * (v2 + v1 + v0) - COSTO * a - MERMA * (u2 - v2) - FALTANTE * falt
                    s2 = PanaderiaEnv.codificar((d + 1) % 7, a - v0, u1 - v1)
                    T[s, a, s2] += pmf[d][D]
                    R[s, a] += pmf[d][D] * rec

def value_iteration(gamma=GAMMA, tol=1e-9):
    """Iteración de valor: resuelve las ecuaciones de optimalidad de Bellman con el modelo exacto."""
    V, it = np.zeros(N_ESTADOS), 0
    while True:
        Q = R + gamma * T @ V                    # Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s')
        V_nuevo = Q.max(axis=1)
        it += 1
        if np.abs(V_nuevo - V).max() < tol:
            return Q, it
        V = V_nuevo

Q_opt, iteraciones = value_iteration()
V_opt = Q_opt.max(axis=1)
PI_OPT = Q_opt.argmax(axis=1)
print("Filas de T suman 1:", np.allclose(T.sum(axis=2), 1.0))
print("Iteraciones hasta converger:", iteraciones)
print(print("Pedido óptimo con inventario vacío (u1=u2=0):", dict(zip(DIAS, PI_OPT.reshape(7, NA, NA)[:, 0, 0].tolist()))))

# ======================================================================
# evaluador
# ======================================================================
def evaluar_politica(pi, n_ep=1000, seed=123, horizonte=H):
    """Evalúa una política determinista pi (arreglo de N_ESTADOS acciones) simulando n_ep episodios en paralelo.

    Con la misma `seed` todas las políticas ven exactamente la misma demanda (números aleatorios comunes),
    de modo que las comparaciones son pareadas. Devuelve la ganancia media por día (miles de COP), su
    intervalo de confianza del 95 %, el nivel de servicio (ventas / demanda) y el % de lo pedido que se desecha.
    """
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 7, n_ep)
    u1 = np.zeros(n_ep, dtype=int); u2 = np.zeros(n_ep, dtype=int)
    tot = np.zeros(n_ep); ventas = np.zeros(n_ep); falt_t = np.zeros(n_ep); merma_t = np.zeros(n_ep); pedido_t = np.zeros(n_ep)
    for _ in range(horizonte):
        D = rng.poisson(DEMANDA_MEDIA[d])
        a = pi[(d * NA + u1) * NA + u2]
        v2 = np.minimum(u2, D); r_ = D - v2
        v1 = np.minimum(u1, r_); r_ = r_ - v1
        v0 = np.minimum(a, r_);  falt = r_ - v0
        merma = u2 - v2
        tot += PRECIO * (v2 + v1 + v0) - COSTO * a - MERMA * merma - FALTANTE * falt
        ventas += v2 + v1 + v0; falt_t += falt; merma_t += merma; pedido_t += a
        u2, u1, d = u1 - v1, a - v0, (d + 1) % 7
    return {"Ganancia/día": tot.mean() / horizonte,
            "IC95 ±": 1.96 * tot.std(ddof=1) / np.sqrt(n_ep) / horizonte,
            "Nivel de servicio": ventas.sum() / (ventas.sum() + falt_t.sum()),
            "Merma (% de lo pedido)": 100 * merma_t.sum() / max(pedido_t.sum(), 1)}


def politica_desde_regla(regla):
    """Construye el arreglo de acciones de una política dada por una función regla(d, u1, u2) -> pedido."""
    pi = np.zeros(N_ESTADOS, dtype=int)
    for s in range(N_ESTADOS):
        d, u1, u2 = PanaderiaEnv.decodificar(s)
        pi[s] = int(np.clip(regla(d, u1, u2), 0, CAP))
    return pi

# ======================================================================
# validacion
# ======================================================================
# Validación cruzada: (1) el simulador paso a paso (entorno) y el evaluador vectorizado deben coincidir,
# (2) la recompensa esperada del modelo exacto R debe coincidir con una estimación Monte Carlo.
env_val = PanaderiaEnv(seed=7)
tot_env = []
for _ in range(1000):
    s = env_val.reset(); suma = 0.0; trunc = False
    while not trunc:
        s, r, term, trunc, _ = env_val.step(int(PI_OPT[s])); suma += r
    tot_env.append(suma / H)
vec = evaluar_politica(PI_OPT, n_ep=1000, seed=7)
print(f"Política óptima | entorno paso a paso: {np.mean(tot_env):.3f} ± {1.96*np.std(tot_env)/np.sqrt(1000):.3f}"
      f" | evaluador vectorizado: {vec['Ganancia/día']:.3f} ± {vec['IC95 ±']:.3f}")

rng_v = np.random.default_rng(1)
filas = []
for _ in range(5):
    s = int(rng_v.integers(N_ESTADOS)); a = int(rng_v.integers(NA)); d, u1, u2 = PanaderiaEnv.decodificar(s)
    D = rng_v.poisson(DEMANDA_MEDIA[d], 200_000)
    v2 = np.minimum(u2, D); r_ = D - v2; v1 = np.minimum(u1, r_); r_ -= v1; v0 = np.minimum(a, r_); falt = r_ - v0
    mc = (PRECIO * (v2 + v1 + v0) - COSTO * a - MERMA * (u2 - v2) - FALTANTE * falt).mean()
    filas.append({"Estado": f"({DIAS[d][:3]}, u1={u1}, u2={u2})", "Pedido": a, "R(s,a) exacta": R[s, a], "Monte Carlo": mc})
print(pd.DataFrame(filas).round(3))

# ======================================================================
# baselines
# ======================================================================
# Políticas de referencia (no aprenden). Todas se evalúan con la misma demanda (seed=123).
def regla_s_global(S):
    return lambda d, u1, u2: S - u1 - u2            # pedir hasta completar S unidades en total

rng_pol = np.random.default_rng(SEED)
politicas = {
    "Aleatoria":                        rng_pol.integers(0, NA, N_ESTADOS),
    "Intuitiva (pedir la media del día)": politica_desde_regla(lambda d, u1, u2: round(DEMANDA_MEDIA[d])),
    "Cuantil crítico por día":          politica_desde_regla(lambda d, u1, u2: q_crit[d] - u1 - u2),
}

# Barrido del nivel objetivo S global (la mejor regla "pedir hasta S" que no distingue el día)
barrido_S = pd.DataFrame({S: evaluar_politica(politica_desde_regla(regla_s_global(S))) for S in range(0, CAP + 1)}).T
barrido_S.index.name = "S"
S_MEJOR = int(barrido_S["Ganancia/día"].idxmax())
politicas[f"Mejor S global (S = {S_MEJOR})"] = politica_desde_regla(regla_s_global(S_MEJOR))
print("Mejor nivel objetivo global: S =", S_MEJOR)
print(barrido_S.round(3))

# ======================================================================
# baselines_fig
# ======================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.errorbar(barrido_S.index, barrido_S["Ganancia/día"], yerr=barrido_S["IC95 ±"], fmt="o-", color="teal", capsize=3)
ax.scatter([S_MEJOR], [barrido_S.loc[S_MEJOR, "Ganancia/día"]], s=180, marker="*", color="black", zorder=3,
           label=f"Máximo: S = {S_MEJOR} ({barrido_S.loc[S_MEJOR, 'Ganancia/día']:.2f})")
ax.set_xlabel("Nivel objetivo S (unidades en inventario tras pedir)")
ax.set_ylabel("Ganancia media por día (miles de COP)")
ax.set_title("Regla 'pedir hasta S' · Barrido del nivel objetivo")
ax.legend()
plt.tight_layout()
guardar_figura("fig2_barrido_S")
plt.show()

# ======================================================================
# tabla_ref
# ======================================================================
# Referencias: política óptima (Value Iteration) y políticas fijas
politicas["Óptima (Value Iteration)"] = PI_OPT
tabla_ref = pd.DataFrame({nombre: evaluar_politica(pi) for nombre, pi in politicas.items()}).T
GAN_OPT = tabla_ref.loc["Óptima (Value Iteration)", "Ganancia/día"]
GAN_ALE = tabla_ref.loc["Aleatoria", "Ganancia/día"]
GAN_SGL = tabla_ref.loc[f"Mejor S global (S = {S_MEJOR})", "Ganancia/día"]
print(tabla_ref.round(3))

# ======================================================================
# agentes
# ======================================================================
def entrenar(algo="q", episodios=EPISODIOS, alpha=0.05, gamma=GAMMA, eps_ini=1.0, eps_fin=0.05,
             frac_explor=0.6, seed=0, cada=2000, terminal_al_truncar=False):
    """Aprendizaje por refuerzo tabular con política epsilon-voraz.

    algo = "q"     -> Q-learning (fuera de política): objetivo r + gamma * max_a' Q(s', a')
    algo = "sarsa" -> SARSA (dentro de la política): objetivo r + gamma * Q(s', a'), con a' la acción realmente elegida
    epsilon decae de forma geométrica de eps_ini a eps_fin durante el primer `frac_explor` de los episodios.
    terminal_al_truncar = True trata el fin del episodio (límite de 28 días) como estado terminal (sin arranque);
    por defecto es False, porque el estado no contiene el tiempo restante.
    Devuelve la tabla Q (N_ESTADOS x NA) y la curva de aprendizaje [(episodio, ganancia/día de la política voraz, epsilon)].
    """
    env = PanaderiaEnv(seed)
    rng = random.Random(seed)
    Q = [[0.0] * NA for _ in range(N_ESTADOS)]
    eps = eps_ini
    decay = (eps_fin / eps_ini) ** (1.0 / (frac_explor * episodios))
    curva = []

    def elegir(s):
        if rng.random() < eps:
            return rng.randrange(NA)                       # explorar
        fila = Q[s]; m = max(fila)
        if fila.count(m) > 1:                              # desempate aleatorio entre acciones igual de buenas
            return rng.choice([i for i, v in enumerate(fila) if v == m])
        return fila.index(m)                               # explotar

    for ep in range(1, episodios + 1):
        s = env.reset(); a = elegir(s); trunc = False
        while not trunc:
            s2, r, term, trunc, _ = env.step(a)
            a2 = elegir(s2)
            if algo == "q":
                futuro = max(Q[s2])
            else:
                futuro = Q[s2][a2]
            objetivo = r if (trunc and terminal_al_truncar) else r + gamma * futuro
            Q[s][a] += alpha * (objetivo - Q[s][a])
            s, a = s2, a2
        eps = max(eps_fin, eps * decay)
        if ep % cada == 0:
            curva.append((ep, evaluar_politica(np.argmax(Q, axis=1), n_ep=300, seed=999)["Ganancia/día"], eps))
    return np.array(Q), curva

# ======================================================================
# entrena_principal
# ======================================================================
# Comparación principal: Q-learning y SARSA con el mismo presupuesto y las mismas semillas
resultados = {"q": {}, "sarsa": {}}
t0 = time.time()
for algo in ["q", "sarsa"]:
    for sd in SEMILLAS:
        Q_tab, curva = entrenar(algo, episodios=EPISODIOS, seed=sd)
        resultados[algo][sd] = {"Q": Q_tab, "curva": np.array(curva)}
        print(f"{algo:>5} | semilla {sd} | ganancia final (curva) = {curva[-1][1]:.3f} | {time.time() - t0:5.0f} s acumulados")

np.save(os.path.join(DATA_DIR, "Q_qlearning_semilla0.npy"), resultados["q"][SEMILLAS[0]]["Q"])
print(np.save(os.path.join(DATA_DIR, "Q_optima_value_iteration.npy"), Q_opt))

# ======================================================================
# curvas_fig
# ======================================================================
def banda(ax, algo, color, etiqueta):
    """Dibuja la curva media entre semillas con una banda de ± 1 desviación estándar."""
    x = resultados[algo][SEMILLAS[0]]["curva"][:, 0]
    y = np.array([resultados[algo][sd]["curva"][:, 1] for sd in SEMILLAS])
    ax.plot(x, y.mean(0), color=color, label=etiqueta)
    ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=color, alpha=0.2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, zoom in zip(axes, [False, True]):
    banda(ax, "q", "teal", "Q-learning (media ± desv. entre semillas)")
    banda(ax, "sarsa", "#8064A2", "SARSA (media ± desv. entre semillas)")
    ax.axhline(GAN_OPT, color="black", ls="--", label=f"Óptima (Value Iteration): {GAN_OPT:.2f}")
    ax.axhline(GAN_SGL, color="goldenrod", ls="--", label=f"Mejor S global: {GAN_SGL:.2f}")
    ax.set_xlabel("Episodios de entrenamiento")
    if zoom:
        ax.set_xlim(EPISODIOS * 0.3, EPISODIOS); ax.set_ylim(GAN_SGL - 0.35, GAN_OPT + 0.1)
        ax.set_title("Detalle de la fase final")
    else:
        ax.axhline(tabla_ref.loc["Aleatoria", "Ganancia/día"], color="salmon", ls=":", label="Aleatoria")
        ax.set_title("Curvas de aprendizaje (política voraz evaluada cada 2.000 episodios)")
        ax.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel("Ganancia media por día (miles de COP)")
plt.tight_layout()
guardar_figura("fig3_curvas_aprendizaje")
plt.show()

# ======================================================================
# tabla_final
# ======================================================================
# Evaluación final con una demanda NUEVA (seed=2026, 3.000 episodios) distinta de la usada al entrenar
EVAL = dict(n_ep=3000, seed=2026)
filas = {}
for nombre, pi in politicas.items():
    filas[nombre] = {**evaluar_politica(pi, **EVAL), "Desv. entre semillas": np.nan}

politicas_rl = {}
for algo, nombre in [("q", "Q-learning"), ("sarsa", "SARSA")]:
    evals = [evaluar_politica(np.argmax(resultados[algo][sd]["Q"], axis=1), **EVAL) for sd in SEMILLAS]
    politicas_rl[nombre] = [np.argmax(resultados[algo][sd]["Q"], axis=1) for sd in SEMILLAS]
    prom = pd.DataFrame(evals).mean()
    filas[nombre] = {**prom.to_dict(), "Desv. entre semillas": pd.DataFrame(evals)["Ganancia/día"].std()}

tabla_final = pd.DataFrame(filas).T
g_opt = tabla_final.loc["Óptima (Value Iteration)", "Ganancia/día"]
g_ale = tabla_final.loc["Aleatoria", "Ganancia/día"]
tabla_final["% de la brecha aleatoria→óptima"] = [pct_optimo(g, g_opt, g_ale) for g in tabla_final["Ganancia/día"]]
orden = ["Aleatoria", "Intuitiva (pedir la media del día)", "Cuantil crítico por día", f"Mejor S global (S = {S_MEJOR})",
         "SARSA", "Q-learning", "Óptima (Value Iteration)"]
tabla_final = tabla_final.loc[orden]
tabla_final.to_csv(os.path.join(DATA_DIR, "resultados_finales.csv"))
print(tabla_final.round(3))

# ======================================================================
# pruebas_t
# ======================================================================
# Prueba pareada: ¿la política de Q-learning supera a la mejor regla fija? (misma demanda, 3.000 episodios, por semilla)
def totales_por_episodio(pi, n_ep=3000, seed=2026):
    """Ganancia media por día de cada episodio (para pruebas pareadas)."""
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 7, n_ep); u1 = np.zeros(n_ep, dtype=int); u2 = np.zeros(n_ep, dtype=int); tot = np.zeros(n_ep)
    for _ in range(H):
        D = rng.poisson(DEMANDA_MEDIA[d]); a = pi[(d * NA + u1) * NA + u2]
        v2 = np.minimum(u2, D); r_ = D - v2; v1 = np.minimum(u1, r_); r_ = r_ - v1; v0 = np.minimum(a, r_); falt = r_ - v0
        tot += PRECIO * (v2 + v1 + v0) - COSTO * a - MERMA * (u2 - v2) - FALTANTE * falt
        u2, u1, d = u1 - v1, a - v0, (d + 1) % 7
    return tot / H

ref = totales_por_episodio(politicas[f"Mejor S global (S = {S_MEJOR})"])
filas = []
for nombre in ["Q-learning", "SARSA"]:
    for sd, pi in zip(SEMILLAS, politicas_rl[nombre]):
        dif = totales_por_episodio(pi) - ref
        filas.append({"Algoritmo": nombre, "Semilla": sd, "Diferencia media vs. mejor S": dif.mean(),
                      "IC95 ±": 1.96 * dif.std(ddof=1) / np.sqrt(len(dif))})
pruebas = pd.DataFrame(filas)
pruebas["¿Supera a la mejor S?"] = np.where(pruebas["Diferencia media vs. mejor S"] - pruebas["IC95 ±"] > 0, "Sí", "No")
print(pruebas.round(3))

# ======================================================================
# visitas
# ======================================================================
# Frecuencia con que la política óptima visita cada estado (cadena de Markov exacta, 28 días desde inventario vacío)
P_pi = T[np.arange(N_ESTADOS), PI_OPT]                      # matriz de transición bajo la política óptima
mu = np.zeros(N_ESTADOS)
for d in range(7):
    mu[PanaderiaEnv.codificar(d, 0, 0)] = 1 / 7
visitas = np.zeros(N_ESTADOS)
for _ in range(H):
    visitas += mu / H
    mu = mu @ P_pi
print("Estados visitados con probabilidad > 0,01 %:", int((visitas > 1e-4).sum()), "de", N_ESTADOS)
print("Masa de probabilidad en los 100 estados más visitados:", round(float(np.sort(visitas)[::-1][:100].sum()), 3))

def brecha_y_acuerdo(pi):
    """Brecha de valor (Q*(s,pi*(s)) - Q*(s,pi(s))) y coincidencia de acciones, ponderadas por las visitas de la política óptima."""
    brecha = (V_opt - Q_opt[np.arange(N_ESTADOS), pi])
    return (visitas * brecha).sum(), (visitas * (pi == PI_OPT)).sum()

filas = {}
for nombre, pis in politicas_rl.items():
    vals = np.array([brecha_y_acuerdo(pi) for pi in pis])
    filas[nombre] = {"Brecha de valor media (miles COP)": vals[:, 0].mean(), "Coincidencia con la óptima (ponderada)": vals[:, 1].mean()}
filas["Mejor S global"] = dict(zip(["Brecha de valor media (miles COP)", "Coincidencia con la óptima (ponderada)"],
                                   brecha_y_acuerdo(politicas[f"Mejor S global (S = {S_MEJOR})"])))
print(pd.DataFrame(filas).T.round(3))

# ======================================================================
# politica_fig
# ======================================================================
# ¿Qué aprendió el agente? Pedido con inventario vacío cada día y mapa de decisiones del viernes
pedido_vacio = lambda pi: pi.reshape(7, NA, NA)[:, 0, 0]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})

ax = axes[0]
x = np.arange(7); w = 0.38
ax.bar(x - w / 2, pedido_vacio(PI_OPT), w, color="black", alpha=0.85, label="Óptima (Value Iteration)")
pq = np.array([pedido_vacio(pi) for pi in politicas_rl["Q-learning"]])
ax.bar(x + w / 2, pq.mean(0), w, color="teal", alpha=0.9, label="Q-learning (media de semillas)")
ax.scatter(np.repeat(x, len(SEMILLAS)) + w / 2, pq.T.ravel(), s=14, color="white", edgecolor="black", zorder=3)
ax.set_xticks(x); ax.set_xticklabels([d[:3] for d in DIAS])
ax.set_ylabel("Unidades pedidas con inventario vacío")
ax.set_title("Pedido con u1 = u2 = 0")
ax.legend(fontsize=8)

dia_mapa = 4   # viernes
for ax, (titulo, pi) in zip(axes[1:], [("Óptima (Value Iteration)", PI_OPT), ("Q-learning (semilla 0)", politicas_rl["Q-learning"][0])]):
    M = pi.reshape(7, NA, NA)[dia_mapa].astype(float)
    V = visitas.reshape(7, NA, NA)[dia_mapa]
    M[V < 1e-5] = np.nan                                    # se ocultan los estados prácticamente inalcanzables
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=CAP, origin="lower")
    for i in range(NA):
        for j in range(NA):
            if not np.isnan(M[i, j]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel("u2: unidades que vencen hoy"); ax.set_ylabel("u1: unidades que vencen mañana")
    ax.set_title(f"{titulo}\nViernes: pedido según inventario"); ax.grid(False)
plt.colorbar(im, ax=axes[2], label="Unidades pedidas")
plt.tight_layout()
guardar_figura("fig4_politica_aprendida")
plt.show()

# ======================================================================
# traza_aprendida
# ======================================================================
# Dos semanas con la política aprendida (Q-learning, semilla 0) frente a la mejor regla fija, con la misma demanda
def simular_traza(pi, dias=14, dia_ini=0, seed=11):
    envt = PanaderiaEnv(seed); s = envt.reset(dia=dia_ini); filas = []
    for _ in range(dias):
        d, u1, u2 = envt.decodificar(s)
        a = int(pi[s]); s, r, term, trunc, info = envt.step(a)
        filas.append({"Día": DIAS[d][:3], "u1": u1, "u2": u2, "Pedido": a, "Demanda": info["demanda"], "Ventas": info["ventas"],
                      "Faltante": info["faltante"], "Merma": info["merma"], "Ganancia": r})
    return pd.DataFrame(filas)

tr_q = simular_traza(politicas_rl["Q-learning"][0])
tr_s = simular_traza(politicas[f"Mejor S global (S = {S_MEJOR})"])
print(f"Ganancia en 14 días | Q-learning: {tr_q['Ganancia'].sum():.1f} | Mejor S global: {tr_s['Ganancia'].sum():.1f}"
      f" | faltantes: {tr_q['Faltante'].sum()} vs {tr_s['Faltante'].sum()} | merma: {tr_q['Merma'].sum()} vs {tr_s['Merma'].sum()}")
print(tr_q)

# ======================================================================
# sens_func
# ======================================================================
def barrer(nombre, valores, **fijos):
    """Entrena Q-learning para cada valor de un hiperparámetro y mide la política voraz final (demanda nueva)."""
    filas = []
    for v in valores:
        gs = []
        for sd in SEMILLAS_BARRIDO:
            kw = dict(fijos); kw.update({nombre: v})
            Q_tab, _ = entrenar("q", episodios=EPISODIOS_BARRIDO, seed=sd, cada=EPISODIOS_BARRIDO, **kw)
            gs.append(evaluar_politica(np.argmax(Q_tab, axis=1), n_ep=2000, seed=2026)["Ganancia/día"])
        filas.append({"Valor": v, "Ganancia/día (media)": np.mean(gs), "Desv. entre semillas": np.std(gs, ddof=1), "Mínimo": np.min(gs)})
        print(f"  {nombre} = {v}: {np.mean(gs):.3f} ± {np.std(gs, ddof=1):.3f}", flush=True)
    return pd.DataFrame(filas).set_index("Valor")

# ======================================================================
# sens_alpha
# ======================================================================
print("Tasa de aprendizaje alpha:")
sens_alpha = barrer("alpha", [0.01, 0.05, 0.1, 0.3, 0.5])
print(sens_alpha.round(3))

# ======================================================================
# sens_gamma
# ======================================================================
print("Factor de descuento gamma:")
sens_gamma = barrer("gamma", [0.0, 0.5, 0.9, 0.95, 0.99])
print(sens_gamma.round(3))

# ======================================================================
# sens_eps
# ======================================================================
print("Epsilon final (exploración residual):")
sens_eps = barrer("eps_fin", [0.0, 0.01, 0.05, 0.2, 0.5])
print("Fracción del entrenamiento con exploración decreciente:")
sens_frac = barrer("frac_explor", [0.1, 0.3, 0.6, 0.9])
print(pd.concat({"epsilon final": sens_eps, "fracción de decaimiento": sens_frac}).round(3))

# ======================================================================
# sens_fig
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharey=True)
for ax, (df, titulo, xlab) in zip(axes.ravel(), [(sens_alpha, "Tasa de aprendizaje", "alpha"),
                                          (sens_gamma, "Factor de descuento", "gamma"),
                                          (sens_eps, "Exploración residual", "epsilon final"),
                                          (sens_frac, "Duración del decaimiento", "fracción del entrenamiento")]):
    ax.errorbar(range(len(df)), df["Ganancia/día (media)"], yerr=df["Desv. entre semillas"], fmt="o-", color="teal", capsize=3)
    ax.axhline(GAN_OPT, color="black", ls="--", lw=1, label="Óptima")
    ax.axhline(GAN_SGL, color="goldenrod", ls="--", lw=1, label="Mejor S global")
    ax.set_xticks(range(len(df))); ax.set_xticklabels([str(v) for v in df.index])
    ax.set_xlabel(xlab); ax.set_title(titulo, fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("Ganancia media por día (miles de COP)")
axes[0, 0].legend(fontsize=8, loc="lower right")
plt.tight_layout()
guardar_figura("fig5_sensibilidad_hiperparametros")
plt.show()

# ======================================================================
# terminal
# ======================================================================
# Dificultad detectada: ¿qué pasa si el fin del episodio (día 28) se trata como estado terminal?
filas = []
for etiqueta, flag in [("Con arranque al truncar (usado)", False), ("Terminal al truncar", True)]:
    gs = []
    for sd in SEMILLAS_BARRIDO:
        Q_tab, _ = entrenar("q", episodios=EPISODIOS_BARRIDO, seed=sd, cada=EPISODIOS_BARRIDO, terminal_al_truncar=flag)
        gs.append(evaluar_politica(np.argmax(Q_tab, axis=1), n_ep=2000, seed=2026)["Ganancia/día"])
    filas.append({"Tratamiento del día 28": etiqueta, "Ganancia/día (media)": np.mean(gs), "Desv. entre semillas": np.std(gs, ddof=1)})
print(pd.DataFrame(filas).set_index("Tratamiento del día 28").round(3))

# ======================================================================
# final_tabla_ref
# ======================================================================
# Tabla resumen para el informe (valores de todas las secciones)
resumen = tabla_final[["Ganancia/día", "Nivel de servicio", "Merma (% de lo pedido)", "% de la brecha aleatoria→óptima"]].copy()
resumen.to_csv(os.path.join(DATA_DIR, "resumen_para_informe.csv"))
for nombre, df in [("alpha", sens_alpha), ("gamma", sens_gamma), ("eps_fin", sens_eps), ("frac_explor", sens_frac)]:
    df.to_csv(os.path.join(DATA_DIR, f"sensibilidad_{nombre}.csv"))
print(resumen.round(3))
