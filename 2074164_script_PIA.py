"""
Nombre: Gustavo Arreola Almaguer
Matricula: 2074164
Declaro que este codigo es de mi autoria 
y ha sido desarrollado bajo los criterios de integridad academica de la FCFM
"""

"""
Para la correcta ejecución de este codigo utilice el siguiente comando en la aplicación 
conocida como simbolo de sistema en Windows o terminal si esta en Linux o macOS
(este comando varia si usa Windows, macOS o Linux)

Para Windows use py -m pip install numpy matplotlib scipy si este falla solo debe cambiar py por python
Para macOS y Linux use el comando python3 -m pip install numpy matplotlib scipy
"""

"""
Competencia de precios y promociones entre dos cadenas de retail
Cadena 1: PriceCorp
Cadena 2: MaxClub
"""
# ── Librerias ───────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
 
# ── Estrategias ───────────────────────────────────────────────────────────────
epc = ["Descuento -15%", "2x1 Electrónicos", "12 MSI", "EDLP", "Flash Sale", "Cashback 5%"]
emc = ["Sin promo", "Cupones -10%", "Puntos dobles", "Liquidación", "Bundle Hogar", "IA Pricing"]
M, N = len(epc), len(emc)
 
# ── 5 aspectos: costos (MXN Millones), tiempos (sem activa / prep), capacidad (%),
#    elasticidad demanda, probabilidad de éxito — escala trimestral ───────────
T = 13  # semanas por trimestre
 
#                           Desc   2x1   MSI  EDLP  Flash  Cash
costo_pc  = np.array([      25.5,  15.6, 11.4,  6.3,  12.8, 14.1])
dur_pc    = np.array([        13,    6,   13,   13,     3,   13 ])
prep_pc   = np.array([         1,    3,    4,    1,     1,    3 ])
cap_pc    = np.array([      0.93,  0.55, 0.78, 1.00,  0.65, 0.82])
elast_pc  = np.array([      0.24,  0.14, 0.11, 0.08,  0.27, 0.13])
prob_pc   = np.array([      0.76,  0.58, 0.67, 0.91,  0.69, 0.64])
 
#                           S/p  Cup   Ptos  Liq   Bund   IA
costo_mc  = np.array([       0.0, 12.9,  9.3, 17.4, 18.6, 22.5])
dur_mc    = np.array([        13,   9,   13,    6,    9,   13 ])
prep_mc   = np.array([         1,   2,    3,    2,    3,    5 ])
cap_mc    = np.array([      1.00, 0.72, 0.88, 0.60, 0.50, 0.78])
elast_mc  = np.array([      0.00, 0.17, 0.10, 0.19, 0.13, 0.16])
prob_mc   = np.array([      1.00, 0.72, 0.77, 0.63, 0.57, 0.71])
 
# ── Factores combinados ───────────────────────────────────────────────────────
fpc = ((1 - 0.20*(costo_pc/costo_pc.max())) *     # costo
      (dur_pc/T) * (1 - 0.08*prep_pc/T)    *     # tiempo
      cap_pc                               *     # capacidad
      (1 + elast_pc)                       *     # demanda
      prob_pc)                                   # probabilidad
 
fmc = ((1 - 0.20*(costo_mc/costo_mc.max())) *
      (dur_mc/T) * (1 - 0.08*prep_mc/T)    *
      cap_mc                               *
      (1 + elast_mc)                       *
      prob_mc) 
 
# ── Matriz base y ajustada ──────────────────────────────────────────────────── 
A_base = np.array([ 
    [ 4.2, -1.8,  2.1,  3.5, -0.9,  1.2], 
    [ 5.1,  0.7, -2.3,  4.8,  1.4, -1.5], 
    [ 3.8,  2.2,  0.5,  2.9,  3.1,  0.8], 
    [ 1.5, -0.3,  1.8,  0.6,  2.4, -2.1], 
    [ 6.3,  1.1, -0.8,  5.2, -1.2,  2.4], 
    [ 2.7,  3.4,  1.2,  1.9,  0.6, -0.5], 
])  
A = A_base * fpc[:, None] / fmc[None, :]
 
# ── Punto de silla ────────────────────────────────────────────────────────────
maximin, minimax = A.min(axis=1).max(), A.max(axis=0).min()
print("=" * 55)
print("  JUEGO DE SUMA CERO — PriceCorp vs MaxClub (Q)")
print("=" * 55)
print(f"  Maximin : {maximin:+.4f} pp")
print(f"  Minimax : {minimax:+.4f} pp")
if np.isclose(maximin, minimax, atol=1e-4):
    r, c = np.unravel_index(np.argmin(np.abs(A - maximin)), A.shape)
    print(f"   Punto de silla → {epc[r]}  vs  {emc[c]}")
    print(f"  Valor del juego : {maximin:+.4f} pp")

    p = np.zeros(M); p[r] = 1.0   # PriceCorp juega siempre la estrategia r
    q = np.zeros(N); q[c] = 1.0   # MaxClub juega siempre la estrategia c
    V = maximin                    # El valor del juego es el punto de silla
    
else:
    # ── Estrategias mixtas (LP) ───────────────────────────────────────────────
    print("   Sin punto de silla → estrategias mixtas")
    shift = -A.min() + 1.0
    B = A + shift
 
    y = linprog(np.ones(M), A_ub=-B.T, b_ub=-np.ones(N),
            bounds=[(0,None)]*M, method='highs').x
    p = y / y.sum()
    V = 1 / y.sum() - shift
 
    c2 = np.zeros(N+1); c2[-1] = 1
    q  = linprog(c2, A_ub=np.hstack([B,-np.ones((M,1))]), b_ub=np.zeros(M),
             A_eq=np.hstack([np.ones((1,N)), [[0]]]), b_eq=[1],
             bounds=[(0,None)]*N+[(None,None)], method='highs').x[:N]
 
    print(f"\n  V* = {V:+.4f} pp / trimestre (ventaja PriceCorp)\n")
    print("  PriceCorp  p*:", {s: round(pi,3) for s,pi in zip(epc,p) if pi>0.01})
    print("  MaxClub q*:", {s: round(qi,3) for s,qi in zip(emc,q) if qi>0.01})

# ── Gráficas ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Juego de Suma Cero — PriceCorp vs MaxClub  |  Trimestral\n"
             "Matriz ajustada: Costos · Tiempos · Capacidades · Demandas · Probabilidades",
             fontsize=12)
etp = [s.replace(" ","\n") for s in epc]
etm = [s.replace(" ","\n") for s in emc]
 
# Panel 1 — Heatmap
ax = axes[0]
vabs = max(abs(A.min()), abs(A.max()))
im = ax.imshow(A, cmap="RdYlGn", aspect="auto", vmin=-vabs, vmax=vabs)
ax.set_xticks(range(N)); ax.set_xticklabels(etp, fontsize=7.5)
ax.set_yticks(range(M)); ax.set_yticklabels(etm, fontsize=7.5)
ax.set_title("Matriz ajustada (pp / trimestre)", fontsize=10)
for i in range(M):
    for j in range(N):
        ax.text(j, i, f"{A[i,j]:+.1f}", ha="center", va="center", fontsize=7.5)
plt.colorbar(im, ax=ax, shrink=0.75)
 
# Panel 2 — PriceCorp p*
ax = axes[1]
ax.bar(range(M), p, color=["#1565C0" if pi>0.05 else "#BBDEFB" for pi in p], width=0.6)
ax.set_xticks(range(M)); ax.set_xticklabels(etm, fontsize=7.5)
ax.set_title(f"PriceCorp — p*  (V*={V:+.4f} pp)", fontsize=10)
ax.set_ylabel("Probabilidad óptima"); ax.set_ylim(-0.13, 1.05)
[ax.text(i, pi+0.02, f"{pi:.3f}", ha="center", fontsize=8) for i,pi in enumerate(p) if pi>0.01]
[ax.text(i, -0.08, f"f={f:.2f}", ha="center", fontsize=6.5, color="#777", style="italic")
 for i,f in enumerate(fpc)]
 
# Panel 3 — MaxClub q*
ax = axes[2]
ax.bar(range(N), q, color=["#B71C1C" if qi>0.05 else "#FFCDD2" for qi in q], width=0.6)
ax.set_xticks(range(N)); ax.set_xticklabels(etp, fontsize=7.5)
ax.set_title("MaxClub — q*", fontsize=10)
ax.set_ylabel("Probabilidad óptima"); ax.set_ylim(-0.13, 1.05)
[ax.text(i, qi+0.02, f"{qi:.3f}", ha="center", fontsize=8) for i,qi in enumerate(q) if qi>0.01]
[ax.text(i, -0.08, f"f={f:.2f}", ha="center", fontsize=6.5, color="#777", style="italic")
 for i,f in enumerate(fmc)]

plt.tight_layout(pad=2.5)
plt.show()
