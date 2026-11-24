# FIS 1344-1 Tarea 3.8
# Florencia Antonia Munoz Aburto
# Numero de particulas condicionado

import math 
import numpy as np
import matplotlib.pyplot as plt

lam = 50 / 60  # 50 partículas/minuto equivalen a 50/60 en un segundo.
prob_impacto = -math.expm1(-lam)  # P(N>=1) = 1-exp(-lam).

n = np.arange(0, 13) # define numeros de particulas posibles de 0 a 12. 
probabilidades = np.array([ # calcula probabilidades condicionales.
    0.0 if k == 0 else # excluye caso de cero particulas.
    math.exp(-lam) * lam**int(k) / math.factorial(int(k)) / prob_impacto # aplica distribucion de Poisson condicionada a N>=1.
    for k in n
])
print("TAREA 3.8")
print(f"Promedio en un segundo: lambda = {lam:.8f}")
print(f"Probabilidad de detectar impacto: {prob_impacto:.8f}\n")
print(f"{'n':>3} {'P(N=n | impacto)':>20}")
for k, p in zip(n, probabilidades): # recorre valores y probabilidades.
    print(f"{k:3d} {p:20.12f}")

cola = max(0.0, 1 - probabilidades.sum()) # calcula probabilidad restante para N>=13.
print(f"\nProbabilidad omitida para N>12: {cola:.3e}")
print(f"P(N=1 | impacto)  = {probabilidades[1]:.6%}") # probabilidad de recibir una particula.
print(f"P(N>=2 | impacto) = {1 - probabilidades[1]:.6%}") # probabilidad de recibir dos o mas particulas.
print("Detectar un impacto no significa que haya llegado solo una partícula.")

plt.figure(figsize=(8, 4.5)) # crea figura del grafico.
plt.bar(n, probabilidades, color="steelblue", edgecolor="black") # probabilidades mediante barras.
plt.xticks(n)
plt.xlabel("Número de partículas n")
plt.ylabel("P(N=n | N≥1)")
plt.title("Partículas recibidas, dado que el sensor indicó impacto") # agrega cuadricula.
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show() # muestra grafico.


# Nota: 6.0
# Base: 1/1; funcionamiento: 4/4; comentarios: 1/1; claridad: 1/1.
# Nota según rúbrica: 7.0; descuento por atraso: 1.0.
# Se descuenta un punto por entrega tardía según el acuerdo especial.
