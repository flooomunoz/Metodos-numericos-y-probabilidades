# FIS 1344-1 Tarea 10.2
# Florencia Antonia Munoz Aburto
# RK4 para una ecuacion de orden n.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def rk4_orden_n(f, a, b, condiciones, N):
    inicial = np.asarray(condiciones, dtype=float) # condiciones iniciales y sus derivadas.

    if inicial.ndim != 1 or inicial.size == 0:
        raise ValueError("Condiciones iniciales deben ser vector no vacio")

    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N es un entero positivo")
    n = inicial.size # numero de condicionamiento determina orden de la ecuacion.
    h = (b - a) / N
    t = np.linspace(a, b, N + 1)
    U = np.zeros((N + 1, n)) # guarda y a sus derivadas en cada momento.
    U[0] = inicial

    def sistema(ti, u):
        du = np.empty(n)
        du[:-1] = u[1:]  # transforma derivadas en sistema de primer orden.
        du[-1] = f(ti, *u)  # calcula derivada de mayor orden con f.
        return du   
    
    for i in range(N): 
        ti, u = t[i], U[i]
        k1 = sistema(ti, u)
        k2 = sistema(ti + h / 2, u + h * k1 / 2)
        k3 = sistema(ti + h / 2, u + h * k2 / 2)
        k4 = sistema(ti + h, u + h * k3)
        U[i + 1] = u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, U[:, 0] # devuelve los tiempos y la solucion y (t)

def f(t, y, dy):
    return 2 * dy - y + t * np.exp(t) - t

def exacta(t):
    return t ** 3 * np.exp(t) / 6 - t * np.exp(t) + 2 * np.exp(t) - t - 2 # solucion exacta para comprobar resultado.

N = 10  
t, y = rk4_orden_n(f, 0.0, 1.0, [0.0, 0.0], N) 
errores = np.abs(y - exacta(t)) # error absoluto de la solucion exacta.

reporte = [
    "TAREA 10.2",
    "RK4 general para una EDO de orden n.",
    "Prueba: y'' = 2y' - y + t*exp(t) - t; y(0)=y'(0)=0.",
    "\n       t          y_RK4          y_exacta       error absoluto"
]

for ti, yi, ei in zip(t, y, errores):
    reporte.append(f"{ti:8.3f} {yi:15.10f} {exacta(ti):15.10f} {ei:17.6e}")

reporte.append("\nConvergencia: N, h, error máximo y orden observado")
error_anterior = None

for pasos in [10, 20, 40, 80, 160]: # comprueba convergencia.
    ts, ys = rk4_orden_n(f, 0.0, 1.0, [0.0, 0.0], pasos)
    error = np.max(np.abs(ys - exacta(ts))) # orden cercano a cuatro.
    orden = "---" if error_anterior is None else f"{np.log2(error_anterior / error):.4f}"
    reporte.append(f"N={pasos:3d}  h={1 / pasos:.6f}  error={error:.6e}  p={orden}")
    error_anterior = error  

reporte.append(
    "\nSe transforma la EDO de orden n en n ecuaciones de primer orden.\n"
    "Las cuatro etapas se calculan para todo el vector; el error global es O(h^4)."
)

salida = Path(__file__).resolve().parent / "resultados"
salida.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
td = np.linspace(0, 1, 300)

ax[0].plot(td, exacta(td), label="Solución exacta")
ax[0].plot(t, y, "o--", label="RK4, N = 10")
ax[0].set(xlabel="t", ylabel="y(t)", title="Tarea 10.2: solución")
ax[0].legend()

ax[1].plot(t, errores, "o-")
ax[1].set(xlabel="t", ylabel="Error absoluto", title="Error en la malla, N = 10")

for eje in ax: 
    eje.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(salida / "Tarea_10_2.png", dpi=200) # guarda el grafico.

texto = "\n".join(reporte)
(salida / "Tarea_10_2.txt").write_text(texto, encoding="utf-8")
print(texto)
print(f"\nResultados guardados en: {salida}")
plt.show()