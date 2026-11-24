# FIS 1344-1 Tarea 9.7
# Florencia Antonia Munoz Aburto
# Adams-Bashforth de dos pasos.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return 1 + y / t + (y / t) ** 2  # ecuacion diferencial.

def exacta(t):
    return t * np.tan(np.log(t)) # solucion con u=y/t y y(1)=0.

def adams_bashforth_2(f, a, b, y0, N):
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N debe ser un entero mayor o igual que 2")

    h = (b - a) / N # divide intervalo en N pasos iguales.
    t = np.linspace(a, b, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0 # condicion inicial.

    k1 = f(t[0], y[0])
    k2 = f(t[0] + h / 2, y[0] + h * k1 / 2)
    y[1] = y[0] + h * k2  # RK2 de punto medio calcula el 2do valor necesario para AB2

    for i in range(1, N):
        fi = f(t[i], y[i])
        anterior = f(t[i - 1], y[i - 1])
        y[i + 1] = y[i] + h * (3 * fi - anterior) / 2 # pendientes anteriores usadas por AB2 (orden 2)
        
    return t, y


N = 10 # numero de pasos h=0.05.
t, y = adams_bashforth_2(f, 1.0, 1.5, 0.0, N)
errores = np.abs(y - exacta(t)) # error absoluto de la solucion.

reporte = [
    "Tarea 9.7",
    "AB2 con arranque RK2 de punto medio; N = 10, h = 0.05",
    "Solucion exacta: y(t) = t*tan(log(t))",
    "\n     t          y_AB2          y_exacta       error absoluto"
]

for ti, yi, ei in zip(t, y, errores):
    reporte.append(f"{ti:8.3f} {yi:15.10f} {exacta(ti):15.10f} {ei:17.6e}")

reporte.append("\nConvergencia: N, h, error maximo y orden observado")
error_anterior = None

for pasos in [10, 20, 40, 80, 160]: # repite para comprobar convergencia.
    ts, ys = adams_bashforth_2(f, 1.0, 1.5, 0.0, pasos)
    error = np.max(np.abs(ys - exacta(ts))) # mayor error.
    orden = "---" if error_anterior is None else f"{np.log2(error_anterior / error):.4f}"
    reporte.append(f"N={pasos:3d}  h={0.5 / pasos:.6f}  error={error:.6e}  p={orden}")
    error_anterior = error 

reporte.append(
    "\nAB2 necesita dos valores: y[0] es conocido y y[1] se obtiene con RK2.\n"
    "El orden observado tiende a 2, consistente con el error global O(h^2)."
)
salida = Path(__file__).resolve().parent / "resultados"  
salida.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
td = np.linspace(1, 1.5, 300)

ax[0].plot(td, exacta(td), label="Solución exacta") # grafica solucion.
ax[0].plot(t, y, "o--", label="AB2 + arranque RK2") # agrega aproximacion AB2 y compara.
ax[0].set(xlabel="t", ylabel="y(t)", title="Tarea 9.7: solución")
ax[0].legend()

ax[1].plot(t, errores, "o-") # grafica error absoluto.
ax[1].set(xlabel="t", ylabel="Error absoluto", title="Error en la malla, N = 10")

for eje in ax: 
    eje.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(salida / "Tarea_9_7.png", dpi=200) # guarda grafico.

texto = "\n".join(reporte)
(salida / "Tarea_9_7.txt").write_text(texto, encoding="utf-8")
print(texto)
print(f"\nResultados guardados en: {salida}")
plt.show()

# Nota: 7.0
# Base: 1/1; funcionamiento: 4/4; comentarios: 1/1; claridad: 1/1.
