# FIS 1344-1 Tarea 11.4
# Florencia Antonia Munoz Aburto
# Descenso de gradiente desde los puntos iniciales.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x_data = np.linspace(0, 10, 30)
y_true = (x_data /2) ** 2 + 50 * x_data / 2
y_data = y_true + np.random.normal(0, 0.2, size=x_data.shape) # añade ruido normal con  desviacion estandar de 0.2.

def modelo(x, theta):
    return (theta * x) ** 2 + 50 * theta * x

def costo(theta):
    return np.mean((modelo(x_data, theta) - y_data) ** 2) # error cuadratico medio que busca minimizar.

def gradiente(theta):
    residuo = modelo(x_data, theta) - y_data
    derivada = 2 * theta * x_data ** 2 + 50 * x_data # derivada del modelo segun theta.
    return 2 * np.mean(residuo * derivada) # gradiente por regla de la cadena.

def descenso(theta0, eta=1e-6, tol=1e-7, max_iter=10000):
    theta = float(theta0)
    parametros = [theta]
    costos = [costo(theta)]

    for _ in range(max_iter):
        g = gradiente(theta)

        if abs(g) < tol: # termina cuando el gradiente es lo suficientemnete pequeño.
            return theta, np.array(parametros), np.array(costos)
        
        theta = theta - eta * g 

        if not np.isfinite(theta):
            raise RuntimeError("El descenso divergió: reduzca la tasa de aprendizaje")

        parametros.append(theta)
        costos.append(costo(theta))

    raise RuntimeError("No se alcanzó la tolerancia dentro del máximo de iteraciones")

reporte = [
    "TAREA 11.4",
    "Costo: error cuadrático medio (ECM).",
    "Tasa de aprendizaje = 1e-6; tolerancia del gradiente = 1e-7."
]

soluciones = []

for inicio in [2.0, -9.0]:  # compara datos iniciales.
    theta, historial, costos = descenso(inicio)
    soluciones.append((inicio, theta, historial, costos))
    reporte.append(
        f"\nInicio={inicio:g}; theta final={theta:.12f}; "
        f"iteraciones={len(costos) - 1}"
    )
    reporte.append(f"ECM={costo(theta):.10f}")
    reporte.append(f"|gradiente|={abs(gradiente(theta)):.3e}")

coeficientes = [
    2 * np.mean(x_data ** 4),
    150 * np.mean(x_data ** 3),
    np.mean(2500 * x_data ** 2 - 2 * x_data ** 2 * y_data),
    -50 * np.mean(x_data * y_data)
]  

criticos = np.sort(np.roots(coeficientes).real)  # puntos del gradiente cero.
reporte.append("\nPuntos estacionarios del costo:")

for theta in criticos:
    d = 2 * theta * x_data ** 2 + 50 * x_data
    r = modelo(x_data, theta) - y_data
    segunda = 2 * np.mean(d ** 2 + 2 * x_data ** 2 * r)
    tipo = "mínimo" if segunda > 0 else "máximo"  # minimos y maximos de la segunda derivada.
    reporte.append(f"theta={theta:.9f}; ECM={costo(theta):.9f}; {tipo}")

reporte.append(
    "\nDesde theta=2 se alcanza el mínimo global, cercano al valor real 0.5.\n"
    "Desde theta=-9 se alcanza un mínimo local con un error mucho mayor.\n"
    "El costo no es convexo: tiene dos mínimos separados por un máximo.\n"
    "Por eso, con la tasa utilizada, el resultado depende del punto inicial.\n"
    "Un gradiente pequeño no garantiza un buen ajuste.\n"
    "El error del mejor ajuste no es cero porque los datos contienen ruido."
)

salida = Path(__file__).resolve().parent / "resultados"
salida.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
xd = np.linspace(0, 10, 300)

ax[0].scatter(x_data, y_data, s=18, color="black", label="Datos con ruido")
ax[0].plot(xd, modelo(xd, 0.5), "k:", label="Modelo verdadero")

theta_grid = np.sort(np.r_[np.linspace(-10, 3, 1500), criticos])  
ax[2].plot(theta_grid, [costo(v) for v in theta_grid], color="black")

for inicio, theta, historial, costos in soluciones:
    etiqueta = f"Inicio {inicio:g}; final {theta:.4f}"
    ax[0].plot(xd, modelo(xd, theta), label=etiqueta) # compara ajustes.
    ax[1].semilogy(np.arange(len(costos)), costos, label=f"Inicio {inicio:g}")
    ax[2].semilogy(historial, costos, "o-", markersize=3, label=f"Inicio {inicio:g}") # muestra trayectorias a cada minimo.

ax[0].set(xlabel="x", ylabel="y", title="Tarea 11.4: ajustes")
ax[1].set(xlabel="Iteración", ylabel="ECM (escala logarítmica)", title="Evolución del costo")
ax[2].set(xlabel="theta", ylabel="ECM (escala logarítmica)", title="Mínimos y trayectorias")

for eje in ax:
    eje.grid(True, alpha=0.3)
    eje.legend(fontsize=8)

fig.tight_layout()
fig.savefig(salida / "Tarea_11_4.png", dpi=200) # guarda imagen.

texto = "\n".join(reporte)
(salida / "Tarea_11_4.txt").write_text(texto, encoding="utf-8")
print(texto)
print(f"\nResultados guardados en: {salida}")
plt.show()

