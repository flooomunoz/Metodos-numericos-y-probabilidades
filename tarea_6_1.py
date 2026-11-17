# FIS 1344-1 Tarea 6.1
# Florencia Antonia Munoz Aburto 
# Biseccion, tolerancia relativa y cota teorica

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 - np.cos(x) - np.sin(x) / x # define funcion que busca raiz.


def biseccion(f, a, b, tolerancia, max_iter=100):
    if not a < b or a * b <= 0 or tolerancia <= 0: # valida intervalo y tolerancia.
        raise ValueError("Se requiere a<b, tolerancia>0 e intervalo sin cero.")
    fa = f(a)
    if fa * f(b) >= 0: # verifica el cambio de signo.
        raise ValueError("Los extremos deben tener signos opuestos.")

    minimo_modulo = min(abs(a), abs(b)) # obtiene cota inferior para valor absoluto de la raiz.
    historial = [] # establece historial de iteraciones.
    for k in range(1, max_iter + 1):
        medio = a + (b - a) / 2 # calcula punto medio del intervalo.
        semiancho = (b - a) / 2 # calcula mitad del ancho del intervalo.
        cota_relativa = semiancho / minimo_modulo # estima cota del error relativo.
        historial.append([k, medio, cota_relativa]) # guarda resultados de cada iteracion.

        if cota_relativa <= tolerancia: # comprueba alcance de tolerancia.
            return medio, k, np.array(historial)

        fm = f(medio)
        if fa * fm > 0: # determina la mitad en que permanece la raiz.
            a, fa = medio, fm # actualiza extremo izquierdo.
        else:
            b = medio # actualiza extremo derecho.
    raise RuntimeError("No se alcanzó la tolerancia.")

a, b = -6.5, -6.0
raiz_exacta = -2 * np.pi # define raiz exacta de referencia.
tolerancias = 10.0**(-np.arange(2, 13)) # genera tolerancias entre 10^-2 y 10^-12.
iteraciones = []

print("TAREA 6.1")
print(f"{'Tolerancia':>12} {'N':>5} {'N teórico':>11} {'Error relativo real':>22}")
for tol in tolerancias:
    raiz, pasos, historial = biseccion(f, a, b, tol) # ejecuta biseccion para cada tolerancia.
    iteraciones.append(pasos)
    n_teorico = max(1, int(np.ceil(np.log2((b - a) / (6 * tol))))) # calcula iteraciones teoricas.
    error_real = abs(raiz - raiz_exacta) / abs(raiz_exacta) # calcula error relativo real.
    print(f"{tol:12.0e} {pasos:5d} {n_teorico:11d} {error_real:22.12e}")

print(f"\nRaíz por bisección: {raiz:.15f}")
print(f"Referencia -2*pi:  {raiz_exacta:.15f}")
print(f"Residuo f(raíz):   {f(raiz):.3e}")
print("N coincide con la cota porque la parada usa esa misma garantía.")
print("El error real puede ser bastante menor que la cota.")

k = historial[:, 0].astype(int)
errores = abs(historial[:, 1] - raiz_exacta) / abs(raiz_exacta) # calcula errores reales de cada iteracion.
cotas = historial[:, 2] # recupera cotas relativas garantizadas.

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].semilogy(iteraciones, tolerancias, "o", label="Tolerancia solicitada") # grafica tolerancia frente a iteraciones.
ax[0].semilogy(k, (b - a) / (6 * 2.0**k), "--", label="Cota teórica") # grafica cota teorica de convergencia.
ax[0].set_xlabel("Número de iteraciones")
ax[0].set_ylabel("Error relativo requerido")
ax[0].set_title("Tolerancia frente a iteraciones")

ax[1].semilogy(k, np.where(errores > 0, errores, np.nan), "o-", label="Error real") # grafica error real en escala logaritmica.
ax[1].semilogy(k, cotas, "--", label="Cota relativa garantizada") # compara error real con su cota garantizada.
ax[1].set_xlabel("Número de iteraciones")
ax[1].set_ylabel("Error relativo")
ax[1].set_title("Convergencia de la bisección")
for eje in ax:
    eje.grid(True, which="both", alpha=0.3)
    eje.legend()
plt.tight_layout()
plt.show()