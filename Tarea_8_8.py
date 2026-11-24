# FIS 1344-1 Tarea 8.8
# Florencia Antonia Munoz Aburto
# Euler y Taylor de orden 2 y 3

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def solucion_exacta(t):
    return math.acos(math.cos(1.0) - math.sin(t)) # calcula la solucion analitica de referencia.


def resolver(n, orden):  
    if not isinstance(n, (int, np.integer)) or n < 1: # verifica que n sea entero positivo.
        raise ValueError("n debe ser un entero positivo.")
    if orden not in (1, 2, 3): # valida orden 1, 2 o 3.
        raise ValueError("El orden debe ser 1, 2 o 3.")

    h = 1.0 / n # calcula tamaño del paso h.
    y = 1.0  # establece condicion inical.
    for paso in range(n):
        t = paso * h  # calcula tiempo de cada paso.
        seno = math.sin(y)
        coseno = math.cos(y)
        d1 = math.cos(t) / seno  #evalua primera derivada de la ecuacion.
        incremento = h * d1  # calcula incremento del metodo Euler.

        if orden >= 2:
            d2 = (-math.sin(t) - coseno * d1**2) / seno  # calcula segunda derivada.
            incremento += h**2 * d2 / 2  # agrega termino de Taylor de orden 2.

        if orden == 3:
            d3 = -d1 + d1**3 - 3 * coseno * d1 * d2 / seno  # calcula tercera derivada.
            incremento += h**3 * d3 / 6 # agrega termino de taylor de orden 3.

        y += incremento  # actualiza solucion aproximada.
    return y


numeros_pasos = 10 * 2**np.arange(8)  # duplica numero de pasos progresivamente.
hs = 1.0 / numeros_pasos
exacta = solucion_exacta(1.0) # obtiene solucion exacta en t=1.
nombres = ["Euler", "Taylor 2", "Taylor 3"]
aproximaciones = np.zeros((len(numeros_pasos), 3)) # alamacena resultados de los tre metodos.

for i, n in enumerate(numeros_pasos):
    for j, orden in enumerate((1, 2, 3)):
        aproximaciones[i, j] = resolver(int(n), orden)  # ejecuta cada metodo para distintos pasos.

errores = np.abs(aproximaciones - exacta) # calcula errores absolutos.
ordenes = np.log2(errores[:-1] / errores[1:])  # estima ordenes de convergencia observados.

print("TAREA 8.8")
print("Problema: y'=cos(t)/sin(y), y(0)=1, 0<=t<=1.")
print("\nSOLUCION ANALITICA:")
print("Al separar variables: sin(y) dy = cos(t) dt.")
print("Integrando: -cos(y)=sin(t)+C; con y(0)=1 se obtiene C=-cos(1).")
print("Por tanto, y(t)=arccos(cos(1)-sin(t)), en la rama que cumple y(0)=1.")
print(f"y(1) exacta = {exacta:.15f}\n")
print(f"{'n':>6} {'h':>13} {'Error Euler':>16} {'Error Taylor 2':>16} {'Error Taylor 3':>16}")

for n, h, fila in zip(numeros_pasos, hs, errores):
    print(f"{n:6d} {h:13.6e} {fila[0]:16.8e} {fila[1]:16.8e} {fila[2]:16.8e}")

print(f"\nAproximaciones finales con h={hs[-1]:.8f}:")
for j, nombre in enumerate(nombres):
    print(f"{nombre:>8}: y(1)={aproximaciones[-1,j]:.15f}; p observado={ordenes[-1,j]:.4f}")

print("\nINTERPRETACION:")
print("Los errores globales esperados son O(h), O(h**2) y O(h**3).")
print("Al dividir h por dos, los errores tienden a reducirse por factores 2, 4 y 8.")
print("Los ordenes observados se aproximan a 1, 2 y 3, como predice la teoria.")
print("La solucion permanece en (0,pi), donde sin(y) no se anula y la ecuacion es regular.")
print("Taylor de orden p omite terminos de orden h**(p+1) en cada paso;")
print("al recorrer un intervalo fijo, el error global es de orden h**p.")
print("Para pasos extremadamente pequenos, el redondeo puede alterar esta tendencia.")

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for j, nombre in enumerate(nombres):
    ax.loglog(hs, errores[:, j], "o-", label=nombre) # compara errores en escala logaritmica.
ax.set_xlabel("Paso h")
ax.set_ylabel("Error absoluto en t=1")
ax.set_title("Tarea 8.8: error de Euler y Taylor frente al paso h")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

carpeta = Path(__file__).resolve().parent / "resultados" # define la carpeta de resultados.
carpeta.mkdir(exist_ok=True)  # crea carpeta si no existe.
fig.savefig(carpeta / "Tarea_8_8.png", dpi=180, bbox_inches="tight") # guarda el grafico en formato PNG.
print("\nGrafico guardado en resultados/Tarea_8_8.png")
plt.show()  

# Nota: 7.0
# Base: 1/1; funcionamiento: 4/4; comentarios: 1/1; claridad: 1/1.
