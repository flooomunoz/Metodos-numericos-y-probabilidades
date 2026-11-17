# FIS 1344-1 Tarea 4.3
# Florencia Antonia Munoz Aburto
# Funcion caracteristica y cumulantes de la exponencial

import math
import numpy as np
import matplotlib.pyplot as plt

def caracteristica(k, lam):
    if lam <= 0:
        raise ValueError("lambda debe ser positiva.") # verifica lambda positivo.
    return lam / (lam + 1j * k) # calcula funcion caracteristica exponencial.

def cumulantes_exponencial(lam):
    if lam <= 0:
        raise ValueError("lambda debe ser positiva.")
    return np.array([math.factorial(r - 1) / lam**r for r in range(1, 5)]) # calcula primeros cuatro cumulantes.

lam = 2.0 # define parametro de distribucion.
k_exp = cumulantes_exponencial(lam)
mu = 1 / lam # calcula la media de ambas distribuciones.
sigma = 1 / lam # calcula desviacion estandar. 
k_gauss = np.array([mu, sigma**2, 0.0, 0.0]) #define cumulantes de la gaussiana.

print("TAREA 4.3")
print("Resultado general: phi(k) = lambda/(lambda + i*k).")
print("Cumulantes: 1/lambda, 1/lambda², 2/lambda³, 6/lambda⁴.")
print(f"\nEjemplo con lambda = {lam} y gaussiana de igual media y varianza:")
print(f"{'Orden':>6} {'Exponencial':>15} {'Gaussiana':>15}")
for r in range(4): # compara cuatro cumulantes de ambas distribuciones.
    print(f"{r+1:6d} {k_exp[r]:15.8f} {k_gauss[r]:15.8f}")

print(f"\nAsimetría exponencial: {k_exp[2] / k_exp[1]**1.5:.1f}") # calcula asimetria exponencial.
print(f"Exceso de curtosis: {k_exp[3] / k_exp[1]**2:.1f}") # calcula exceso de curtosis.
print("La gaussiana tiene asimetría y exceso de curtosis iguales a cero.")
print("kappa_4 es el cuarto cumulante, no el cuarto momento central.")
print(f"Verificación de normalización: phi(0) = {caracteristica(0, lam)}") # verifica que funcion caracteristica valga uno en cero.

x = np.linspace(-2 * sigma, mu + 6 * sigma, 1000) # genera valores para graficar ambas distribuciones.
rho_exp = np.where(x >= 0, lam * np.exp(-lam * x), 0.0) # calcula densidad exponencial para x>=0
rho_gauss = np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi)) # calcula densidad gaussiana.
plt.figure(figsize=(8, 4.5))
plt.plot(x, rho_exp, label="Exponencial") # grafica distribucion exponencial.
plt.plot(x, rho_gauss, label="Gaussiana: misma media y varianza") # grafica gaussiana con misma media y varianza. 
plt.xlabel("x")
plt.ylabel("Densidad de probabilidad")
plt.title("La media y la varianza no determinan toda la distribución")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
