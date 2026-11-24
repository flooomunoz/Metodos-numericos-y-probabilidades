# FIS 1344-1 Tarea 7.7
# Florencia Antonia Munoz Aburto
# Cuadratura de Gauss-Legendre compuesta

import numpy as np
from numpy.polynomial.legendre import leggauss

def gauss_compuesta(f, a, b, n, m=5):
    """Integra f entre a y b con n subintervalos y m nodos en cada uno."""
    if not isinstance(n, (int, np.integer)) or n < 1: # verifica que n se a entero positivo.
        raise ValueError("n debe ser un entero positivo.")
    if not isinstance(m, (int, np.integer)) or m < 1: # verifica que m sea entero positivo.
        raise ValueError("m debe ser un entero positivo.")

    nodos, pesos = leggauss(m) # obtiene nodos y pesos de Gauss-Legendre.
    h = (b - a) / n # calcula ancho de cada intervalo.
    integral = 0.0 # prepara el acumulador de la integral.
    for j in range(n): # recorre n subintervalos.
        izquierda = a + j * h
        derecha = a + (j + 1) * h
        centro = (izquierda + derecha) / 2 # calcula el centro del subintervalo.
        semiancho = (derecha - izquierda) / 2 # calcula la mitad del ancho del subintervalo.

        suma_local = 0.0 
        for t, w in zip(nodos, pesos): # recorre nodos y pesos de cuadratura.
            x = centro + semiancho * t # transforma cada nodo al subintervalo actual.
            suma_local += w * f(x) # suma valores ponderados de la funcion.

        integral += semiancho * suma_local # acumula integral aproximada de cada subintervalo.
    return integral

exacta = np.sin(2) # calcula valor exacto para comparar.
print("TAREA 7.7") 
print(f"Integral exacta de cos(x) en [0,2]: {exacta:.15f}\n")
print(f"{'n':>4} {'m':>4} {'Integral aproximada':>23} {'Error absoluto':>18}")
for n in [1, 2, 4, 8, 16, 32]: # prueba diversos numeros de subintervalos.
    for m in [2, 5]: # compara reglas de dos y cinco nodos.
        resultado = gauss_compuesta(np.cos, 0.0, 2.0, n, m) # calcula integral aproximada.
        print(f"{n:4d} {m:4d} {resultado:23.15f} {abs(resultado-exacta):18.8e}") # muestra el resultado y su error absoluto.

polinomio = lambda x: x**3 - 2*x + 1 # define polinomio cubico para verificar metodo.
resultado = gauss_compuesta(polinomio, 0.0, 2.0, n=4, m=2) # comprueba exactitud de la regla de los nodos.
print(f"\nVerificación con x³-2x+1 en [0,2]: {resultado:.15f}; exacta=2.")
print("n es el número de subintervalos; m, los nodos por subintervalo.")
print("Con m nodos la regla es exacta hasta grado 2m-1, salvo redondeo.") 


# Nota: 6.0
# Base: 1/1; funcionamiento: 4/4; comentarios: 1/1; claridad: 1/1.
# Nota según rúbrica: 7.0; descuento por atraso: 1.0.
# Se descuenta un punto por entrega tardía según el acuerdo especial.
