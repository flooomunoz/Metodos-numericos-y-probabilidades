# FIS 1344-1 Tarea 5.8
# Florencia Antonia Munoz Aburto
# Sistema lineal mediante LU sin pivoteo

import numpy as np

def lu_sin_pivoteo(A):
    """Factoriza A=L@U sin intercambiar filas; L tiene diagonal unitaria."""
    U = np.array(A, dtype=float, copy=True) # copia matriz para no cambiar la original.
    n = len(U)
    L = np.eye(n) #  usa L como matriz identidada.
    for k in range(n - 1): # recorre etapas de eliminacion gaussiana.
        if U[k, k] == 0: # comprueba que pivoteo no sea cero.
            raise ValueError("Pivote nulo: esta factorización sin pivoteo no puede continuar.")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]  # calcula multiplicadores de eliminacion.
            U[i, k:] -= L[i, k] * U[k, k:] # elimina elementos bajo el pivoteo.
    if U[-1, -1] == 0:
        raise ValueError("La matriz es singular.")
    return L, U # devuelve matrices L y U.

def sustitucion_adelante(L, b):
    n = len(b)
    y = np.zeros(n) # prepara vector solucion intermedio.
    for i in range(n):
        suma = 0.0
        for j in range(i):
            suma += L[i, j] * y[j]
        y[i] = (b[i] - suma) / L[i, i] # resuelve Ly=b con sustitucion adelante.
    return y

def sustitucion_atras(U, y):
    n = len(y)
    x = np.zeros(n) # configura el vector solucion final.
    for i in range(n - 1, -1, -1): # recorre las filas desde la ultima.
        suma = 0.0
        for j in range(i + 1, n):
            suma += U[i, j] * x[j]
        x[i] = (y[i] - suma) / U[i, i] # resuelve Ux=y con sustitucion atras.
    return x

alpha = 0.1 # define parametro alpha.
b = np.array([alpha, 0, 0, 0, 1], dtype=float) # construye vector de terminos independientes.

print("TAREA 5.8")
print(f"{'beta':>12} {'x_1 - 1':>22} {'kappa_2(A)':>22}")
for potencia in range(1, 13):
    beta = 10.0**potencia # aumenta beta en potencias de diez.
    A = np.array([ # construye la matriz del sistema, para cada beta.
        [1, -1,  0, alpha - beta, beta],
        [0,  1, -1,            0,    0],
        [0,  0,  1,           -1,    0],
        [0,  0,  0,            1,   -1],
        [0,  0,  0,            0,    1]
    ], dtype=float)

    L, U = lu_sin_pivoteo(A) # factoriza A mediante LU sin pivoteo.
    y = sustitucion_adelante(L, b) # resuelve sistema triangular inferior.
    x = sustitucion_atras(U, y) # obtiene solucion del sistema triangular superior.
    kappa = np.linalg.cond(A, p=2)  # calcula numero de condicionamiento en norma 2.
    print(f"{beta:12.0e} {x[0] - 1:22.12e} {kappa:22.12e}") # muestra beta, error en x1 y condicionamiento.

print("\nCONCLUSIÓN:")
print("En aritmética exacta, x=(1,1,1,1,1). Como A ya es triangular")
print("superior, L=I y U=A. La solución se obtiene por sustitución atrás.")
print("Al aumentar beta, se pierden cifras al formar alpha-beta y al")
print("sumar términos grandes de signos opuestos en la primera ecuación.")
print("kappa_2(A) crece aproximadamente como sqrt(2)*beta²: el sistema")
print("es cada vez más sensible. Es una cota de peor caso, no el error efectivo.")