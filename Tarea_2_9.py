# FIS 1344-1 Tarea 2.9
# Florencia Antonia Munoz Aburto
# Condicionamiento y estabilidad 

import math

def taylor_8(x):   # aproxima la funcion usando 8 terminos, para reducir errores numericos.
    suma = 0.0
    for k in range(8): # calcula los primeros 8 terminos.
        suma+= x**k /math.factorial(k + 1)
        return suma

def condicionamiento(x): # calcula la sensibilidad de la funcion ante variaciones de x.
    if x == 0: 
        return 0.0 # la condicion tiende a cero.
    return abs(x * math.exp(x) / math.expm1(x) - 1) # calcula la sensibilidad relativa de la funcion.

print("tarea 2.9") # identifica ejercicio y muestra sus resultados.
print(f"kappa(-1) = {condicionamiento(-1):.12f}") # evalua condicionamiento en x=-1.
print(f"kappa(0)  = {condicionamiento(0):.12f}") #evalua el valor limite en x=0.
print(f"Maximo en [-1,1]: kappa(1) = {1 / (math.e - 1):.12f}") # muestra el maximo condicionamineto del intervalo.
print()
print(f"{'x':>8} {'Directo':>20} {'Taylor (8)':>20} "
      f"{'Dif. relativa':>15} {'Error directo':>15} {'Error Taylor':>15}")

for j in range(2, 9): # evalua valores de x cada vez mas cerca a cero.
    x = 10.0**(-j) # genera valores de x cada vez mas pequeños.
    directo = (math.e**x - 1) / x # metodo directo, puede perder precision.
    taylor = taylor_8(x) # aproximacion mas estable para pequenos valores de x.
    
    referencia = math.expm1(x) / x # calcula una referencia mas precisa.
    diferencia = abs(directo - taylor) / abs(taylor) # compara diferencia relativa entre metodos.
    error_directo = abs(directo - referencia) / abs(referencia) # calcula error relativo del metodo directo.
    error_taylor = abs(taylor - referencia) / abs(referencia) # calcula error relativo de Taylor.

    print(f"{x:8.0e} {directo:20.15f} {taylor:20.15f} "
          f"{diferencia:15.6e} {error_directo:15.6e} {error_taylor:15.6e}")

print("\nCONCLUSIÓN:")
print("La función está bien condicionada, pero la evaluación directa pierde")
print("precisión por cancelación al restar exp(x)-1 cuando x es pequeño.")
print("Taylor evita esa cancelación y es más preciso para estos valores.")
print("Un error mostrado como cero indica coincidencia con la referencia")
print("a la precisión usada; no demuestra ausencia de error matemático.")

    


