import random
import math
from typing import List, Tuple

def generar_solucion(objetos: List[int], capacidades: List[int]) -> Tuple[List[List[int]], List[int]]:
    """
    Genera una solución inicial en base a los objetos que tenemos y las capacidades de contenedores disponibles
    
    Args:
        objetos: List[int]: Conjuntos de objetos que se busca acomodar.
        capacidades: List[int]: Capacidades de contenedores disponibles

    Returns:
        Tuple[List[List[int]], List[int]]: Solución generada y capacidades de cada contenedor (bin).
    """
    bins = []  # Contenedores utilizados
    capacidades_usadas = []  # Capacidades asociadas a los contenedores usados

    for obj in objetos: #Por cada objeto
        if random.random() < 0.5: #probabilidad de añadirlo en un contenedor existente o en uno nuevo
            
            #Añadir el objeto en un nuevo contenedor
            capacidades_validas = [cap for cap in capacidades if cap >= obj]
            if not capacidades_validas:
                raise ValueError(f"No hay contenedores con suficiente capacidad para el objeto de tamaño {obj}.")
            capacidad_nueva = random.choice(capacidades_validas)
            bins.append([obj])
            capacidades_usadas.append(capacidad_nueva)
        else:
            #Añadir el objeto a un contenedor existente
            colocado = False
            indices = list(range(len(bins)))
            random.shuffle(indices)
            for i in indices:
                if sum(bins[i]) + obj <= capacidades_usadas[i]:
                    bins[i].append(obj)
                    colocado = True
                    break
                
            if not colocado:  # Si no cabe en ningún contenedor existente, crea uno nuevo
                capacidades_validas = [cap for cap in capacidades if cap >= obj]
                if not capacidades_validas:
                    raise ValueError(f"No hay contenedores con suficiente capacidad para el objeto de tamaño {obj}.")
                capacidad_nueva = random.choice(capacidades_validas)
                bins.append([obj])
                capacidades_usadas.append(capacidad_nueva)
    return bins, capacidades_usadas

def calcular_rpfm(solucion: List[List[int]], num_objetos: int) -> List[List[float]]:
    """
    Genera una matriz de probabilidades de acuerdo a la probabilidad de que 2 objetos de un determinado tamaño esté en un mismo paquete.
    
    Args:
        solucion: List[List[int]]: Recibe la solución con la cual se generará la matriz (la mejor solución hasta el momento).
        num_objetos: int: Número de objetos diferentes disponibles (cantidad de tamaños de objeto diferentes).

    Returns:
        List[List[float]]: Matriz de probabilidad
    """
    
    rpfm = [[0 for _ in range(num_objetos)] for _ in range(num_objetos)] #Generación de la matriz inicial

    #Rellenar la matriz de probabilidad con el valor de ocurrencia de cada combinación (matriz simétrica)
    for bin_actual in solucion:
        for i in range(len(bin_actual)):
            for j in range(i + 1, len(bin_actual)):
                rpfm[bin_actual[i] - 1][bin_actual[j] - 1] += 1
                rpfm[bin_actual[j] - 1][bin_actual[i] - 1] += 1

    #Normalización de la matriz
    total_sum = sum(sum(row) for row in rpfm)
    if total_sum > 0:
        for i in range(num_objetos):
            for j in range(num_objetos):
                rpfm[i][j] /= total_sum
                rpfm[i][j] *= 2 #Puesto que la matriz es simétrica, solo nos interesa alguna de las mitades, la multiplicación por 2 para mantener la suma de probabilidades en 1

    #Mantener la simetría de la matriz
    for i in range(num_objetos):
        for j in range(num_objetos):
            rpfm[j][i] = rpfm[i][j]

    return rpfm


def evaluar_solucion(solucion: List[List[int]], capacidades: List[int], alpha: float, beta: float) -> float:
    """
    Calcula la aptitud de una solución específica
    
    Args:
        solucion: List[List[int]]: La solución a evaluar.
        capacidades: List[int]: Capacidades de cada uno de los contenedores.
        alpha: float: Ponderación para el número de contenedores.
        beta: float: Ponderación para el espacio sobrante.

    Returns:
        float: Aptitud de la solución.
    """
    num_bins = len(solucion) 
    espacio_sobrante = sum(capacidades[i] - sum(sol) for i, sol in enumerate(solucion))
    return alpha * num_bins + beta * espacio_sobrante #Cálculo de la aptitud de la solución de acuerda a la ponderación de cada parámetro

def mutacion(solucion: List[List[int]], capacidades: List[int], rpfm: List[List[float]], tamanos_disponibles: List[int]) -> Tuple[List[List[int]], List[int]]:
    """
    Aplica mutación a una solución, la cual consiste en el intercambio de objetos entre paquetes y ajuste de tamaño de los paquetes de acuerdo a la probabilidad de que 2 objetos estén en un mismo paquete.
    
    Args:
        solucion: List[List[int]]: La solución a mutar.
        capacidades: List[int]: Capacidades de cada uno de los contenedores.
        rpfm: List[List[float]]: Matriz de probabilidades de que 2 objetos estén en un mismo paquete.
        tamanos_disponibles: List[int]: Lista de todos los tamaños de contenedores disponibles.
    Returns:
        Tuple[List[List[int]], List[int]]: La solución mutada y la lista de las nuevas capacidades de los contenedores.
    """
    #Copiar la solución y las capacidades acutales
    nueva_solucion = [bin_actual[:] for bin_actual in solucion]
    capacidades_actualizadas = capacidades[:]

    # Filtrar bins con más de 2 objetos
    bins_mutables = [i for i in range(len(nueva_solucion)) if len(nueva_solucion[i]) > 2]

    # En caso de no existir bins mutables, se termina la función, en caso contrario, se puede ajustar el número de mutaciones a realizar (en este caso, se realizan todos los cambios posibles)
    num_mutaciones = len(bins_mutables)
    if num_mutaciones != 0:
        num_mutaciones = len(bins_mutables)
    else:
        return nueva_solucion, capacidades_actualizadas
    
    #Revolver los bins mutables
    bins_seleccionados = random.sample(bins_mutables, num_mutaciones)

    for i in bins_seleccionados:
        bin_actual = nueva_solucion[i]

        # Seleccionar dos objetos aleatoriamente del bin
        obj1, obj2 = random.sample(bin_actual, 2)

        # Evaluar si realizar la mutación según la probabilidad en RPFM
        if rpfm[obj1 - 1][obj2 - 1] < random.random():
            bin_actual.remove(obj1)  # Eliminar obj1 del bin actual
            colocado = False

            # Intentar recolocar obj1 en otro bin
            for j in range(len(nueva_solucion)):
                # Verificar si cabe en el bin actual
                if sum(nueva_solucion[j]) + obj1 <= capacidades_actualizadas[j]:
                    nueva_solucion[j].append(obj1)
                    colocado = True
                    break

                # Intentar aumentar el tamaño del bin si es necesario
                for nuevo_tamano in sorted(tamanos_disponibles):
                    if nuevo_tamano > capacidades_actualizadas[j] and sum(nueva_solucion[j]) + obj1 <= nuevo_tamano:
                        capacidades_actualizadas[j] = nuevo_tamano
                        nueva_solucion[j].append(obj1)
                        colocado = True
                        break

                if colocado:
                    break

            # Si no cabe en ningún bin, crear un nuevo bin
            if not colocado:
                nueva_solucion.append([obj1])
                capacidades_actualizadas.append(min(t for t in tamanos_disponibles if t >= obj1))

            # Reducir el tamaño del bin actual y del bin colocado si es posible
            if bin_actual:
                nuevo_tamano = min([t for t in tamanos_disponibles if t >= sum(bin_actual)], default=capacidades_actualizadas[i])
                capacidades_actualizadas[i] = nuevo_tamano

            if colocado:
                nuevo_tamano_destino = min([t for t in tamanos_disponibles if t >= sum(nueva_solucion[j])], default=capacidades_actualizadas[j])
                capacidades_actualizadas[j] = nuevo_tamano_destino
                
    return nueva_solucion, capacidades_actualizadas


def busqueda_local(solucion: List[List[int]], capacidades: List[int], tamanos_disponibles: List[int]) -> Tuple[List[List[int]], List[int]]:
    """
    Aplica una búsqueda local para mejorar la solución actual ajustando la distribución de los objetos entre los bins y optimizando sus tamaños.

    Args:
        solucion (List[List[int]]): Solución actual, una lista de bins con objetos.
        capacidades (List[int]): Capacidades actuales de los bins.
        tamanos_disponibles (List[int]): Lista de tamaños posibles para los bins.

    Returns:
        Tuple[List[List[int]], List[int]]: Solución mejorada y capacidades ajustadas.
    """
    nueva_solucion = [bin_actual[:] for bin_actual in solucion]
    capacidades_actualizadas = capacidades[:]

    for i in range(len(nueva_solucion)):
        for j in range(i + 1, len(nueva_solucion)):
            bin_actual = nueva_solucion[i]
            bin_destino = nueva_solucion[j]

            # Intentar mover objetos del bin_actual al bin_destino
            for obj in bin_actual[:]:
                if sum(bin_destino) + obj <= capacidades_actualizadas[j]:
                    bin_actual.remove(obj)
                    bin_destino.append(obj)

                    # Reducir tamaño del bin_actual si es posible
                    nuevo_tamano_actual = min([t for t in tamanos_disponibles if t >= sum(bin_actual)],
                                              default=capacidades_actualizadas[i])
                    capacidades_actualizadas[i] = nuevo_tamano_actual

                    # Ajustar tamaño del bin_destino si es necesario
                    nuevo_tamano_destino = min([t for t in tamanos_disponibles if t >= sum(bin_destino)],
                                               default=capacidades_actualizadas[j])
                    capacidades_actualizadas[j] = nuevo_tamano_destino

    # Limpiar bins vacíos
    bins_no_vacios = []
    capacidades_no_vacias = []
    for idx, bin_actual in enumerate(nueva_solucion):
        if bin_actual:
            bins_no_vacios.append(bin_actual)
            capacidades_no_vacias.append(capacidades_actualizadas[idx])

    nueva_solucion = bins_no_vacios
    capacidades_actualizadas = capacidades_no_vacias

    return nueva_solucion, capacidades_actualizadas

def recocido_simulado(objetos: List[int], capacidades_originales: List[int], iteraciones: int, temperatura_inicial: float, tasa_enfriamiento: float, rpfm: List[List[float]], alpha: float, beta: float) -> List[Tuple[List[List[int]], List[int]]]:
    """
    Aplicación de un recocido simulado con el fin de generar soluciones dentro de un vecindario.
    
    Args:
    objetos: List[int]: Lista de objetos que buscamos acomodar
    capacidades_originales: List[int]: Tamaños de contenedor disponibles
    iteraciones: int: Número de iteraciones para el recocido simulado (criterio de paro) 
    temperatura_inicial: float: Temperatura inicial del recocido (probabilidad de aceptar nuevas soluciones) 
    tasa_enfriamiento: float: valor por el que se multiplicará la temperatura en cada iteración (enfriamiento geométrico, 0<tasa_enfriamiento<1) 
    rpfm: List[List[float]]: Matriz de probabilidad.
    alpha: float: Ponderación para el número de contenedores.
    beta: float: Ponderación para el espacio sobrante.
    
    Returns:
        List[Tuple[List[List[int]], List[int]]]: Conjunto de soluciones generadas.
    """
    #Generación de la solución inicial
    solucion_actual, capacidades_usadas = generar_solucion(objetos, capacidades_originales)
    vecinos = []
    temperatura = temperatura_inicial

    for _ in range(iteraciones):
        #Generación de una nueva solución
        vecino = mutacion(solucion_actual, capacidades_usadas, rpfm, capacidades_originales)
        """ vecino = busqueda_local(vecino[0], vecino[1], capacidades_originales) """
        vecinos.append(vecino)

        #Delta representa que tan buena o mala es la nueva solución. Si es menor a cero, significa que es mejor que la mejor solución al momento. En otros caso, junto con la temperatura se calcula la probabilidad e ser aceptada como mejor (aunque no lo sea)
        delta = evaluar_solucion(vecino[0], vecino[1], alpha, beta) - evaluar_solucion(solucion_actual, capacidades_usadas, alpha, beta)

        if delta < 0 or random.random() < math.exp(-delta / temperatura):
            solucion_actual, capacidades_usadas = vecino

        #Aplicar la tasa de enfriamiento
        temperatura *= tasa_enfriamiento

    return vecinos

def busqueda_tabu(objetos: List[int], capacidades: List[int], iteraciones_tabu: int, tabu_tam: int, rpfm: List[List[float]], iteraciones_recocido: int, temperatura_inicial: float, tasa_enfriamiento: float, alpha: float, beta: float, num_objetos: int) -> Tuple[List[List[int]], List[int]]:
    """
    Implementación de una búsqueda tabú para encontrar la mejor solución posible en una determinada cantidad de iteraciones
    
    Args:
    objetos: List[int]: Lista de objetos que buscamos acomodar
    capacidades: List[int]: Tamaños de contenedor disponibles
    iteraciones_tabu: int: Número de iteraciones para la búsqueda tabú (criterio de paro) 
    tabu_tam: int: tamaño de la lista tabú a utilizar
    rpfm: List[List[float]]: Matriz de probabilidad.
    iteraciones_recocido: int: el número de iteraciones para el recocido simulado  (criterio de paro)
    temperatura_inicial: float: Temperatura inicial del recocido (probabilidad de aceptar nuevas soluciones) 
    tasa_enfriamiento: float: valor por el que se multiplicará la temperatura en cada iteración (enfriamiento geométrico, 0<tasa_enfriamiento<1) 
    alpha: float: Ponderación para el número de contenedores.
    beta: float: Ponderación para el espacio sobrante.
    num_objetos: int: Número de objetos diferentes disponibles (cantidad de tamaños de objeto diferentes).
    
    Returns:
        List[Tuple[List[List[int]], List[int]]]: Conjunto de soluciones generadas.
    """
    #Generación de la solución acutal
    solucion_actual, capacidades_usadas = generar_solucion(objetos, capacidades)
    mejor_solucion = solucion_actual[:]
    mejor_capacidades = capacidades_usadas[:]
    lista_tabu = []

    for _ in range(iteraciones_tabu):
        #Generación de vecinos
        vecinos = recocido_simulado(objetos, capacidades, iteraciones_recocido, temperatura_inicial, tasa_enfriamiento, rpfm, alpha, beta)
        vecinos_permitidos = [v for v in vecinos if v not in lista_tabu]

        if vecinos_permitidos:
            #Obtención del mejor vecino
            mejor_vecino = min(vecinos_permitidos, key=lambda x: evaluar_solucion(x[0], x[1], alpha, beta))

            #Si el mejor vecino es mejor que la mejor solución al momento, se acpeta como la neuva mejor solución
            if evaluar_solucion(mejor_vecino[0], mejor_vecino[1], alpha, beta) < evaluar_solucion(mejor_solucion, mejor_capacidades, alpha, beta):
                mejor_solucion, mejor_capacidades = mejor_vecino

            #Añadir le mejor vecino a la lista tabú
            lista_tabu.append(mejor_vecino)
            #Eliminar objetos de la lista tabú si excede el límite de tamaño
            if len(lista_tabu) > tabu_tam:
                lista_tabu.pop(0)
            #Actualizar la matriz de probabilidades
            rpfm = calcular_rpfm(mejor_solucion, num_objetos)
    return mejor_solucion, mejor_capacidades


# Función principal

def main():
    num_aleatorios=[2,3,5,7,11]
    
    capacidades = [10, 15, 20]  # Tamaños de los contenedores disponibles
    objetos = [1,3,5,7,3,8,9,10,2,6,7,3,8,9,10,2,5,7,3,6,2,5,10,2,1,6,7,4]
    iteraciones_tabu1 = [50, 100, 150]
    iteraciones_recocido2 = [60, 80,100]
    tabu_tam2 = [20]
    temperatura_inicial2 = [5.0]
    tasa_enfriamiento2 = [0.95]
    alpha = 0.80 # Peso para el número de contenedores
    beta = 0.20  # Peso para el espacio desperdiciado
    num_objetos = 10
    
    for iteraciones_tabu in iteraciones_tabu1:
        for iteraciones_recocido in iteraciones_recocido2:
            for tabu_tam in tabu_tam2:
                for temperatura_inicial in temperatura_inicial2:
                    for tasa_enfriamiento in tasa_enfriamiento2:
                        for num in num_aleatorios:
                            print("iteraciones_tabu:", iteraciones_tabu, "\titeraciones_recocido:", iteraciones_recocido, "\ttabu_tam",tabu_tam,"\ttemperatura_inicial",temperatura_inicial,"\ttasa_enfriamiento",tasa_enfriamiento,"\tnum",num)
                            random.seed(num)
                            solucion_inicial, capacidades_usadas = generar_solucion(objetos, capacidades)
                            rpfm = calcular_rpfm(solucion_inicial, num_objetos)
                            mejor_solucion, mejor_capacidades = busqueda_tabu(objetos, capacidades, iteraciones_tabu, tabu_tam, rpfm, iteraciones_recocido, temperatura_inicial, tasa_enfriamiento, alpha, beta, num_objetos)

                            print(f"Mejor solución encontrada: {mejor_solucion}")
                            valor_evaluacion = evaluar_solucion(mejor_solucion, mejor_capacidades, alpha, beta)
                            print(f"Evaluación: {valor_evaluacion}\n")
                            """ for i, contenedor in enumerate(mejor_solucion):
                                capacidad = mejor_capacidades[i]
                                espacio_sobrante = capacidad - sum(contenedor)
                                print(f"  Contenedor {i + 1} (Capacidad: {capacidad}): {contenedor}, Espacio sobrante: {espacio_sobrante}")  """
        
if __name__ == "__main__":
    main()