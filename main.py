
from calcium_kinetics.core import Procesamiento


procesamiento = Procesamiento(
    "data/raw/transientph7-33.csv"
)

# 2. Limpiar calcio
procesamiento.limpiar_calcio()

# 3. Construir interpolador
procesamiento.construir_interpolador()

# 4. Evaluar calcio en 10 ms
calcio_10ms = procesamiento.evaluar_calcio(10.0)

# 5. Calcular derivada en 10 ms
derivada_10ms = procesamiento.derivada_interpolador(10.0)

# 6. Calcular exposición acumulada
auc = procesamiento.area_bajo_curva(
    procesamiento.tiempo[0],
    procesamiento.tiempo[-1]
)

print("Calcio a 10 ms:", calcio_10ms, "µM")
print("Derivada a 10 ms:", derivada_10ms, "µM/ms")
print("Exposición acumulada:", auc, "µM·ms")
