import numpy as np

from calcium_kinetics import (
    Procesamiento,
    ModeloCinetico,
    Analizador
)

procesamiento = Procesamiento(
    "data/raw/transientph7-33.csv"
)

procesamiento.limpiar_calcio()
procesamiento.construir_interpolador()


troponina = ModeloCinetico(
    procesamiento=procesamiento,
    kon=0.12,
    koff=0.12,
    S_total=240.0,
    SCa_inicial=0.0
)

resultado = troponina.resolver()

print("\n--- RESULTADOS DE TROPONINA C ---")

print(
    f"Calcio unido inicial: "
    f"{troponina.calcio_unido[0]:.6f} µM"
)

print(
    f"Calcio unido final: "
    f"{troponina.calcio_unido[-1]:.6f} µM"
)

print(
    f"Máximo calcio unido: "
    f"{np.max(troponina.calcio_unido):.6f} µM"
)

analizador = Analizador(procesamiento)
# Graficar interpolación PCHIP
analizador.graficar_interpolador()

# Graficar derivada
analizador.graficar_derivada(t_destacado=10.0)

# Graficar área bajo la curva
auc = analizador.graficar_integracion(
    t0=0.0,
    tf=56.2
)

print(f"AUC del calcio libre: {auc:.6f} µM·ms")