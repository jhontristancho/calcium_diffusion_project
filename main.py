import numpy as np

from calcium_kinetics import (
    Procesamiento,
    ModeloCinetico
)

# ==========================================
# FASE 1: PROCESAMIENTO EXPERIMENTAL
# ==========================================

procesamiento = Procesamiento(
    "data/raw/transientph7-33.csv"
)

procesamiento.limpiar_calcio()
procesamiento.construir_interpolador()

# ==========================================
# FASE 2: MODELO CINÉTICO DE TROPONINA C
# ==========================================

troponina = ModeloCinetico(
    procesamiento=procesamiento,
    kon=0.12,
    koff=0.12,
    S_total=240.0,
    SCa_inicial=0.0
)

resultado = troponina.resolver()

# ==========================================
# RESULTADOS
# ==========================================

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