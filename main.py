import numpy as np

from calcium_kinetics.procesamiento import Procesamiento
from calcium_kinetics.modelo_cinetico import (
    ModeloCinetico,
    PARAMETROS_AMORTIGUADORES
)
from calcium_kinetics.recapturacion import Recapturacion
from calcium_kinetics.analizador import Analizador


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

RUTA_DATOS = "data/raw/transientph7-33.csv"


# ============================================================
# ETAPA 1. PROCESAMIENTO EXPERIMENTAL
# ============================================================

print("\n========== ETAPA 1: PROCESAMIENTO ==========")

procesamiento = Procesamiento(RUTA_DATOS)

assert procesamiento.calcio_limpio is not None
assert procesamiento._hermite is not None

print(f"Número de datos: {len(procesamiento.tiempo)}")
print(f"Calcio máximo: {np.max(procesamiento.calcio_limpio):.6f} µM")

auc = procesamiento.area_bajo_curva(
    procesamiento.tiempo[0],
    procesamiento.tiempo[-1]
)

print(f"Área bajo la curva: {auc:.6f} µM·ms")
print("Procesamiento verificado correctamente.")


# ============================================================
# ETAPA 2. MODELOS CINÉTICOS DE AMORTIGUADORES (DEFINICIÓN)
# ============================================================

print("\n========== ETAPA 2: AMORTIGUADORES ==========")

# Se declara el diccionario 'amortiguadores' en el ámbito global de main.py
amortiguadores = {}

for nombre in PARAMETROS_AMORTIGUADORES:
    print(f"\nResolviendo amortiguador: {nombre}")

    modelo = ModeloCinetico(
        procesamiento=procesamiento,
        amortiguador=nombre
    )

    solucion = modelo.resolver()

    # Verificar límites físicos
    assert np.all(modelo.calcio_unido >= -1e-7)
    assert np.all(modelo.calcio_unido <= modelo.S_total + 1e-7)

    # Almacenar en el diccionario
    amortiguadores[nombre] = modelo

    print(f"Calcio unido final: {modelo.calcio_unido[-1]:.6f} µM")
    print(f"Calcio unido máximo: {np.max(modelo.calcio_unido):.6f} µM")

print("\nTodos los amortiguadores fueron resueltos correctamente.")


# ============================================================
# ETAPA 3. TRANSPORTE ACTIVO SERCA Y HOMEOSTASIS (DEFINICIÓN)
# ============================================================

print("\n========== ETAPA 3: RECAPTURACIÓN SERCA ==========")

# Se declara la variable 'serca' en el ámbito global de main.py
serca = Recapturacion(
    Vmax=2.5,
    Km=0.44,
    n_hill=2.1,
    Ca_reposo=0.06
)

print(f"Vmax: {serca.Vmax} µM/ms")
print(f"Km: {serca.Km} µM")
print(f"Coeficiente Hill: {serca.n_hill}")
print(f"Calcio de reposo: {serca.Ca_reposo} µM")
print(f"Fuga pasiva: {serca.J_leak:.10f} µM/ms")

# Pruebas de consistencia de la bomba
flujo_reposo = serca.evaluar_flujo_neto(serca.Ca_reposo)
assert np.isclose(flujo_reposo, 0.0, atol=1e-12)

flujo_km = serca.evaluar_flujo_bomba(serca.Km)
assert np.isclose(flujo_km, serca.Vmax / 2.0)

print("Modelo SERCA verificado correctamente.")


# ============================================================
# ETAPA 4. INTEGRACIÓN Y MENÚ INTERACTIVO CON EL ANALIZADOR
# ============================================================

print("\n========== ETAPA 4: ANALIZADOR ==========")

# Ahora 'procesamiento', 'amortiguadores' y 'serca' están 100% definidos
analizador = Analizador(
    procesamiento=procesamiento,
    buffers=amortiguadores,
    flujos={"SERCA": serca}
)

print("Analizador conectado con Procesamiento, Buffers y SERCA correctamente.")

# Iniciar simulación e interfaz interactiva
analizador.menu_interactivo()