import numpy as np

from calcium_kinetics.procesamiento import Procesamiento
from calcium_kinetics.modelo_cinetico import ModeloCinetico, PARAMETROS_AMORTIGUADORES
from calcium_kinetics.recapturacion import Recapturacion
from calcium_kinetics.analizador import Analizador

# 1. Cargar datos
procesamiento = Procesamiento("data/raw/transientph7-33.csv")

# 2. Instanciar amortiguadores pasivos
buffers = {
    nombre: ModeloCinetico(procesamiento=procesamiento, amortiguador=nombre)
    for nombre in PARAMETROS_AMORTIGUADORES
}

# 3. Instanciar transporte activo (SERCA1a)
flujos = {
    "SERCA1a": Recapturacion(Vmax=2.5, Km=0.44, n_hill=2.1, Ca_reposo=0.06)
}

# 4. Instanciar el Analizador
print("\n========== ETAPA 4: ANALIZADOR ==========")

analizador = Analizador(
    procesamiento=procesamiento,
    buffers=amortiguadores,
    flujos={"SERCA": serca}
)

print("Analizador conectado con Procesamiento, Buffers y SERCA correctamente.")

# Ejecución automática e inicio del menú interactivo
analizador.menu_interactivo()

# 5. Ejecutar primero la prueba automatizada
print("--- PRUEBA AUTOMÁTICA DE SIMULACIÓN ---")
resultados = analizador.simular_sistema_completo()

print(f"Puntos de tiempo procesados: {len(resultados['tiempo'])}")
print(f"Buffers integrados: {list(resultados['buffers'].keys())}")
print(f"Flujos evaluados   : {list(resultados['flujos'].keys())}")

# 6. Abrir menú interactivo para inspección paso a paso
analizador.menu_interactivo()