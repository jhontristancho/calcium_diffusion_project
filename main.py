
from pathlib import Path
import sys

# Permitir importar el paquete desde src/
DIRECTORIO_PROYECTO = Path(__file__).resolve().parent
DIRECTORIO_SRC = DIRECTORIO_PROYECTO / "src"

sys.path.insert(0, str(DIRECTORIO_SRC))

from calcium_kinetics.procesamiento import Procesamiento
from calcium_kinetics.modelo_cinetico import crear_diccionario_buffers
from calcium_kinetics.recapturacion import Recapturacion
from calcium_kinetics.analizador import Analizador


def main():

    # 1. Procesar los datos experimentales
    ruta_datos = (
        DIRECTORIO_PROYECTO
        / "data"
        / "raw"
        / "transientph7-33.csv"
    )

    procesamiento = Procesamiento(str(ruta_datos))

    # 2. Construir automáticamente todos los amortiguadores
    buffers = crear_diccionario_buffers(procesamiento)

    # 3. Construir SERCA con los parámetros predeterminados
    flujos = {
        "SERCA": Recapturacion()
    }

    # 4. Crear el analizador
    analizador = Analizador(
        procesamiento=procesamiento,
        buffers=buffers,
        flujos=flujos
    )

    # 5. Ejecutar la simulación completa
    resultados = analizador.simular_sistema_completo()

    # 6. Mostrar un resumen
    print("\n========== RESUMEN DE SIMULACIÓN ==========")

    print(f"Buffers simulados: {len(resultados['buffers'])}")
    print(f"Flujos evaluados: {len(resultados['flujos'])}")

    for nombre, datos in resultados["buffers"].items():

        modelo = buffers[nombre]

        print(f"\n{nombre}")
        print(
            f"Calcio unido final: "
            f"{datos['calcio_unido'][-1]:.6f} µM"
        )

        if hasattr(modelo, "complejo_b") and modelo.complejo_b is not None:

            print(
                f"Magnesio unido final: "
                f"{modelo.complejo_b[-1]:.6f} µM"
            )

            print(
                f"Sitios libres finales: "
                f"{modelo.sitios_libres[-1]:.6f} µM"
            )
    for nombre, datos in resultados["flujos"].items():
        print(
            f"{nombre}: "
            f"flujo neto máximo = "
            f"{max(datos['flujo_neto']):.6f} µM/ms"
        )

    # 7. Generar reporte gráfico
    analizador.graficar_reporte_completo()



main()
