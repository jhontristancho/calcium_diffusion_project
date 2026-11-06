import numpy as np
from scipy.integrate import solve_ivp
from .procesamiento import Procesamiento
class ModeloCinetico:
    """
    Modelo de unión reversible de calcio a un amortiguador proteico.

    Resuelve:
    d[SCa]/dt = kon * Ca(t) * (S_total - [SCa]) - koff * [SCa]

    Unidades:
        Ca y SCa: µM
        kon: µM⁻¹ ms⁻¹
        koff: ms⁻¹
        tiempo: ms
    """

    def __init__(
        self,
        procesamiento,
        kon,
        koff,
        S_total,
        SCa_inicial=0.0
    ):

        # Guardar la referencia al procesamiento experimental
        self.procesamiento = procesamiento

        # Guardar los parámetros cinéticos
        self.kon = float(kon)
        self.koff = float(koff)
        self.S_total = float(S_total)
        self.SCa_inicial = float(SCa_inicial)

        # Validar los parámetros
        self.validar_parametros()

        # Comprobar que exista el interpolador de calcio
        if self.procesamiento._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP "
                "en Procesamiento."
            )

        # Inicializar resultados
        self.solucion = None
        self.tiempo = None
        self.calcio_unido = None

    def validar_parametros(self):
        """
        Verifica que los parámetros cinéticos sean físicamente válidos.
        """

        parametros = {
            "kon": self.kon,
            "koff": self.koff,
            "S_total": self.S_total,
            "SCa_inicial": self.SCa_inicial
        }

        # Comprobar valores finitos y no negativos
        for nombre, valor in parametros.items():

            if not np.isfinite(valor):
                raise ValueError(
                    f"El parámetro {nombre} debe ser finito."
                )

            if valor < 0:
                raise ValueError(
                    f"El parámetro {nombre} no puede ser negativo."
                )

        # El complejo inicial no puede superar los sitios disponibles
        if self.SCa_inicial > self.S_total:
            raise ValueError(
                "SCa_inicial no puede superar S_total."
            )

    def ecuacion_accion_masas(self, t, y):
        """
        Define la EDO de asociación y disociación del calcio.
        """

        # Concentración de calcio libre en el instante t
        calcio_libre = float(
            self.procesamiento.evaluar_calcio(t)
        )

        # Concentración actual del complejo proteína-calcio
        SCa = y[0]

        # Sitios de unión disponibles
        sitios_libres = self.S_total - SCa

        # Flujo de asociación
        asociacion = (
            self.kon
            * calcio_libre
            * sitios_libres
        )

        # Flujo de disociación
        disociacion = self.koff * SCa

        # Balance neto
        dSCa_dt = asociacion - disociacion

        return [dSCa_dt]

    def resolver(self, rtol=1e-6, atol=1e-9):
        """
        Resuelve la EDO utilizando el método implícito Radau.
        """

        # Intervalo temporal experimental
        t_inicio = self.procesamiento.tiempo[0]
        t_final = self.procesamiento.tiempo[-1]

        # Evaluar la solución en los tiempos experimentales
        tiempos_evaluacion = self.procesamiento.tiempo

        # Resolver la EDO
        solucion = solve_ivp(
            fun=self.ecuacion_accion_masas,
            t_span=(t_inicio, t_final),
            y0=[self.SCa_inicial],
            t_eval=tiempos_evaluacion,
            method="Radau",
            rtol=rtol,
            atol=atol
        )

        # Comprobar que la integración haya terminado correctamente
        if not solucion.success:
            raise RuntimeError(
                f"Error al resolver la EDO: {solucion.message}"
            )

        # Guardar los resultados
        self.solucion = solucion
        self.tiempo = solucion.t
        self.calcio_unido = solucion.y[0]

        # Comprobar los límites físicos de la solución
        tolerancia = 1e-7

        if np.any(self.calcio_unido < -tolerancia):
            raise RuntimeError(
                "La solución presenta concentraciones negativas "
                "de complejo proteína-calcio."
            )

        if np.any(self.calcio_unido > self.S_total + tolerancia):
            raise RuntimeError(
                "La solución supera la concentración total "
                "de sitios de unión."
            )

        print("Integración cinética completada correctamente.")
        print(f"Método numérico: {solucion.message}")
        print(f"Número de evaluaciones: {solucion.nfev}")

        return solucion