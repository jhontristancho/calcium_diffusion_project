import numpy as np
from scipy.integrate import solve_ivp
from .procesamiento import Procesamiento

# Parámetros cinéticos de los amortiguadores.
# Las unidades de tiempo del modelo son milisegundos (ms).

PARAMETROS_AMORTIGUADORES = {
    "Troponina C": {
        "kon": 0.12,
        "koff": 0.12,
        "S_total": 240.0
    },

    "Parvalbumina": {
        "kon": 0.25,
        "koff": 0.001,
        "S_total": 380.0
    },

    "Mag-Fluo-4": {
        "kon": 0.01,
        "koff": 1.65,
        "S_total": 229.1,
        "Kd_reportado": 1.65e5
    }
}
class ModeloCinetico:

    """
    modelo de union del calcio a un amortiguador proteico.
    Unidades:
        Ca y SCa: µM
        kon: µM⁻¹ ms⁻¹
        koff: ms⁻¹
        tiempo: ms
    """


    def __init__(
        self,
        procesamiento,
        kon=None,
        koff=None,
        S_total=None,
        SCa_inicial=0.0,
        amortiguador=None
    ):
        self.procesamiento = procesamiento

        # Cargar parámetros automáticamente si se indica un amortiguador.
        if amortiguador is not None:
            if amortiguador not in PARAMETROS_AMORTIGUADORES:
                raise ValueError(
                    f"Amortiguador desconocido: {amortiguador}. "
                    f"Opciones disponibles: {list(PARAMETROS_AMORTIGUADORES)}"
                )

            parametros = PARAMETROS_AMORTIGUADORES[amortiguador]

            kon = parametros["kon"]
            koff = parametros["koff"]
            S_total = parametros["S_total"]

        # Comprobar que se hayan proporcionado todos los parámetros.
        if kon is None or koff is None or S_total is None:
            raise ValueError(
                "Debes proporcionar kon, koff y S_total, "
                "o indicar el nombre del amortiguador."
            )

        self.kon = float(kon)
        self.koff = float(koff)
        self.S_total = float(S_total)
        self.SCa_inicial = float(SCa_inicial)

        self.amortiguador = amortiguador

        self.validar_parametros()

        if self.procesamiento._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP "
                "en Procesamiento."
            )

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