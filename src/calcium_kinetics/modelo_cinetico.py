import numpy as np
from scipy.integrate import solve_ivp
from .procesamiento import Procesamiento

CONFIGURACION_BUFFERS = {

    # 1. Troponina C: sitios reguladores, unión simple Ca2+
    "Troponina C (Reguladora)": {
        "tipo": "simple",
        "kon": 0.12,
        "koff": 0.12,
        "S_total": 240.0
    },

    # 2. Parvalbúmina: competencia Ca2+ / Mg2+
    "Parvalbumina": {
        "tipo": "competitivo",
        "kon_a": 0.25,
        "koff_a": 0.001,
        "kon_b": 0.000033,
        "koff_b": 0.006,
        "sitios_totales": 380.0,
        "conc_b_libre": 3300.0,
        "conc_a_reposo": 0.06
    },

    # 3. Troponina C estructural TNS: competencia Ca2+ / Mg2+
    "Troponina C (Estructural TNS)": {
        "tipo": "competitivo",
        "kon_a": 0.10,
        "koff_a": 0.001,
        "kon_b": 0.000033,
        "koff_b": 0.006,
        "sitios_totales": 240.0,
        "conc_b_libre": 3300.0,
        "conc_a_reposo": 0.06
    },

    # 4. ATP citosólico: competencia Ca2+ / Mg2+
    "ATP Citosolico": {
        "tipo": "competitivo",
        "kon_a": 0.22,
        "koff_a": 45.0,
        "kon_b": 0.04,
        "koff_b": 4.0,
        "sitios_totales": 5000.0,
        "conc_b_libre": 1000.0,
        "conc_a_reposo": 0.06
    },

    # 5. Indicador fluorescente: unión simple Ca2+
    "Mag-Fluo-4": {
        "tipo": "simple",
        "kon": 0.01,
        "koff": 1.65,
        "S_total": 229.1
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
        if amortiguador is not None:
            if amortiguador not in CONFIGURACION_BUFFERS:
                raise ValueError(
                    f"Amortiguador desconocido: {amortiguador}. "
                    f"Opciones disponibles: {list(CONFIGURACION_BUFFERS)}"  
                ) # esto valida que el amortiguador si este en la lista 
            parametros = CONFIGURACION_BUFFERS[amortiguador]
            kon = parametros["kon"]
            koff = parametros["koff"]
            S_total = parametros["S_total"]
        if kon is None or koff is None or S_total is None:
            raise ValueError(
                "esto sirve no funcionara si no proporcionas kon, koff y S_total, "
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
class ModeloCineticoCompetitivo:
    """
    Modelo cinético general de competencia entre dos especies
    por una población común de sitios de unión.

    La especie A cambia con el tiempo y se obtiene del objeto
    Procesamiento mediante evaluar_calcio(t).

    La especie B se considera constante en esta versión.

    Unidades:
        Concentraciones: µM
        kon: µM⁻¹ ms⁻¹
        koff: ms⁻¹
        Tiempo: ms
    """

    def __init__(
        self,
        procesamiento,
        kon_a,
        koff_a,
        kon_b,
        koff_b,
        sitios_totales,
        conc_b_libre,
        conc_a_reposo=0.06,
        metodo="Radau"
    ):

        # 1. Guardar el procesamiento experimental
        self.procesamiento = procesamiento

        # 2. Guardar las constantes cinéticas
        self.kon_a = float(kon_a)
        self.koff_a = float(koff_a)

        self.kon_b = float(kon_b)
        self.koff_b = float(koff_b)

        # 3. Guardar las concentraciones
        self.sitios_totales = float(sitios_totales)
        self.conc_b_libre = float(conc_b_libre)
        self.conc_a_reposo = float(conc_a_reposo)

        self.metodo = metodo

        # Alias para facilitar la compatibilidad
        # con el código existente del proyecto.
        self.S_total = self.sitios_totales

        # 4. Validar parámetros
        self.validar_parametros()

        # 5. Calcular constantes de disociación
        self.Kd_a = self.koff_a / self.kon_a
        self.Kd_b = self.koff_b / self.kon_b

        # 6. Comprobar que PCHIP esté construido
        if self.procesamiento._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP "
                "en Procesamiento."
            )

        # 7. Calcular condiciones iniciales de equilibrio
        self.calcular_condiciones_iniciales()

        # 8. Inicializar atributos de resultados
        self.solucion = None
        self.tiempo = None

        self.complejo_a = None
        self.complejo_b = None
        self.sitios_libres = None

        # A es el calcio en nuestro proyecto actual.
        # Este alias permite reutilizar código que espera
        # encontrar el atributo calcio_unido.
        self.calcio_unido = None

    def validar_parametros(self):
        """
        Comprueba que los parámetros sean finitos
        y físicamente válidos.
        """

        parametros = {
            "kon_a": self.kon_a,
            "koff_a": self.koff_a,
            "kon_b": self.kon_b,
            "koff_b": self.koff_b,
            "sitios_totales": self.sitios_totales,
            "conc_b_libre": self.conc_b_libre,
            "conc_a_reposo": self.conc_a_reposo
        }

        for nombre, valor in parametros.items():

            if not np.isfinite(valor):
                raise ValueError(
                    f"El parámetro {nombre} debe ser finito."
                )

            if valor < 0:
                raise ValueError(
                    f"El parámetro {nombre} no puede ser negativo."
                )

        # Las constantes de asociación y disociación
        # deben ser estrictamente positivas.
        if self.kon_a <= 0 or self.koff_a <= 0:
            raise ValueError(
                "kon_a y koff_a deben ser mayores que cero."
            )

        if self.kon_b <= 0 or self.koff_b <= 0:
            raise ValueError(
                "kon_b y koff_b deben ser mayores que cero."
            )

        if self.sitios_totales <= 0:
            raise ValueError(
                "sitios_totales debe ser mayor que cero."
            )

        metodos_validos = (
            "Radau", "BDF", "LSODA", "RK45", "DOP853"
        )

        if self.metodo not in metodos_validos:
            raise ValueError(
                f"Método de integración no válido: {self.metodo}"
            )

    def calcular_condiciones_iniciales(self):
        """
        Calcula el equilibrio inicial de las dos especies
        a partir de sus concentraciones libres basales.
        """

        relacion_a = self.conc_a_reposo / self.Kd_a
        relacion_b = self.conc_b_libre / self.Kd_b

        denominador = 1.0 + relacion_a + relacion_b

        # Sitios libres en equilibrio basal
        self.sitios_libres_iniciales = (
            self.sitios_totales / denominador
        )

        # Sitios ocupados por la especie A
        self.complejo_a_inicial = (
            self.sitios_libres_iniciales * relacion_a
        )

        # Sitios ocupados por la especie B
        self.complejo_b_inicial = (
            self.sitios_libres_iniciales * relacion_b
        )

        # Vector de condiciones iniciales para solve_ivp
        self.y0 = [
            self.complejo_a_inicial,
            self.complejo_b_inicial
        ]

        # Verificar conservación de sitios
        suma = (
            self.complejo_a_inicial
            + self.complejo_b_inicial
            + self.sitios_libres_iniciales
        )

        if not np.isclose(
            suma,
            self.sitios_totales,
            rtol=1e-10,
            atol=1e-10
        ):
            raise RuntimeError(
                "Las condiciones iniciales no conservan "
                "los sitios totales."
            )

    def sistema_edos(self, t, y):
        """
        Sistema de dos EDOs acopladas.

        y[0]: complejo formado por la especie A.
        y[1]: complejo formado por la especie B.
        """

        # Extraer las variables de estado
        sa = y[0]
        sb = y[1]

        # Evaluar la concentración variable de A
        a_libre = float(
            self.procesamiento.evaluar_calcio(t)
        )

        if not np.isfinite(a_libre) or a_libre < 0:
            raise ValueError(
                "La concentración de A debe ser finita "
                "y no negativa."
            )

        # Calcular sitios libres compartidos
        s_libre = (
            self.sitios_totales - sa - sb
        )

        # EDO de la especie A
        asociacion_a = (
            self.kon_a * a_libre * s_libre
        )

        disociacion_a = self.koff_a * sa

        dsa_dt = asociacion_a - disociacion_a

        # EDO de la especie B
        asociacion_b = (
            self.kon_b * self.conc_b_libre * s_libre
        )

        disociacion_b = self.koff_b * sb

        dsb_dt = asociacion_b - disociacion_b

        return [dsa_dt, dsb_dt]

    def resolver(self, rtol=1e-6, atol=1e-9):
        """
        Resuelve las dos EDOs acopladas mediante solve_ivp.
        """

        # Intervalo temporal experimental
        t_inicio = self.procesamiento.tiempo[0]
        t_final = self.procesamiento.tiempo[-1]

        # Tiempos donde se evaluará la solución
        tiempos_evaluacion = self.procesamiento.tiempo

        # Integrar el sistema
        solucion = solve_ivp(
            fun=self.sistema_edos,
            t_span=(t_inicio, t_final),
            y0=self.y0,
            t_eval=tiempos_evaluacion,
            method=self.metodo,
            rtol=rtol,
            atol=atol
        )

        # Comprobar integración correcta
        if not solucion.success:
            raise RuntimeError(
                f"Error al resolver el sistema competitivo: "
                f"{solucion.message}"
            )

        if not np.all(np.isfinite(solucion.y)):
            raise RuntimeError(
                "La solución contiene valores no finitos."
            )

        # Guardar los perfiles temporales
        self.complejo_a = solucion.y[0]
        self.complejo_b = solucion.y[1]

        # En el proyecto actual, A representa calcio.
        self.calcio_unido = self.complejo_a

        self.tiempo = solucion.t

        # Calcular sitios libres en cada instante
        self.sitios_libres = (
            self.sitios_totales
            - self.complejo_a
            - self.complejo_b
        )

        # Comprobar límites físicos
        tolerancia = 1e-7

        if np.any(self.complejo_a < -tolerancia):
            raise RuntimeError(
                "La concentración del complejo A es negativa."
            )

        if np.any(self.complejo_b < -tolerancia):
            raise RuntimeError(
                "La concentración del complejo B es negativa."
            )

        if np.any(self.sitios_libres < -tolerancia):
            raise RuntimeError(
                "La ocupación conjunta supera los sitios totales."
            )

        # Guardar objeto completo de solución
        self.solucion = solucion

        print(
            "Integración cinética competitiva "
            "completada correctamente."
        )
        print(f"Método numérico: {solucion.message}")
        print(f"Número de evaluaciones: {solucion.nfev}")

        return solucion
    
    def simular_completo(self, nombres=("complejo_a", "complejo_b")):
        """
        Devuelve las trayectorias de ambos complejos y de los sitios libres.

        nombres permite asignar nombres específicos a los dos complejos.
        Por ejemplo: ("CaPV", "MgPV").
        """

        if len(nombres) != 2:
            raise ValueError("Debes proporcionar exactamente dos nombres.")

        # Resolver la integración si todavía no se ha ejecutado.
        if self.solucion is None:
            self.resolver()

        return {
            nombres[0]: self.complejo_a.copy(),
            nombres[1]: self.complejo_b.copy(),
            "sitios_libres": self.sitios_libres.copy(),
            "tiempo": self.tiempo.copy()
        }

def crear_diccionario_buffers(procesamiento):
    """
    Construye todos los modelos cinéticos a partir de las
    configuraciones y de una instancia de Procesamiento.
    """

    buffers = {}

    for nombre, parametros in CONFIGURACION_BUFFERS.items():

        tipo = parametros["tipo"]

        if tipo == "simple":

            buffers[nombre] = ModeloCinetico(
                procesamiento=procesamiento,
                kon=parametros["kon"],
                koff=parametros["koff"],
                S_total=parametros["S_total"]
            )

        elif tipo == "competitivo":

            buffers[nombre] = ModeloCineticoCompetitivo(
                procesamiento=procesamiento,
                kon_a=parametros["kon_a"],
                koff_a=parametros["koff_a"],
                kon_b=parametros["kon_b"],
                koff_b=parametros["koff_b"],
                sitios_totales=parametros["sitios_totales"],
                conc_b_libre=parametros["conc_b_libre"],
                conc_a_reposo=parametros["conc_a_reposo"]
            )

        else:
            raise ValueError(
                f"Tipo de amortiguador desconocido: {tipo}"
            )

    return buffers