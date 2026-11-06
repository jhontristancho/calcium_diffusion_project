import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt 

class Procesamiento:
    """
    clase que procesa los datos experimentales de calcio libre
    """
#todo esta basado en POO, el objetivo de mi proyecto es tener una base solida para poder hacer mayores implementaciones en el futuro
    def __init__(self, ruta_archivo):
        self.ruta_archivo = ruta_archivo
        self.datos = pd.read_csv(self.ruta_archivo)#esto lee el archivo
        columnas_requeridas = ["time_ms", "fluorescence_AU"]
        for columna in columnas_requeridas:
            if columna not in self.datos.columns:
                raise ValueError(
                    f"El archivo no contiene la columna '{columna}'. "
                    f"Columnas disponibles: {list(self.datos.columns)}"
                )

        self.tiempo = self.datos["time_ms"].to_numpy(dtype=float)#convertimos a numpy array
        self.calcio_raw = self.datos["fluorescence_AU"].to_numpy(dtype=float)
        self.calcio_limpio = None
        self.validar_datos()
    def validar_datos(self):
        """
        Comprueba que los arreglos experimentales sean válidos.
        """
        #Comprobar que ambos arreglos sean unidimensionales
        if self.tiempo.ndim != 1:
            raise ValueError("El arreglo de tiempo debe ser unidimensional (1D).")
        if self.calcio_raw.ndim != 1:
            raise ValueError("El arreglo de calcio debe ser unidimensional (1D).")
        #Comprobar que tengan la misma longitud
        if len(self.tiempo) != len(self.calcio_raw):
            raise ValueError(
                "Los arreglos de tiempo y calcio deben tener la misma longitud."
            )
        # Comprobar que no existan valores NaN o infinitos
        if not np.all(np.isfinite(self.tiempo)):
            raise ValueError("El tiempo contiene valores no finitos.")

        if not np.all(np.isfinite(self.calcio_raw)):
            raise ValueError("El calcio contiene valores no finitos.")

        #Comprobar que el tiempo sea estrictamente creciente
        if np.any(np.diff(self.tiempo) <= 0):
            raise ValueError(
                "Los tiempos deben ser estrictamente crecientes (Delta t > 0)."
            )
        print("Validación de datos completada correctamente.")
    def limpiar_calcio(self):
        """
        Elimina los valores negativos de calcio y los reemplaza por cero.
        """
        self.calcio_limpio = np.maximum(0.0, self.calcio_raw)
#Los valores negativos se sustituyen por cero porque los interpretamos como ruido o artefactos de la corrección de fondo.
        return self.calcio_limpio
    def construir_interpolador(self):
        """
        Construye la funció mediante PCHIP.
        """

        if self.calcio_limpio is None:
            raise ValueError(
                "Primero debes ejecutar limpiar_calcio()."
            )
        self._hermite = PchipInterpolator(
            self.tiempo,
            self.calcio_limpio,
            extrapolate=False
        )
        return self._hermite
    def evaluar_calcio(self, t):
        """
        evalua la concentración de calcio en un instante o varios.

        Parámetro:
            t: tiempo en ms.

        Retorna:
            Concentración de calcio en µM.
        """

        if self._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP."
            )

        t = np.asarray(t, dtype=float)

        if not np.all(np.isfinite(t)):
            raise ValueError("El tiempo de evaluación debe ser finito.")

        if np.any(t < self.tiempo[0]) or np.any(t > self.tiempo[-1]):
            raise ValueError(
                "El tiempo solicitado está fuera del intervalo experimental."
            )

        return self._hermite(t)
    def derivada_interpolador(self, t):
        """
        Calcula la derivada analítica del interpolador PCHIP.

        Retorna:
            Tasa de cambio del calcio en µM/ms.
        """

        if self._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP."
            )

        t = np.asarray(t, dtype=float)

        if not np.all(np.isfinite(t)):
            raise ValueError("El tiempo de evaluación debe ser finito.")

        if np.any(t < self.tiempo[0]) or np.any(t > self.tiempo[-1]):
            raise ValueError(
                "El tiempo solicitado está fuera del intervalo experimental."
            )

        # Derivada analítica del interpolador
        derivada = self._hermite.derivative()

        return derivada(t)
    def area_bajo_curva(self, t0, tf):
        """
        Calcula la exposición acumulada de calcio libre (AUC).

        Retorna:
            Área en µM·ms.
        """

        if self._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP."
            )

        if not np.isfinite(t0) or not np.isfinite(tf):
            raise ValueError("Los límites de integración deben ser finitos.")

        if t0 >= tf:
            raise ValueError("t0 debe ser menor que tf.")

        if t0 < self.tiempo[0] or tf > self.tiempo[-1]:
            raise ValueError(
                "Los límites de integración deben estar dentro "
                "del intervalo experimental."
            )

        # Integración analítica del interpolador por tramos
        auc = self._hermite.integrate(t0, tf)

        return float(auc)  
    def graficar_interpolador(self):
        """
        Grafica los datos experimentales y la interpolación PCHIP.
        """

        if self._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP."
            )

        # Generamos 1000 puntos para representar la curva continua
        tiempo_continuo = np.linspace(
            self.tiempo[0],
            self.tiempo[-1],
            1000
        )

        # Evaluamos el interpolador en esos puntos
        calcio_continuo = self._hermite(tiempo_continuo)

        # Creamos la figura
        plt.figure(figsize=(10, 6))

        # Datos experimentales limpios
        plt.scatter(
            self.tiempo,
            self.calcio_limpio,
            color="blue",
            label="Datos experimentales",
            s=25,
            zorder=3
        )

        # Curva interpolada
        plt.plot(
            tiempo_continuo,
            calcio_continuo,
            color="red",
            linewidth=2,
            label="Interpolación PCHIP"
        )

        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Concentración de calcio (µM)")
        plt.title("Interpolación PCHIP del transitorio de calcio")

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show() 
    def graficar_derivada(self, t_destacado=None):
        """
     Grafica la derivada del interpolador PCHIP.

    Parámetro:
        t_destacado: instante opcional que queremos destacar (ms).
        """

        if self._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP."
            )

        # Generamos una malla temporal continua
        tiempo_continuo = np.linspace(
            self.tiempo[0],
            self.tiempo[-1],
            1000
        )

        # Calculamos la derivada analítica
        derivada = self._hermite.derivative()(tiempo_continuo)

        plt.figure(figsize=(10, 6))

        plt.plot(
            tiempo_continuo,
            derivada,
            color="darkorange",
            linewidth=2,
            label="Derivada PCHIP"
        )

        # Línea horizontal que representa derivada igual a cero
        plt.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1
        )

        # Destacamos un instante específico si se solicita
        if t_destacado is not None:

            if not np.isfinite(t_destacado):
                raise ValueError("El tiempo destacado debe ser finito.")

            if not self.tiempo[0] <= t_destacado <= self.tiempo[-1]:
                raise ValueError("El tiempo destacado está fuera del intervalo.")

            valor_derivada = self._hermite.derivative()(t_destacado)

            plt.scatter(
                t_destacado,
                valor_derivada,
                color="red",
                s=70,
                zorder=3,
                label=f"Derivada a {t_destacado} ms"
            )

        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Variación del calcio (µM/ms)")
        plt.title("Derivada temporal del calcio citosólico")

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    def graficar_integracion(self, t0, tf):
        
        """
        Grafica el área bajo la curva de calcio entre t0 y tf.
        """

        if self._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP."
            )

        # Calculamos el área mediante el método que ya tenemos
        auc = self.area_bajo_curva(t0, tf)

        # Generamos puntos dentro del intervalo de integración
        tiempo_integracion = np.linspace(t0, tf, 500)

        calcio_integracion = self._hermite(tiempo_integracion)

        plt.figure(figsize=(10, 6))

        # Curva completa
        plt.plot(
            self.tiempo,
            self.calcio_limpio,
            color="blue",
            alpha=0.5,
            label="Datos experimentales"
        )

        # Curva interpolada
        tiempo_completo = np.linspace(
            self.tiempo[0],
            self.tiempo[-1],
            1000
        )

        plt.plot(
            tiempo_completo,
            self._hermite(tiempo_completo),
            color="red",
            linewidth=2,
            label="Interpolación PCHIP"
        )

        # Área bajo la curva en el intervalo seleccionado
        plt.fill_between(
            tiempo_integracion,
            calcio_integracion,
            0,
            color="orange",
            alpha=0.4,
            label="Área integrada"
        )

        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Concentración de calcio (µM)")
        plt.title("Exposición acumulada de calcio citosólico")

        plt.legend()
        plt.grid(True, alpha=0.3)

        # Mostramos el resultado numérico
        plt.text(
            0.03,
            0.95,
            f"AUC = {auc:.6f} µM·ms",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8)
        )

        plt.tight_layout()
        plt.show()

        return auc

