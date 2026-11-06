import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator 

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
        self._hermite = None
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

