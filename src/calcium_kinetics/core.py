
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

class Procesamiento:
    """
    Carga, valida y limpia un transitorio experimental
    de concentración citosólica de calcio.
    """

    def __init__(self, ruta_archivo):
        # Guardamos la ubicación del archivo
        self.ruta_archivo = ruta_archivo

        # Leemos el archivo CSV
        self.datos = pd.read_csv(self.ruta_archivo)

        # Verificamos que existan las columnas necesarias
        columnas_requeridas = ["time_ms", "fluorescence_AU"]

        for columna in columnas_requeridas:
            if columna not in self.datos.columns:
                raise ValueError(
                    f"El archivo no contiene la columna '{columna}'. "
                    f"Columnas disponibles: {list(self.datos.columns)}"
                )

        # Extraemos los datos experimentales como arreglos NumPy
        self.tiempo = self.datos["time_ms"].to_numpy(dtype=float)
        self.calcio_raw = self.datos["fluorescence_AU"].to_numpy(dtype=float)

        # Inicializamos el arreglo que contendrá el calcio limpio
        self.calcio_limpio = None

        # Validamos los datos al crear el objeto
        self.validar_datos()

    def validar_datos(self):
        """
        Comprueba que los arreglos experimentales sean válidos.
        """

        # 1. Comprobar que ambos arreglos sean unidimensionales
        if self.tiempo.ndim != 1:
            raise ValueError("El arreglo de tiempo debe ser unidimensional (1D).")

        if self.calcio_raw.ndim != 1:
            raise ValueError("El arreglo de calcio debe ser unidimensional (1D).")

        # 2. Comprobar que tengan la misma longitud
        if len(self.tiempo) != len(self.calcio_raw):
            raise ValueError(
                "Los arreglos de tiempo y calcio deben tener la misma longitud."
            )

        # 3. Comprobar que no existan valores NaN o infinitos
        if not np.all(np.isfinite(self.tiempo)):
            raise ValueError("El tiempo contiene valores no finitos.")

        if not np.all(np.isfinite(self.calcio_raw)):
            raise ValueError("El calcio contiene valores no finitos.")

        # 4. Comprobar que el tiempo sea estrictamente creciente
        if np.any(np.diff(self.tiempo) <= 0):
            raise ValueError(
                "Los tiempos deben ser estrictamente crecientes (Delta t > 0)."
            )

        print("Validación de datos completada correctamente.")

    def limpiar_calcio(self):
        """
        Elimina los valores negativos de calcio mediante
        umbralización basal a cero.
        """

        self.calcio_limpio = np.maximum(0.0, self.calcio_raw)

        return self.calcio_limpio


    def construir_interpolador(self):
        """
        Construye la función continua de calcio mediante PCHIP.
        """

        # No podemos interpolar si todavía no limpiamos los datos
        if self.calcio_limpio is None:
            raise ValueError(
                "Primero debes ejecutar limpiar_calcio()."
            )

        # Construimos el interpolador PCHIP
        self._hermite = PchipInterpolator(
            self.tiempo,
            self.calcio_limpio,
            extrapolate=False
        )

        return self._hermite

    def evaluar_calcio(self, t):
        """
        Evalúa la concentración de calcio en un instante o varios.

        Parámetro:
            t: tiempo en ms (float o arreglo NumPy).

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

