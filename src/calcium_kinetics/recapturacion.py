
import numpy as np


class Recapturacion:
    """
    Modelo de transporte activo de calcio mediante SERCA.

    Implementa una cinética de Hill y una fuga pasiva constante
    que equilibra el flujo de la bomba en el calcio de reposo.

    Unidades:
        Ca_libre: µM
        Vmax: µM/ms
        Km: µM
        J_bomba y J_neto: µM/ms
    """

    def __init__(
        self,
        Vmax: float = 2.5,
        Km: float = 0.44,
        n_hill: float = 2.1,
        Ca_reposo: float = 0.06
    ):

        # Convertir los parámetros a números reales
        self.Vmax = float(Vmax)
        self.Km = float(Km)
        self.n_hill = float(n_hill)
        self.Ca_reposo = float(Ca_reposo)

        # Validar que todos los parámetros sean finitos
        parametros = {
            "Vmax": self.Vmax,
            "Km": self.Km,
            "n_hill": self.n_hill,
            "Ca_reposo": self.Ca_reposo
        }

        for nombre, valor in parametros.items():
            if not np.isfinite(valor):
                raise ValueError(
                    f"El parámetro {nombre} debe ser finito."
                )

        # Validar parámetros positivos
        if self.Vmax <= 0 or self.Km <= 0 or self.n_hill <= 0:
            raise ValueError(
                "Vmax, Km y n_hill deben ser estrictamente positivos."
            )

        if self.Ca_reposo < 0:
            raise ValueError(
                "Ca_reposo no puede ser negativo."
            )

        # Precalcular potencias
        self._Km_pow = self.Km ** self.n_hill
        self._Ca_reposo_pow = self.Ca_reposo ** self.n_hill

        # Calcular fuga pasiva para equilibrar el reposo
        self.J_leak = self._calcular_fuga_reposo()

    def _calcular_fuga_reposo(self) -> float:
        """
        Calcula J_leak para que J_neto(Ca_reposo) = 0.
        """

        return (
            self.Vmax * self._Ca_reposo_pow
        ) / (
            self._Ca_reposo_pow + self._Km_pow
        )

    def evaluar_flujo_bomba(
        self,
        Ca_libre: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calcula la tasa bruta de transporte activo SERCA.
        """

        Ca_arr = np.asarray(Ca_libre, dtype=float)

        if not np.all(np.isfinite(Ca_arr)):
            raise ValueError(
                "El calcio libre debe contener valores finitos."
            )

        if np.any(Ca_arr < 0):
            raise ValueError(
                "La concentración de calcio no puede ser negativa."
            )

        ca_pow = Ca_arr ** self.n_hill

        j_bomba = (
            self.Vmax * ca_pow
        ) / (
            ca_pow + self._Km_pow
        )

        # Devolver float para escalares y array para vectores
        if j_bomba.ndim == 0:
            return float(j_bomba)

        return j_bomba

    def evaluar_flujo_neto(
        self,
        Ca_libre: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calcula el flujo neto de recapturación:

        J_neto = J_bomba - J_leak
        """

        j_bomba = self.evaluar_flujo_bomba(Ca_libre)

        return j_bomba - self.J_leak
