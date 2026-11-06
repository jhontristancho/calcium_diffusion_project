
import numpy as np
import matplotlib.pyplot as plt


class Analizador:
    """
    Clase responsable de la visualización y análisis
    de los datos experimentales de calcio.
    """

    def __init__(self, procesamiento):

        self.procesamiento = procesamiento

        # Comprobar que exista el interpolador PCHIP
        if self.procesamiento._hermite is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP "
                "en Procesamiento."
            )

    def graficar_interpolador(self):
        """
        Grafica los datos experimentales y la interpolación PCHIP.
        """

        proc = self.procesamiento

        tiempo_continuo = np.linspace(
            proc.tiempo[0],
            proc.tiempo[-1],
            1000
        )

        calcio_continuo = proc.evaluar_calcio(tiempo_continuo)

        plt.figure(figsize=(10, 6))

        plt.scatter(
            proc.tiempo,
            proc.calcio_limpio,
            color="blue",
            label="Datos experimentales",
            s=25,
            zorder=3
        )

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

        proc = self.procesamiento

        tiempo_continuo = np.linspace(
            proc.tiempo[0],
            proc.tiempo[-1],
            1000
        )

        derivada = proc.derivada_interpolador(tiempo_continuo)

        plt.figure(figsize=(10, 6))

        plt.plot(
            tiempo_continuo,
            derivada,
            color="darkorange",
            linewidth=2,
            label="Derivada PCHIP"
        )

        plt.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1
        )

        if t_destacado is not None:

            if not np.isfinite(t_destacado):
                raise ValueError(
                    "El tiempo destacado debe ser finito."
                )

            if not proc.tiempo[0] <= t_destacado <= proc.tiempo[-1]:
                raise ValueError(
                    "El tiempo destacado está fuera del intervalo."
                )

            valor_derivada = proc.derivada_interpolador(t_destacado)

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

        proc = self.procesamiento

        # Calcular el área mediante Procesamiento
        auc = proc.area_bajo_curva(t0, tf)

        tiempo_integracion = np.linspace(t0, tf, 500)

        calcio_integracion = proc.evaluar_calcio(
            tiempo_integracion
        )

        tiempo_completo = np.linspace(
            proc.tiempo[0],
            proc.tiempo[-1],
            1000
        )

        calcio_completo = proc.evaluar_calcio(tiempo_completo)

        plt.figure(figsize=(10, 6))

        plt.plot(
            proc.tiempo,
            proc.calcio_limpio,
            color="blue",
            alpha=0.5,
            label="Datos experimentales"
        )

        plt.plot(
            tiempo_completo,
            calcio_completo,
            color="red",
            linewidth=2,
            label="Interpolación PCHIP"
        )

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