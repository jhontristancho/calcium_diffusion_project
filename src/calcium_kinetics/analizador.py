import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from calcium_kinetics.procesamiento import Procesamiento
from calcium_kinetics.modelo_cinetico import (
    ModeloCinetico,
    ModeloCineticoCompetitivo,
    crear_diccionario_buffers,
)
from calcium_kinetics.recapturacion import Recapturacion


class AnalizadorError(Exception):
    """exceprcion para errores"""
    pass


class Analizador:
    """
    Orquesta toda la simulacion 
    """

    def __init__(
        self,
        procesamiento: Procesamiento,
        buffers: dict[str, ModeloCinetico | ModeloCineticoCompetitivo] | None = None,
        flujos: dict[str, Recapturacion] | None = None,
        pH: float | None = None,
        nombre_archivo: str = "Experimento",
    ):
        if not isinstance(procesamiento, Procesamiento):
            raise TypeError("procesamiento debe ser una instancia válida de Procesamiento.")
        if getattr(procesamiento, "_hermite", None) is None:
            raise ValueError("El objeto Procesamiento no tiene un interpolador PCHIP construido.")
        self.procesamiento = procesamiento
        self.pH = float(pH) if pH is not None else None
        self.nombre_archivo = nombre_archivo
        self.buffers = dict(buffers) if buffers else {}
        for nombre, modelo in self.buffers.items():
            if not isinstance(modelo, (ModeloCinetico, ModeloCineticoCompetitivo)):
                raise TypeError(f"'{nombre}' debe ser ModeloCinetico o ModeloCineticoCompetitivo.")
            if modelo.procesamiento is not self.procesamiento:
                raise ValueError(f"'{nombre}' está vinculado a otra instancia de Procesamiento.")
        self.flujos = dict(flujos) if flujos else {}
        for nombre, flujo in self.flujos.items():
            if not isinstance(flujo, Recapturacion):
                raise TypeError(f"'{nombre}' debe ser una instancia de Recapturacion.")
        self.resultados: dict[str, object] = {
            "tiempo": np.array([]),
            "calcio_libre": np.array([]),
            "buffers": {},
            "flujos": {},
        }
        self.historial_calibraciones: dict[str, dict] = {}
    @property
    def _etiqueta_exp(self) -> str:
        return f"{self.nombre_archivo} (pH = {self.pH:.2f})" if self.pH is not None else self.nombre_archivo
    @classmethod
    def desde_experimento(
        cls,
        ruta_csv: str | Path,
        pH: float,
        Vmax: float = 2.5,
        Km: float = 0.44,
        n_hill: float = 2.1,
        Ca_reposo: float = 0.06,
    ) -> "Analizador":
        """Carga un CSV experimental"""
        ruta = Path(ruta_csv)
        if not ruta.exists():
            raise FileNotFoundError(f"no hay el archivo el archivo experimental: {ruta}")#eto me esta generando dudas
        proc = Procesamiento(str(ruta))
        instancia = cls(
            procesamiento=proc,
            buffers=crear_diccionario_buffers(proc, pH=pH),
            flujos={"SERCA1a": Recapturacion(Vmax=Vmax, Km=Km, n_hill=n_hill, Ca_reposo=Ca_reposo)},
            pH=pH,
            nombre_archivo=ruta.name,
        )
        instancia.simular_sistema_completo()
        return instancia
    def simular_sistema_completo(self) -> dict:
        """integra todo, amortiguadores y recapturacion"""
        tiempos = np.copy(self.procesamiento.tiempo)
        ca_libre = self.procesamiento.evaluar_calcio(tiempos)
        self.resultados = {
            "tiempo": tiempos,
            "calcio_libre": ca_libre,
            "buffers": {},
            "flujos": {},
        }
        for nombre, modelo in self.buffers.items():
            modelo.resolver()
            if isinstance(modelo, ModeloCineticoCompetitivo):
                s_tot = modelo.sitios_totales
                ca_u = modelo.complejo_a.copy()
                esp_b = modelo.complejo_b.copy()
                self.resultados["buffers"][nombre] = {
                    "tipo": "competitivo",
                    "calcio_unido": ca_u,
                    "especie_b_unida": esp_b,
                    "nombre_especie_b": "H+" if "Reguladora" in nombre else "Mg2+",
                    "sitios_libres": modelo.sitios_libres.copy(),
                    "fraccion_ocupacion_calcio": ca_u / s_tot,
                    "fraccion_ocupacion_especie_b": esp_b / s_tot,
                    "sitios_totales": s_tot,
                    "kon": modelo.kon_a,
                    "koff": modelo.koff_a,
                    "kon_b": modelo.kon_b,
                    "koff_b": modelo.koff_b,
                }
            else:
                s_tot = modelo.S_total
                ca_u = modelo.calcio_unido.copy()
                self.resultados["buffers"][nombre] = {
                    "tipo": "simple",
                    "calcio_unido": ca_u,
                    "sitios_libres": s_tot - ca_u,
                    "fraccion_ocupacion": ca_u / s_tot,
                    "sitios_totales": s_tot,
                    "kon": modelo.kon,
                    "koff": modelo.koff,
                }

        for nombre, flujo in self.flujos.items():
            j_bomba = flujo.evaluar_flujo_bomba(ca_libre)
            j_neto = flujo.evaluar_flujo_neto(ca_libre)
            self.resultados["flujos"][nombre] = {
                "flujo_bomba": j_bomba,
                "flujo_neto": j_neto,
                "masa_acumulada": cumulative_trapezoid(j_neto, tiempos, initial=0.0),
                "Vmax": flujo.Vmax,
                "Km": flujo.Km,
                "n_hill": flujo.n_hill,
                "J_leak": flujo.J_leak,
            }

        return self.resultados

    def busqueda_estocastica(
        self,
        nombre_buffer: str,
        objetivo_SCa: np.ndarray,
        n_iteraciones: int = 500,
        sigma_log: float = 0.1,
        semilla: int | None = None,
    ) -> dict:
        """Ajusta kon y koff de un ModeloCinetico simple mediante búsqueda aleatoria Log-Normal."""
        if nombre_buffer not in self.buffers:
            raise AnalizadorError(f"El buffer '{nombre_buffer}' no existe.")
        buffer_target = self.buffers[nombre_buffer]
        if not isinstance(buffer_target, ModeloCinetico):
            raise AnalizadorError(f"'{nombre_buffer}' es competitivo; solo se calibran buffers simples.")
        if n_iteraciones <= 0 or sigma_log <= 0 or not np.isfinite(sigma_log):
            raise ValueError("n_iteraciones y sigma_log deben ser positivos y finitos.")

        objetivo = np.asarray(objetivo_SCa, dtype=float)
        tol = 1e-7
        if objetivo.shape != self.procesamiento.tiempo.shape or not np.all(np.isfinite(objetivo)):
            raise ValueError("objetivo_SCa debe ser finito y coincidir con el vector de tiempos.")
        if np.any(objetivo < -tol) or np.any(objetivo > buffer_target.S_total + tol):
            raise ValueError("objetivo_SCa viola los límites físicos [0, S_total].")
        objetivo = np.clip(objetivo, 0.0, buffer_target.S_total)

        kon_orig, koff_orig = float(buffer_target.kon), float(buffer_target.koff)
        rng = np.random.default_rng(semilla)

        def evaluar(kon: float, koff: float) -> tuple[float, np.ndarray]:
            buffer_target.kon, buffer_target.koff = kon, koff
            sol = buffer_target.resolver()
            if not sol.success:
                raise RuntimeError("Fallo de integración.")
            tray = np.asarray(sol.y[0], dtype=float)
            if not np.all(np.isfinite(tray)) or np.any(tray < -tol) or np.any(tray > buffer_target.S_total + tol):
                raise RuntimeError("Trayectoria fuera de límites físicos.")
            return float(np.sqrt(np.mean((tray - objetivo) ** 2))), tray

        try:
            mejor_rmse, mejor_tray = evaluar(kon_orig, koff_orig)
        except Exception as err:
            buffer_target.kon, buffer_target.koff = kon_orig, koff_orig
            raise AnalizadorError(f"Error al evaluar parámetros iniciales: {err}") from err

        rmse_ini = mejor_rmse
        mejor_kon, mejor_koff = kon_orig, koff_orig
        historial_rmse = [mejor_rmse]
        validos = aceptados = fallidos = 0

        for _ in range(int(n_iteraciones)):
            log_kon = np.log(mejor_kon) + rng.normal(0.0, sigma_log)
            log_koff = np.log(mejor_koff) + rng.normal(0.0, sigma_log)
            if not (-700.0 < log_kon < 700.0 and -700.0 < log_koff < 700.0):
                fallidos += 1
                historial_rmse.append(mejor_rmse)
                continue

            try:
                rmse_cand, tray_cand = evaluar(float(np.exp(log_kon)), float(np.exp(log_koff)))
                validos += 1
                if rmse_cand < mejor_rmse:
                    mejor_rmse = rmse_cand
                    mejor_kon, mejor_koff = float(np.exp(log_kon)), float(np.exp(log_koff))
                    mejor_tray = tray_cand.copy()
                    aceptados += 1
            except Exception:
                fallidos += 1
            historial_rmse.append(mejor_rmse)

        buffer_target.kon, buffer_target.koff = mejor_kon, mejor_koff
        buffer_target.resolver()

        salida = {
            "buffer": nombre_buffer,
            "kon_inicial": kon_orig,
            "koff_inicial": koff_orig,
            "kon_optimizado": mejor_kon,
            "koff_optimizado": mejor_koff,
            "Kd_inicial": koff_orig / kon_orig,
            "Kd_optimizada": mejor_koff / mejor_kon,
            "rmse_inicial": rmse_ini,
            "rmse_final": mejor_rmse,
            "mejora_rmse_porcentaje": 100.0 * (rmse_ini - mejor_rmse) / rmse_ini if rmse_ini > 0 else 0.0,
            "candidatos_validos": validos,
            "candidatos_aceptados": aceptados,
            "candidatos_fallidos": fallidos,
            "historial_rmse": np.asarray(historial_rmse, dtype=float),
            "trayectoria_optimizada": mejor_tray,
            "objetivo_SCa": objetivo,
        }
        self.historial_calibraciones[nombre_buffer] = salida
        return salida

    @staticmethod
    def _sincronizar_eje_porcentaje(ax, s_total: float, etiqueta: str = "Ocupación (%)") -> None: #esto es para las graficas
        """Agrega un eje derecho sincronizado en porcentaje respecto a s_total."""
        ax.relim()
        ax.autoscale_view()
        y_min, y_max = ax.get_ylim()
        ax_pct = ax.twinx()
        ax_pct.set_ylim((y_min / s_total) * 100.0, (y_max / s_total) * 100.0)
        ax_pct.set_ylabel(etiqueta)

    def graficar_interpolador(self) -> None:
        proc = self.procesamiento
        t_cont = np.linspace(proc.tiempo[0], proc.tiempo[-1], 1000)

        plt.figure(figsize=(9, 5))
        plt.scatter(proc.tiempo, proc.calcio_limpio, color="blue", s=25, zorder=3, label="Datos experimentales")
        plt.plot(t_cont, proc.evaluar_calcio(t_cont), color="red", lw=2, label="Interpolación PCHIP")
        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Concentración de calcio (µM)")
        plt.title(f"Interpolación PCHIP — {self._etiqueta_exp}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def graficar_derivada(self, t_destacado: float | None = None) -> None:
        proc = self.procesamiento
        t_cont = np.linspace(proc.tiempo[0], proc.tiempo[-1], 1000)

        plt.figure(figsize=(9, 5))
        plt.plot(t_cont, proc.derivada_interpolador(t_cont), color="darkorange", lw=2, label="Derivada PCHIP")
        plt.axhline(0, color="black", ls="--", lw=1)

        if t_destacado is not None:
            if not (np.isfinite(t_destacado) and proc.tiempo[0] <= t_destacado <= proc.tiempo[-1]):
                raise ValueError("El tiempo destacado está fuera del intervalo experimental.")
            plt.scatter(
                t_destacado,
                proc.derivada_interpolador(t_destacado),
                color="red",
                s=70,
                zorder=3,
                label=f"Derivada a {t_destacado} ms",
            )

        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Variación del calcio (µM/ms)")
        plt.title(f"Derivada temporal del calcio citosólico — {self._etiqueta_exp}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def graficar_integracion(self, t0: float, tf: float) -> float:
        proc = self.procesamiento
        auc = proc.area_bajo_curva(t0, tf)
        t_int = np.linspace(t0, tf, 500)
        t_all = np.linspace(proc.tiempo[0], proc.tiempo[-1], 1000)

        plt.figure(figsize=(9, 5))
        plt.plot(proc.tiempo, proc.calcio_limpio, color="blue", alpha=0.5, label="Datos experimentales")
        plt.plot(t_all, proc.evaluar_calcio(t_all), color="red", lw=2, label="Interpolación PCHIP")
        plt.fill_between(t_int, proc.evaluar_calcio(t_int), 0, color="orange", alpha=0.4, label="Área integrada")
        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Concentración de calcio (µM)")
        plt.title(f"Exposición acumulada de calcio citosólico — {self._etiqueta_exp}")
        plt.text(
            0.03, 0.95, f"AUC = {auc:.6f} µM·ms",
            transform=plt.gca().transAxes, va="top", bbox=dict(facecolor="white", alpha=0.8),
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return auc

    def graficar_calcio_unido_buffers(self, escala_log: bool = False) -> None:
        """Grafica [SCa] (µM) en dos paneles con escalas verticales separadas (Alta vs. Baja unión)."""
        t = self.resultados["tiempo"]
        buffers = self.resultados["buffers"]

        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
        fig.suptitle(f"Calcio Unido por Amortiguador — {self._etiqueta_exp}", fontsize=13, fontweight="bold")

        for nombre, datos in buffers.items():
            ax = axes[0] if np.max(datos["calcio_unido"]) > 25.0 else axes[1]
            ax.plot(t, datos["calcio_unido"], lw=2.2, label=nombre)

        axes[0].set_title("amortiguadores principales (TnC y Parvalbúmina)")
        axes[1].set_title("amortiguadores rapidos de baja union(ATP y Mag-Fluo-4)")
        axes[1].set_xlabel("tiempo (ms)")
        for ax, linthresh in zip(axes, (1.0, 0.1)):
            ax.set_ylabel(r"$[\mathrm{SCa}]$ ($\mu\mathrm{M}$)")
            ax.set_yscale("symlog", linthresh=linthresh) if escala_log else ax.set_ylim(bottom=0)
            ax.grid(True, ls=":", alpha=0.6)
            ax.legend(loc="upper right")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    def graficar_ocupacion_buffers(self) -> None:
        """
        Desglosa la ocupación porcentual (%) en 4 paneles con escalas independientes:
        1. Proteínas principales (Ca2+).
        2. Baja afinidad en doble eje: Mag-Fluo-4 (izq) y ATP Citosólico (der).
        3. Protones (H+) en Troponina C Reguladora (en su propia escala).
        4. Magnesio (Mg2+) en ATP, TnC TNS y Parvalbúmina.
        """
        t = self.resultados["tiempo"]
        buffers = self.resultados["buffers"]
        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        fig.suptitle(f"Ocupación Porcentual de Sitios por Especie — {self._etiqueta_exp}", fontsize=13, fontweight="bold")
        # Panel 1: Proteínas Ca2+
        for nombre, datos in buffers.items():
            frac_ca = (
                datos["fraccion_ocupacion_calcio"]
                if datos["tipo"] == "competitivo"
                else datos["fraccion_ocupacion"]
            ) * 100.0
            if np.max(frac_ca) > 10.0:
                axes[0].plot(t, frac_ca, lw=2.2, label=f"{nombre} (Ca²⁺)")
        axes[0].set_ylabel("Ocupación Ca²⁺ (%)")
        axes[0].set_title("1. Ocupación por Ca²⁺ — Proteínas Miofibrilares / Citosólicas")
        axes[0].set_ylim(0, 85)
        axes[0].grid(True, ls=":", alpha=0.6)
        axes[0].legend(loc="upper right")
        ax_atp = axes[1].twinx()
        lineas_p2 = []
        for nombre, datos in buffers.items():
            frac_ca = (
                datos["fraccion_ocupacion_calcio"]
                if datos["tipo"] == "competitivo"
                else datos["fraccion_ocupacion"]
            ) * 100.0
            if np.max(frac_ca) <= 10.0:
                if "ATP" in nombre:
                    lineas_p2 += ax_atp.plot(t, frac_ca, color="tab:blue", ls="--", lw=2.2, label=f"{nombre} (Ca²⁺) [eje der.]")
                else:
                    lineas_p2 += axes[1].plot(t, frac_ca, color="tab:orange", lw=2.2, label=f"{nombre} (Ca²⁺) [eje izq.]")

        axes[1].set_ylabel("Mag-Fluo-4 (%)", color="tab:orange")
        ax_atp.set_ylabel("ATP Citosólico (%)", color="tab:blue")
        axes[1].set_ylim(bottom=0)
        ax_atp.set_ylim(bottom=0)
        axes[1].set_title("2. Ocupación por Ca²⁺ — Escala Ampliada Independiente (Mag-Fluo-4 vs. ATP)")
        axes[1].grid(True, ls=":", alpha=0.6)
        if lineas_p2:
            axes[1].legend(lineas_p2, [l.get_label() for l in lineas_p2], loc="upper right")
        for nombre, datos in buffers.items():
            if datos["tipo"] == "competitivo" and datos["nombre_especie_b"] == "H+":
                axes[2].plot(
                    t,
                    datos["fraccion_ocupacion_especie_b"] * 100.0,
                    color="darkorange",
                    ls="--",
                    lw=2.2,
                    label=f"{nombre} (H⁺)",
                )
        axes[2].set_ylabel("Ocupación H⁺ (%)")
        axes[2].set_title("3. Competencia Protónica (H⁺) en Troponina C Reguladora")
        axes[2].set_ylim(bottom=0)
        axes[2].grid(True, ls=":", alpha=0.6)
        axes[2].legend(loc="lower right")
        for nombre, datos in buffers.items():
            if datos["tipo"] == "competitivo" and datos["nombre_especie_b"] == "Mg2+":
                axes[3].plot(
                    t,
                    datos["fraccion_ocupacion_especie_b"] * 100.0,
                    ls="--",
                    lw=2.0,
                    label=f"{nombre} (Mg²⁺)",
                )
        axes[3].set_xlabel("Tiempo (ms)")
        axes[3].set_ylabel("Ocupación Mg²⁺ (%)")
        axes[3].set_title("4. Ocupación Basal y Desplazamiento de Magnesio (Mg²⁺)")
        axes[3].set_ylim(0, 100)
        axes[3].grid(True, ls=":", alpha=0.6)
        axes[3].legend(loc="center right", ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def graficar_buffer_individual(self, nombre_buffer: str) -> None:
        """
        Grafica un único amortiguador en paneles independientes por especie,
        incluyendo siempre el complejo con Ca2+, el competidor (H+/Mg2+) si aplica,
        y la concentración del buffer libre ([S] libre), con eje derecho en porcentaje (%).
        """
        if nombre_buffer not in self.resultados["buffers"]:
            raise AnalizadorError(f"El amortiguador '{nombre_buffer}' no existe.")
        t = self.resultados["tiempo"]
        b_data = self.resultados["buffers"][nombre_buffer]
        s_tot = b_data["sitios_totales"]
        es_comp = b_data["tipo"] == "competitivo"

        n_paneles = 3 if es_comp else 2
        fig, axes = plt.subplots(n_paneles, 1, figsize=(10.5, 3.6 * n_paneles), sharex=True)
        fig.suptitle(
            f"Análisis Detallado: {nombre_buffer} (Stot = {s_tot:.1f} µM) — {self._etiqueta_exp}",
            fontsize=13,
            fontweight="bold",
        )

        # Panel 1: Complejo con Calcio ([SCa]) en su propia escala
        axes[0].plot(t, b_data["calcio_unido"], color="tab:blue", lw=2.3, label=r"$[\mathrm{Ca}^{2+}]$ unido al buffer")
        axes[0].set_ylabel(r"$[\mathrm{SCa}]$ ($\mu\mathrm{M}$)")
        axes[0].set_title("1. Calcio Unido al Amortiguador")
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, ls=":", alpha=0.6)
        axes[0].legend(loc="upper right")
        self._sincronizar_eje_porcentaje(axes[0], s_tot, "Ocupación Ca²⁺ (%)")

        if es_comp:
            esp_b = b_data["nombre_especie_b"]
            # Panel 2: Especie competidora (H+ o Mg2+) en su propia escala
            axes[1].plot(
                t,
                b_data["especie_b_unida"],
                color="tab:orange",
                ls="--",
                lw=2.3,
                label=f"[{esp_b}] unido al buffer",
            )
            axes[1].set_ylabel(f"[{esp_b}] unido (µM)")
            axes[1].set_title(f"2. Especie Competidora Unida ({esp_b})")
            axes[1].set_ylim(bottom=0)
            axes[1].grid(True, ls=":", alpha=0.6)
            axes[1].legend(loc="upper right")
            self._sincronizar_eje_porcentaje(axes[1], s_tot, f"Ocupación {esp_b} (%)")

        # Último panel: Buffer libre ([S] libre)
        ax_libre = axes[-1]
        ax_libre.plot(t, b_data["sitios_libres"], color="seagreen", lw=2.3, label=f"Buffer libre [{nombre_buffer}]")
        ax_libre.set_xlabel("Tiempo (ms)")
        ax_libre.set_ylabel(r"$[\mathrm{S}]_{\mathrm{libre}}$ ($\mu\mathrm{M}$)")
        ax_libre.set_title(f"{n_paneles}. Concentración de Amortiguador Libre (Sitios Disponibles)")
        ax_libre.grid(True, ls=":", alpha=0.6)
        ax_libre.legend(loc="upper right")
        self._sincronizar_eje_porcentaje(ax_libre, s_tot, "Buffer libre (%)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def graficar_serca(self) -> None:
        """Grafica el flujo neto instantáneo y la masa acumulada de SERCA en dos paneles limpios."""
        t = self.resultados["tiempo"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
        fig.suptitle(f"Transporte Activo de Recaptación — {self._etiqueta_exp}", fontsize=13, fontweight="bold")

        for nombre, f_data in self.resultados["flujos"].items():
            axes[0].plot(t, f_data["flujo_bomba"], ls="--", color="gray", lw=1.6, label=f"J_bomba {nombre}")
            axes[0].plot(t, f_data["flujo_neto"], color="crimson", lw=2.2, label=f"J_neto {nombre}")
            axes[1].plot(t, f_data["masa_acumulada"], color="teal", lw=2.3, label=f"Masa acumulada {nombre}")

        axes[0].axhline(0, color="black", ls=":", lw=0.9)
        axes[0].set_ylabel(r"Flujo ($\mu\mathrm{M/ms}$)")
        axes[0].set_title("1. Tasas Instantáneas de Bombeo")
        axes[1].set_xlabel("Tiempo (ms)")
        axes[1].set_ylabel(r"Masa Acumulada ($\mu\mathrm{M}$)")
        axes[1].set_title("2. Calcio Total Secuestrado por el Retículo Sarcoplásmico")
        axes[1].set_ylim(bottom=0)

        for ax in axes:
            ax.grid(True, ls=":", alpha=0.6)
            ax.legend(loc="best")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def graficar_calibracion(self, nombre_buffer: str) -> None:
        """Grafica el ajuste cinético y la convergencia del RMSE de la calibración Log-Normal."""
        if nombre_buffer not in self.historial_calibraciones:
            print("⚠️ Aún no se ha ejecutado una calibración para ese buffer.")
            return

        cal = self.historial_calibraciones[nombre_buffer]
        t = self.resultados["tiempo"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Calibración Estocástica Log-Normal — {nombre_buffer} ({self._etiqueta_exp})", fontweight="bold")

        axes[0].plot(t, cal["objetivo_SCa"], "ko", markersize=4, alpha=0.7, label="Curva objetivo [SCa]")
        axes[0].plot(t, cal["trayectoria_optimizada"], "r-", lw=2, label="Ajuste optimizado (Radau)")
        axes[0].set_xlabel("Tiempo (ms)")
        axes[0].set_ylabel(r"$[\mathrm{SCa}]$ ($\mu\mathrm{M}$)")
        axes[0].set_title(f"Ajuste Cinético (RMSE = {cal['rmse_final']:.3e} µM)")

        axes[1].plot(cal["historial_rmse"], color="navy", lw=2)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Iteración Monte Carlo")
        axes[1].set_ylabel("RMSE (µM, escala log)")
        axes[1].set_title(f"Convergencia (Mejora: {cal['mejora_rmse_porcentaje']:.2f}%)")

        for ax in axes:
            ax.grid(True, ls=":", alpha=0.6)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def comparar_experimentos(ana_basal: "Analizador", ana_acido: "Analizador") -> None:
        """Compara cuantitativa y gráficamente el efecto de la acidosis (H+) entre dos experimentos."""
        ca_max_1 = float(np.max(ana_basal.resultados["calcio_libre"]))
        ca_max_2 = float(np.max(ana_acido.resultados["calcio_libre"]))
        tnc_1 = ana_basal.resultados["buffers"]["Troponina C (Reguladora)"]
        tnc_2 = ana_acido.resultados["buffers"]["Troponina C (Reguladora)"]
        masa_1 = float(ana_basal.resultados["flujos"]["SERCA1a"]["masa_acumulada"][-1])
        masa_2 = float(ana_acido.resultados["flujos"]["SERCA1a"]["masa_acumulada"][-1])

        print(" COMPARATIVA CUANTITATIVA: normal (pH 7.33) vs acidosis (pH 6.50)")
        print(f"Pico [Ca2+] libre         : {ca_max_1:.4f} µM  -->  {ca_max_2:.4f} µM")
        print(
            f"Ca2+-TnC Reguladora máx   : {np.max(tnc_1['calcio_unido']):.4f} µM "
            f"({np.max(tnc_1['fraccion_ocupacion_calcio']) * 100:.1f}%)  -->  "
            f"{np.max(tnc_2['calcio_unido']):.4f} µM "
            f"({np.max(tnc_2['fraccion_ocupacion_calcio']) * 100:.1f}%)"
        )
        print(
            f"H+-TnC Reguladora inicial : {tnc_1['especie_b_unida'][0]:.4f} µM  -->  "
            f"{tnc_2['especie_b_unida'][0]:.4f} µM"
        )
        print(f"Masa recapturada SERCA1a  : {masa_1:.4f} µM  -->  {masa_2:.4f} µM")

        fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
        fig.suptitle("Comparativa Biofísica: normal (pH 7.33) vs. acidosis (pH 6.50)", fontsize=14, fontweight="bold")

        axes[0].plot(ana_basal.resultados["tiempo"], ana_basal.resultados["calcio_libre"], "b-", lw=2, label="pH 7.33 (Basal)")
        axes[0].plot(ana_acido.resultados["tiempo"], ana_acido.resultados["calcio_libre"], "r--", lw=2, label="pH 6.50 (Acidosis)")
        axes[0].set_ylabel(r"$[\mathrm{Ca}^{2+}]$ libre ($\mu\mathrm{M}$)")
        axes[0].set_title("1. Señal Experimental de Calcio Libre")

        axes[1].plot(ana_basal.resultados["tiempo"], tnc_1["calcio_unido"], "b-", lw=2, label="Ca²⁺-TnC (pH 7.33)")
        axes[1].plot(ana_acido.resultados["tiempo"], tnc_2["calcio_unido"], "r--", lw=2, label="Ca²⁺-TnC (pH 6.50)")
        axes[1].set_ylabel(r"$[\mathrm{TnC\text{-}Ca}]$ ($\mu\mathrm{M}$)")
        axes[1].set_title("2. Desplazamiento de Ca²⁺ por Competencia Protónica en Troponina C (Reguladora)")

        axes[2].plot(ana_basal.resultados["tiempo"], tnc_1["especie_b_unida"], "b-", lw=2, label="H⁺-TnC (pH 7.33)")
        axes[2].plot(ana_acido.resultados["tiempo"], tnc_2["especie_b_unida"], "r--", lw=2, label="H⁺-TnC (pH 6.50)")
        axes[2].set_xlabel("Tiempo (ms)")
        axes[2].set_ylabel(r"$[\mathrm{TnC\text{-}H}]$ ($\mu\mathrm{M}$)")
        axes[2].set_title("3. Ocupación de Sitios Reguladores por Protones (H⁺)")

        for ax in axes:
            ax.legend()
            ax.grid(True, ls=":", alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    @classmethod
    def ejecutar_sesion_interactiva(cls, experimentos: dict[str, "Analizador"]) -> None:
        """Selector principal de experimentos y comparativas."""
        claves = list(experimentos.keys())
        while True:
            print("   mi pequeño programa de analisis, version inicial ")
            for idx, etiqueta in enumerate(claves, 1):
                print(f"  {idx}. {etiqueta}")
            idx_comp = len(claves) + 1
            idx_salir = len(claves) + 2
            if len(claves) >= 2:
                print(f"  {idx_comp}. Comparar {claves[0]} vs {claves[1]} (Efecto H+)")
            print(f"  {idx_salir}. Salir")

            opcion = input(f"\nseleccione una opción (1-{idx_salir}): ").strip()

            if opcion.isdigit() and 1 <= int(opcion) <= len(claves):
                if not experimentos[claves[int(opcion) - 1]].menu_interactivo(permitir_cambio=True):
                    break
            elif len(claves) >= 2 and opcion == str(idx_comp):
                cls.comparar_experimentos(experimentos[claves[0]], experimentos[claves[1]])
            elif opcion == str(idx_salir):
                print("\nSesión finalizada.")
                break
            else:
                print("Opcion no valida. presinone lo que es ")

    def menu_interactivo(self, permitir_cambio: bool = False) -> bool:
        """Menú completo de consultas numéricas, calibración y gráficas seleccionables."""
        while True:
            print(f" menu:  — {self._etiqueta_exp}")
            print(" consulta de datos: ")
            print("  1. datos de entrada y un poco de su analisis ([Ca2+] libre, [H+] ambiental, dCa/dt, AUC)")
            print("  2. tabla resumen delos amortiguadores (Ca2+, H+, Mg2+ y libres)")
            print("  3. Inspeccionar un amortiguador en detalle (kon, koff, Kd)")
            print("  4. Transporte activo SERCA1a (Vmax, Km, n_Hill, J_leak y masa)")
            print("  5. Calibración estocástica Log-Normal (ejecutar y ver constantes)")
            print("\n [GRÁFICAS SELECCIONABLES POR ESCALA]")
            print("  6. Gráfica: Datos experimentales e interpolación PCHIP")
            print("  7. Gráfica: Derivada temporal (d[Ca2+]/dt)")
            print("  8. Gráfica: Área bajo la curva (AUC integrada)")
            print("  9. Gráfica: Calcio unido (µM) — Escalas separadas (Alta vs. Baja unión)")
            print(" 10. Gráfica: Ocupación porcentual (%) — 4 paneles (Ca2+, ATP/Dye, H+, Mg2+)")
            print(" 11. Gráfica: Elegir UN solo amortiguador (Ca2+, H+/Mg2+ y Buffer libre)")
            print(" 12. Gráfica: Transporte activo SERCA1a (Flujo neto y Masa acumulada)")
            print(" 13. Ver secuencia completa de gráficas del sistema")
            print("\n [NAVEGACIÓN]")
            if permitir_cambio:
                print(" 14. Volver al selector de archivo / experimento")
            print(" 15. Salir del programa")

            opcion = input("\nSeleccione una opción: ").strip()

            if opcion == "1":
                self._inspeccionar_calcio_libre()
            elif opcion == "2":
                self._resumen_todos_los_buffers()
            elif opcion == "3":
                self._inspeccionar_buffer_interactivo()
            elif opcion == "4":
                self._inspeccionar_flujo_interactivo()
            elif opcion == "5":
                self._menu_calibracion_estocastica()
            elif opcion == "6":
                self.graficar_interpolador()
            elif opcion == "7":
                t_in = input("Instante t a destacar en ms [Enter para omitir]: ").strip()
                try:
                    self.graficar_derivada(t_destacado=float(t_in) if t_in else None)
                except ValueError as err:
                    print(f"❌ Error: {err}")
            elif opcion == "8":
                try:
                    t0_in = input("Tiempo inicial t0 (ms) [Enter para inicio]: ").strip()
                    tf_in = input("Tiempo final tf (ms) [Enter para final]: ").strip()
                    t0 = float(t0_in) if t0_in else float(self.procesamiento.tiempo[0])
                    tf = float(tf_in) if tf_in else float(self.procesamiento.tiempo[-1])
                    self.graficar_integracion(t0, tf)
                except ValueError as err:
                    print(f"❌ Error en tiempos de integración: {err}")
            elif opcion == "9":
                modo = input("¿Usar escala logarítmica simétrica (symlog)? (s/n) [Enter = n]: ").strip().lower()
                self.graficar_calcio_unido_buffers(escala_log=(modo == "s"))
            elif opcion == "10":
                self.graficar_ocupacion_buffers()
            elif opcion == "11":
                self._seleccionar_y_graficar_buffer()
            elif opcion == "12":
                self.graficar_serca()
            elif opcion == "13":
                self.graficar_calcio_unido_buffers()
                self.graficar_ocupacion_buffers()
                self.graficar_serca()
            elif opcion == "14" and permitir_cambio:
                return True
            elif opcion == "15":
                print("\nSaliendo del Analizador Interactivo. ¡Simulación finalizada!")
                return False
            else:
                print("⚠️ Opción no válida. Intente de nuevo.")

    def _seleccionar_buffer_por_usuario(self) -> str:
        nombres = list(self.resultados["buffers"].keys())
        print("\nAmortiguadores disponibles:")
        for idx, nombre in enumerate(nombres, 1):
            print(f"  {idx}. {nombre}")
        sel = input("\nIngrese el número o nombre del buffer: ").strip()
        nombre = nombres[int(sel) - 1] if sel.isdigit() else sel
        if nombre not in self.resultados["buffers"]:
            raise KeyError(f"El buffer '{nombre}' no existe.")
        return nombre

    def _seleccionar_y_graficar_buffer(self) -> None:
        try:
            self.graficar_buffer_individual(self._seleccionar_buffer_por_usuario())
        except (IndexError, KeyError, ValueError, AnalizadorError) as err:
            print(f"❌ Selección inválida: {err}")

    def _inspeccionar_calcio_libre(self) -> None:
        t = self.resultados["tiempo"]
        ca = self.resultados["calcio_libre"]
        t_pico = float(t[np.argmax(ca)])
        d_ca_max = float(np.max(self.procesamiento.derivada_interpolador(t)))
        auc = self.procesamiento.area_bajo_curva(float(t[0]), float(t[-1]))

        print(f"\n--- SEÑAL EXPERIMENTAL: {self._etiqueta_exp} ---")
        if self.pH is not None:
            print(f"pH experimental    : {self.pH:.2f} ([H+] libre = {(10.0 ** (-self.pH)) * 1e6:.4f} µM)")
        print(f"Puntos temporales  : {len(t)} (de {t[0]:.2f} ms a {t[-1]:.2f} ms)")
        print(f"Calcio basal       : {ca[0]:.6f} µM")
        print(f"Calcio máximo      : {np.max(ca):.6f} µM (en t = {t_pico:.2f} ms)")
        print(f"Tasa máx d[Ca2+]/dt: {d_ca_max:.6f} µM/ms")
        print(f"Área bajo la curva : {auc:.6f} µM·ms")

    def _resumen_todos_los_buffers(self) -> None:
        print(f"\n--- RESUMEN DE AMORTIGUADORES ({self._etiqueta_exp}) ---")
        for nombre, b_data in self.resultados["buffers"].items():
            sca = b_data["calcio_unido"]
            lib = b_data["sitios_libres"]
            if b_data["tipo"] == "competitivo":
                esp_b = b_data["nombre_especie_b"]
                sb = b_data["especie_b_unida"]
                oc_ca = np.max(b_data["fraccion_ocupacion_calcio"]) * 100.0
                print(
                    f"  * {nombre:<28} [Ca2+/{esp_b:<4}] | Ca2+ máx: {np.max(sca):8.4f} µM ({oc_ca:5.1f}%) | "
                    f"{esp_b}: {sb[0]:7.2f}->{np.min(sb):7.2f} µM | Libre: {lib[0]:6.2f}->{np.min(lib):6.2f} µM"
                )
            else:
                oc_ca = np.max(b_data["fraccion_ocupacion"]) * 100.0
                print(
                    f"  * {nombre:<28} [Simple 1 sitio] | Ca2+ máx: {np.max(sca):8.4f} µM ({oc_ca:5.1f}%) | "
                    f"Libre: {lib[0]:6.2f}->{np.min(lib):6.2f} µM"
                )

    def _inspeccionar_buffer_interactivo(self) -> None:
        try:
            nombre = self._seleccionar_buffer_por_usuario()
            b_data = self.resultados["buffers"][nombre]
            sca = b_data["calcio_unido"]
            libres = b_data["sitios_libres"]

            print(f"\n--- DETALLES DEL BUFFER: {nombre} ({b_data['tipo'].upper()}) ---")
            print(f"Sitios totales Stot     : {b_data['sitios_totales']:.4f} µM")
            print(
                f"Cinética Ca2+           : kon = {b_data['kon']:.5e} µM^-1 ms^-1 | "
                f"koff = {b_data['koff']:.5e} ms^-1 | Kd = {b_data['koff'] / b_data['kon']:.5e} µM"
            )
            print(f"Ca2+ unido (ini/máx/fin): {sca[0]:.6f} / {np.max(sca):.6f} / {sca[-1]:.6f} µM")
            print(f"Buffer libre (ini/mín)  : {libres[0]:.6f} / {np.min(libres):.6f} µM")

            if b_data["tipo"] == "competitivo":
                esp_b = b_data["nombre_especie_b"]
                sb = b_data["especie_b_unida"]
                print(
                    f"Cinética {esp_b:<15}: kon = {b_data['kon_b']:.5e} µM^-1 ms^-1 | "
                    f"koff = {b_data['koff_b']:.5e} ms^-1 | Kd = {b_data['koff_b'] / b_data['kon_b']:.5e} µM"
                )
                print(f"{esp_b} unido (ini/mín/fin) : {sb[0]:.6f} / {np.min(sb):.6f} / {sb[-1]:.6f} µM")
                print(
                    f"Ocupación máx Ca2+/{esp_b}: "
                    f"{np.max(b_data['fraccion_ocupacion_calcio']) * 100:.2f} % / "
                    f"{np.max(b_data['fraccion_ocupacion_especie_b']) * 100:.2f} %"
                )
            else:
                print(f"Ocupación máxima Ca2+   : {np.max(b_data['fraccion_ocupacion']) * 100:.2f} %")
        except (IndexError, KeyError, ValueError) as err:
            print(f"❌ Entrada inválida ({err}).")

    def _inspeccionar_flujo_interactivo(self) -> None:
        if not self.resultados["flujos"]:
            print("⚠️ No hay flujos registrados.")
            return

        nombres = list(self.resultados["flujos"].keys())
        print("\nFlujos de recaptación disponibles:")
        for idx, nombre in enumerate(nombres, 1):
            print(f"  {idx}. {nombre}")

        try:
            sel = input("\nIngrese el número o nombre del flujo: ").strip()
            nombre = nombres[int(sel) - 1] if sel.isdigit() else sel
            f_data = self.resultados["flujos"][nombre]

            print(f"\n--- DETALLES DEL FLUJO: {nombre} ---")
            print(f"Parámetros        : Vmax = {f_data['Vmax']} µM/ms | Km = {f_data['Km']} µM | n_Hill = {f_data['n_hill']}")
            print(f"Fuga Pasiva J_leak: {f_data['J_leak']:.10f} µM/ms")
            print(f"Flujo bomba máx   : {np.max(f_data['flujo_bomba']):.6f} µM/ms")
            print(f"Flujo neto máx    : {np.max(f_data['flujo_neto']):.6f} µM/ms")
            print(f"Masa acumulada fin: {f_data['masa_acumulada'][-1]:.6f} µM")
        except (IndexError, KeyError, ValueError) as err:
            print(f"❌ Entrada inválida ({err}).")

    def _menu_calibracion_estocastica(self) -> None:
        simples = [k for k, v in self.buffers.items() if isinstance(v, ModeloCinetico)]
        if not simples:
            print("⚠️ No hay buffers simples disponibles para calibrar.")
            return

        nombre_buf = simples[0]
        modelo = self.buffers[nombre_buf]

        print(f"\n--- CALIBRACIÓN ESTOCÁSTICA LOG-NORMAL ({nombre_buf}) ---")
        print(f"Constantes actuales: kon = {modelo.kon:.5e} | koff = {modelo.koff:.5e} | Kd = {modelo.koff / modelo.kon:.5e} µM")

        if nombre_buf in self.historial_calibraciones:
            prev = self.historial_calibraciones[nombre_buf]
            print("\nÚltima calibración registrada:")
            print(f"  * kon  inicial -> optimizado : {prev['kon_inicial']:.5e} -> {prev['kon_optimizado']:.5e}")
            print(f"  * koff inicial -> optimizado : {prev['koff_inicial']:.5e} -> {prev['koff_optimizado']:.5e}")
            print(f"  * Kd   inicial -> optimizada : {prev['Kd_inicial']:.5e} -> {prev['Kd_optimizada']:.5e} µM")
            print(f"  * RMSE inicial -> final      : {prev['rmse_inicial']:.6e} -> {prev['rmse_final']:.6e} µM ({prev['mejora_rmse_porcentaje']:.2f}%)")

        if input("\n¿Ejecutar nueva búsqueda estocástica ahora? (s/n): ").strip().lower() != "s":
            return

        n_in = input("Número de iteraciones Monte Carlo [Enter = 60]: ").strip()
        n_iter = int(n_in) if n_in.isdigit() and int(n_in) > 0 else 60

        modelo.resolver()
        objetivo_sca = modelo.calcio_unido.copy()

        kon_ref, koff_ref = float(modelo.kon), float(modelo.koff)
        modelo.kon, modelo.koff = kon_ref * 1.40, koff_ref * 0.65

        print(f"\nIniciando desde kon_ini={modelo.kon:.4e}, koff_ini={modelo.koff:.4e} ({n_iter} iteraciones)...", flush=True)
        t0 = time.perf_counter()
        llamadas = {"n": 0}
        resolver_orig = modelo.resolver

        def resolver_progreso(*args, **kwargs):
            sol = resolver_orig(*args, **kwargs)
            llamadas["n"] += 1
            if llamadas["n"] % max(1, n_iter // 4) == 0:
                print(f"  -> {llamadas['n']}/{n_iter + 2} evaluaciones ({time.perf_counter() - t0:.1f} s)...", flush=True)
            return sol

        modelo.resolver = resolver_progreso
        try:
            res = self.busqueda_estocastica(nombre_buf, objetivo_sca, n_iteraciones=n_iter, sigma_log=0.08, semilla=42)
        finally:
            modelo.resolver = resolver_orig

        self.simular_sistema_completo()

        print(f"\n✔ Calibración completada en {time.perf_counter() - t0:.2f} s:")
        print(f"  * kon  inicial -> optimizado : {res['kon_inicial']:.5e} -> {res['kon_optimizado']:.5e} (ref: {kon_ref:.5e})")
        print(f"  * koff inicial -> optimizado : {res['koff_inicial']:.5e} -> {res['koff_optimizado']:.5e} (ref: {koff_ref:.5e})")
        print(f"  * Kd   inicial -> optimizada : {res['Kd_inicial']:.5e} -> {res['Kd_optimizada']:.5e} µM")
        print(f"  * RMSE inicial -> final      : {res['rmse_inicial']:.6e} -> {res['rmse_final']:.6e} µM (Mejora: {res['mejora_rmse_porcentaje']:.2f}%)")
        print(f"  * Pasos aceptados / fallidos : {res['candidatos_aceptados']} / {res['candidatos_fallidos']}")

        if input("\n¿Deseas graficar el ajuste y la convergencia del RMSE? (s/n): ").strip().lower() == "s":
            self.graficar_calibracion(nombre_buf)