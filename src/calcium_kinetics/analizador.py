import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from calcium_kinetics.procesamiento import Procesamiento
from calcium_kinetics.modelo_cinetico import ModeloCinetico, ModeloCineticoCompetitivo
from calcium_kinetics.recapturacion import Recapturacion


class AnalizadorError(Exception):
    """Excepción personalizada para errores dentro del módulo Analizador."""
    pass


class Analizador:
    """
    Orquesta la simulación colectiva del transitorio de calcio,
    coordinando los buffers pasivos (ModeloCinetico) y los flujos
    de transporte activo (Recapturacion), con soporte de diagnóstico 
    visual de señal y menú interactivo.
    """

    def __init__(
        self,
        procesamiento: Procesamiento,
        buffers: dict[
            str,
            ModeloCinetico | ModeloCineticoCompetitivo
        ] | None = None,
        flujos: dict[str, Recapturacion] | None = None
    ):

        # --------------------------------------------------
        # 1. Validar el objeto de procesamiento
        # --------------------------------------------------

        if procesamiento is None:
            raise AnalizadorError(
                "Se requiere una instancia válida de Procesamiento."
            )

        if not isinstance(procesamiento, Procesamiento):
            raise TypeError(
                "procesamiento debe ser una instancia de Procesamiento."
            )

        self.procesamiento = procesamiento

        # --------------------------------------------------
        # 2. Comprobar que exista el interpolador PCHIP
        # --------------------------------------------------

        if getattr(self.procesamiento, "_hermite", None) is None:
            raise ValueError(
                "Primero debes construir el interpolador PCHIP "
                "en Procesamiento."
            )

        # --------------------------------------------------
        # 3. Copiar y validar los amortiguadores
        # --------------------------------------------------

        self.buffers = dict(buffers) if buffers is not None else {}

        for nombre, modelo in self.buffers.items():

            if not isinstance(nombre, str) or not nombre.strip():
                raise ValueError(
                    "Cada amortiguador debe tener un nombre válido."
                )

            if not isinstance(
                modelo,
                (ModeloCinetico, ModeloCineticoCompetitivo)
            ):
                raise TypeError(
                    f"El amortiguador '{nombre}' no es una instancia "
                    "de ModeloCinetico ni ModeloCineticoCompetitivo."
                )

            if modelo.procesamiento is not self.procesamiento:
                raise ValueError(
                    f"El amortiguador '{nombre}' está asociado "
                    "a otro objeto Procesamiento."
                )

        # --------------------------------------------------
        # 4. Copiar y validar los transportadores
        # --------------------------------------------------

        self.flujos = dict(flujos) if flujos is not None else {}

        for nombre, flujo in self.flujos.items():

            if not isinstance(nombre, str) or not nombre.strip():
                raise ValueError(
                    "Cada flujo debe tener un nombre válido."
                )

            if not isinstance(flujo, Recapturacion):
                raise TypeError(
                    f"El flujo '{nombre}' debe ser una instancia "
                    "de Recapturacion."
                )

        # --------------------------------------------------
        # 5. Inicializar la estructura de resultados
        # --------------------------------------------------

        self.resultados: dict[str, object] = {
            "tiempo": np.array([]),
            "calcio_libre": np.array([]),
            "buffers": {},
            "flujos": {}
        }
    def agregar_buffer(self, nombre: str, buffer: ModeloCinetico) -> None:
        """Agrega o reemplaza un objeto ModeloCinetico en el diccionario."""
        if not isinstance(buffer, ModeloCinetico):
            raise AnalizadorError(f"El objeto para '{nombre}' debe ser una instancia de ModeloCinetico.")
        self.buffers[nombre] = buffer

    def agregar_flujo(self, nombre: str, flujo: Recapturacion) -> None:
        """Agrega o reemplaza un objeto Recapturacion en el diccionario."""
        if not isinstance(flujo, Recapturacion):
            raise AnalizadorError(f"El objeto para '{nombre}' debe ser una instancia de Recapturacion.")
        self.flujos[nombre] = flujo

    # ============================================================
    # CATEGORÍA A: DIAGNÓSTICO DE LA SEÑAL EXPERIMENTAL
    # ============================================================

    def graficar_interpolador(self) -> None:
        """Grafica los datos experimentales y la interpolación PCHIP."""
        proc = self.procesamiento
        tiempo_continuo = np.linspace(proc.tiempo[0], proc.tiempo[-1], 1000)
        calcio_continuo = proc.evaluar_calcio(tiempo_continuo)

        plt.figure(figsize=(9, 5))
        plt.scatter(
            proc.tiempo, proc.calcio_limpio, 
            color="blue", label="Datos experimentales", s=25, zorder=3
        )
        plt.plot(
            tiempo_continuo, calcio_continuo, 
            color="red", linewidth=2, label="Interpolación PCHIP"
        )
        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Concentración de calcio (µM)")
        plt.title("Interpolación PCHIP del transitorio de calcio")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def graficar_derivada(self, t_destacado: float | None = None) -> None:
        """Grafica la derivada temporal d[Ca2+]/dt del interpolador PCHIP."""
        proc = self.procesamiento
        tiempo_continuo = np.linspace(proc.tiempo[0], proc.tiempo[-1], 1000)
        derivada = proc.derivada_interpolador(tiempo_continuo)

        plt.figure(figsize=(9, 5))
        plt.plot(
            tiempo_continuo, derivada, 
            color="darkorange", linewidth=2, label="Derivada PCHIP"
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)

        if t_destacado is not None:
            if not np.isfinite(t_destacado):
                raise ValueError("El tiempo destacado debe ser finito.")
            if not proc.tiempo[0] <= t_destacado <= proc.tiempo[-1]:
                raise ValueError("El tiempo destacado está fuera del intervalo.")

            valor_derivada = proc.derivada_interpolador(t_destacado)
            plt.scatter(
                t_destacado, valor_derivada, 
                color="red", s=70, zorder=3, label=f"Derivada a {t_destacado} ms"
            )

        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Variación del calcio (µM/ms)")
        plt.title("Derivada temporal del calcio citosólico")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def graficar_integracion(self, t0: float, tf: float) -> float:
        """Grafica el área bajo la curva (AUC) de calcio entre t0 y tf."""
        proc = self.procesamiento
        auc = proc.area_bajo_curva(t0, tf)

        tiempo_integracion = np.linspace(t0, tf, 500)
        calcio_integracion = proc.evaluar_calcio(tiempo_integracion)

        tiempo_completo = np.linspace(proc.tiempo[0], proc.tiempo[-1], 1000)
        calcio_completo = proc.evaluar_calcio(tiempo_completo)

        plt.figure(figsize=(9, 5))
        plt.plot(
            proc.tiempo, proc.calcio_limpio, 
            color="blue", alpha=0.5, label="Datos experimentales"
        )
        plt.plot(
            tiempo_completo, calcio_completo, 
            color="red", linewidth=2, label="Interpolación PCHIP"
        )
        plt.fill_between(
            tiempo_integracion, calcio_integracion, 0, 
            color="orange", alpha=0.4, label="Área integrada"
        )
        plt.xlabel("Tiempo (ms)")
        plt.ylabel("Concentración de calcio (µM)")
        plt.title("Exposición acumulada de calcio citosólico")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.text(
            0.03, 0.95, f"AUC = {auc:.6f} µM·ms",
            transform=plt.gca().transAxes, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8)
        )
        plt.tight_layout()
        plt.show()

        return auc

    # ============================================================
    # CATEGORÍA B: SIMULACIÓN COMPLETA DEL SISTEMA
    # ============================================================

    def simular_sistema_completo(self, t_span: np.ndarray | None = None) -> dict:
        """
        Ejecuta la simulación temporal coordinada de todos los componentes:
        1. Evalúa la curva continua de [Ca2+](t).
        2. Integra [SCa](t) para cada buffer pasivo llamando a resolver() o simular().
        3. Calcula J_bomba(t), J_neto(t) e integra la masa acumulada.
        4. Empaqueta todo en self.resultados.
        """
        try:
            if t_span is None:
                tiempos = np.copy(self.procesamiento.tiempo)
            else:
                tiempos = np.asarray(t_span, dtype=float)
                if tiempos.ndim != 1 or len(tiempos) < 2:
                    raise AnalizadorError("El vector t_span debe ser unidimensional y contener al menos 2 puntos.")
                if np.any(np.diff(tiempos) <= 0):
                    raise AnalizadorError("Los valores del tiempo t_span deben ser strictly crecientes.")
                if tiempos[0] < self.procesamiento.tiempo[0] or tiempos[-1] > self.procesamiento.tiempo[-1]:
                    raise AnalizadorError("El intervalo de t_span está fuera del rango experimental de Procesamiento.")

            ca_libre = self.procesamiento.evaluar_calcio(tiempos)
            
            self.resultados["tiempo"] = tiempos
            self.resultados["calcio_libre"] = ca_libre
            self.resultados["buffers"] = {}
            self.resultados["flujos"] = {}

            # 2. Integrar EDOs de buffers pasivos
            for nombre, modelo in self.buffers.items():
                try:
                    # Soporte multi-método para ModeloCinetico (resolver vs simular vs precalculado)
                    if getattr(modelo, "calcio_unido", None) is not None:
                        ca_unido = modelo.calcio_unido
                    elif hasattr(modelo, "resolver") and callable(modelo.resolver):
                        solucion = modelo.resolver()
                        ca_unido = modelo.calcio_unido if hasattr(modelo, "calcio_unido") else (solucion.y[0] if hasattr(solucion, "y") else solucion)
                    elif hasattr(modelo, "simular") and callable(modelo.simular):
                        solucion = modelo.simular(tiempos, self.procesamiento)
                        ca_unido = solucion.y[0] if hasattr(solucion, "y") else (solucion if isinstance(solucion, np.ndarray) else modelo.calcio_unido)
                    else:
                        raise AnalizadorError(f"El objeto buffer '{nombre}' no posee un método numérico ejecutable.")

                    s_tot = getattr(modelo, "S_total", None) or getattr(modelo, "sitios_totales", 1.0)
                    fraccion_ocupacion = ca_unido / s_tot

                    self.resultados["buffers"][nombre] = {
                        "calcio_unido": ca_unido,
                        "fraccion_ocupacion": fraccion_ocupacion,
                        "sitios_totales": s_tot,
                        "kon": getattr(modelo, "kon", 0.0),
                        "koff": getattr(modelo, "koff", 0.0)
                    }
                except Exception as e:
                    raise AnalizadorError(f"Error al integrar el buffer '{nombre}': {str(e)}")

            # 3. Evaluar flujos de recaptación activa y masa acumulada
            for nombre, flujo in self.flujos.items():
                try:
                    j_bomba = flujo.evaluar_flujo_bomba(ca_libre)
                    j_neto = flujo.evaluar_flujo_neto(ca_libre)
                    masa_acumulada = cumulative_trapezoid(j_neto, tiempos, initial=0.0)

                    self.resultados["flujos"][nombre] = {
                        "flujo_bomba": j_bomba,
                        "flujo_neto": j_neto,
                        "masa_acumulada": masa_acumulada,
                        "Vmax": flujo.Vmax,
                        "Km": flujo.Km,
                        "n_hill": flujo.n_hill,
                        "J_leak": flujo.J_leak
                    }
                except Exception as e:
                    raise AnalizadorError(f"Error al evaluar el flujo de recaptación '{nombre}': {str(e)}")

            return self.resultados

        except Exception as e:
            if isinstance(e, AnalizadorError):
                raise e
            raise AnalizadorError(f"Falla inesperada durante simular_sistema_completo: {str(e)}")

    def graficar_reporte_completo(self) -> None:
        """Grafica un reporte multipanel completo de la simulación."""
        if not self.resultados["tiempo"].size:
            self.simular_sistema_completo()

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        t = self.resultados["tiempo"]
        ca_libre = self.resultados["calcio_libre"]

        # Panel 1: Calcio Libre y Buffers Pasivos
        axes[0].plot(t, ca_libre, "k--", label=r"$[\mathrm{Ca}^{2+}]_{\mathrm{libre}}$", alpha=0.7, linewidth=1.5)
        for nombre, b_data in self.resultados["buffers"].items():
            axes[0].plot(t, b_data["calcio_unido"], label=f"Unido a {nombre}")
        axes[0].set_ylabel(r"Concentración ($\mu\mathrm{M}$)")
        axes[0].set_title("1. Dinámica de Amortiguamiento Pasivo")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # Panel 2: Tasas Netas de Recaptación Activa
        for nombre, f_data in self.resultados["flujos"].items():
            axes[1].plot(t, f_data["flujo_neto"], label=f"J_neto {nombre}")
        axes[1].set_ylabel(r"Flujo Neto ($\mu\mathrm{M/ms}$)")
        axes[1].set_title("2. Tasas Netas de Recaptación Activa (Bomba - Fuga)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle=":", alpha=0.6)

        # Panel 3: Masa Acumulada Recapturada
        for nombre, f_data in self.resultados["flujos"].items():
            axes[2].plot(t, f_data["masa_acumulada"], label=f"Masa Recapturada {nombre}", linewidth=2)
        axes[2].set_xlabel("Tiempo (ms)")
        axes[2].set_ylabel(r"Masa Acumulada ($\mu\mathrm{M}$)")
        axes[2].set_title("3. Masa Acumulada de Calcio Secuestrado")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.show()

    # ============================================================
    # CATEGORÍA C: MENÚ INTERACTIVO CON ACCESO A TODAS LAS GRÁFICAS
    # ============================================================

    def menu_interactivo(self) -> None:
        """
        Ejecuta la simulación completa y abre una interfaz de consola interactiva
        para inspeccionar datos y visualizar cualquiera de los 4 tipos de gráficas disponibles.
        """
        print("\n=======================================================")
        print("    INICIANDO SIMULACIÓN COMPLETA DEL SISTEMA          ")
        print("=======================================================")

        try:
            self.simular_sistema_completo()
            print("✔ Simulación del sistema completada con éxito.")
        except AnalizadorError as err:
            print(f"❌ Error crítico en la simulación: {err}")
            return

        while True:
            print("\n-------------------------------------------------------")
            print(" MENU INTERACTIVO - DEPURACIÓN Y ANÁLISIS PASO A PASO  ")
            print("-------------------------------------------------------")
            print("1. Inspeccionar datos de entrada ([Ca2+] libre experimental)")
            print("2. Inspeccionar un Amortiguador Pasivo (Buffers)")
            print("3. Inspeccionar un Mecanismo de Recaptación (SERCA / MCU)")
            print("4. Graficar Datos e Interpolación PCHIP")
            print("5. Graficar Derivada Temporal (d[Ca2+]/dt)")
            print("6. Graficar Área Bajo la Curva (AUC Integrada)")
            print("7. Mostrar Reporte Gráfico Completo del Sistema")
            print("8. Salir")

            opcion = input("\nSeleccione una opción (1-8): ").strip()

            if opcion == "1":
                self._inspeccionar_calcio_libre()
            elif opcion == "2":
                self._inspeccionar_buffer_interactivo()
            elif opcion == "3":
                self._inspeccionar_flujo_interactivo()
            elif opcion == "4":
                self.graficar_interpolador()
            elif opcion == "5":
                t_input = input("Ingrese instante t a destacar (o presione Enter para omisión): ").strip()
                t_val = float(t_input) if t_input else None
                try:
                    self.graficar_derivada(t_destacado=t_val)
                except ValueError as ve:
                    print(f"❌ Error: {ve}")
            elif opcion == "6":
                try:
                    t0_in = input("Tiempo inicial t0 (ms) [Enter para inicio]: ").strip()
                    tf_in = input("Tiempo final tf (ms) [Enter para final]: ").strip()
                    t0 = float(t0_in) if t0_in else self.procesamiento.tiempo[0]
                    tf = float(tf_in) if tf_in else self.procesamiento.tiempo[-1]
                    self.graficar_integracion(t0, tf)
                except ValueError as ve:
                    print(f"❌ Error en tiempos de integración: {ve}")
            elif opcion == "7":
                self.graficar_reporte_completo()
            elif opcion == "8":
                print("\nSaliendo del Analizador Interactivo. ¡Simulación finalizada!")
                break
            else:
                print("⚠️ Opción no válida. Intente de nuevo.")

    def _inspeccionar_calcio_libre(self) -> None:
        """Muestra métricas clave de la señal libre."""
        t = self.resultados["tiempo"]
        ca = self.resultados["calcio_libre"]
        auc = self.procesamiento.area_bajo_curva(t[0], t[-1])

        print("\n--- DATOS DE [Ca2+] LIBRE EXPERIMENTAL ---")
        print(f"Puntos temporales : {len(t)}")
        print(f"Tiempo inicial    : {t[0]:.4f} ms")
        print(f"Tiempo final      : {t[-1]:.4f} ms")
        print(f"Calcio basal      : {ca[0]:.6f} µM")
        print(f"Calcio máximo     : {np.max(ca):.6f} µM")
        print(f"Área bajo la curva: {auc:.6f} µM·ms")

    def _inspeccionar_buffer_interactivo(self) -> None:
        """Permite al usuario elegir e inspeccionar un buffer específico."""
        if not self.resultados["buffers"]:
            print("⚠️ No hay buffers registrados en la simulación.")
            return

        print("\nBuffers disponibles:")
        nombres = list(self.resultados["buffers"].keys())
        for idx, nombre in enumerate(nombres, 1):
            print(f"  {idx}. {nombre}")

        try:
            sel = input("\nIngrese el número o nombre del buffer: ").strip()
            if sel.isdigit():
                nombre = nombres[int(sel) - 1]
            else:
                nombre = sel

            if nombre not in self.resultados["buffers"]:
                raise KeyError(f"El buffer '{nombre}' no existe.")

            b_data = self.resultados["buffers"][nombre]
            sca = b_data["calcio_unido"]
            frac = b_data["fraccion_ocupacion"]

            print(f"\n--- DETALLES DEL BUFFER: {nombre} ---")
            print(f"Parámetros        : kon={b_data['kon']} µM^-1 ms^-1, koff={b_data['koff']} ms^-1, Stot={b_data['sitios_totales']} µM")
            print(f"Calcio unido ini  : {sca[0]:.6f} µM")
            print(f"Calcio unido máx  : {np.max(sca):.6f} µM")
            print(f"Calcio unido fin  : {sca[-1]:.6f} µM")
            print(f"Ocupación máxima  : {np.max(frac)*100:.2f} %")

        except (IndexError, KeyError, ValueError) as e:
            print(f"❌ Error al seleccionar buffer: Entrada inválida ({e}).")

    def _inspeccionar_flujo_interactivo(self) -> None:
        """Permite al usuario elegir e inspeccionar un flujo activo específico."""
        if not self.resultados["flujos"]:
            print("⚠️ No hay flujos de recaptación registrados en la simulación.")
            return

        print("\nFlujos de recaptación disponibles:")
        nombres = list(self.resultados["flujos"].keys())
        for idx, nombre in enumerate(nombres, 1):
            print(f"  {idx}. {nombre}")

        try:
            sel = input("\nIngrese el número o nombre del flujo: ").strip()
            if sel.isdigit():
                nombre = nombres[int(sel) - 1]
            else:
                nombre = sel

            if nombre not in self.resultados["flujos"]:
                raise KeyError(f"El flujo '{nombre}' no existe.")

            f_data = self.resultados["flujos"][nombre]
            j_bomba = f_data["flujo_bomba"]
            j_neto = f_data["flujo_neto"]
            masa = f_data["masa_acumulada"]

            print(f"\n--- DETALLES DEL FLUJO: {nombre} ---")
            print(f"Parámetros        : Vmax={f_data['Vmax']} µM/ms, Km={f_data['Km']} µM, n_Hill={f_data['n_hill']}")
            print(f"Fuga Pasiva J_leak: {f_data['J_leak']:.8f} µM/ms")
            print(f"Flujo bomba máx   : {np.max(j_bomba):.6f} µM/ms")
            print(f"Flujo neto máx    : {np.max(j_neto):.6f} µM/ms")
            print(f"Masa acumulada fin: {masa[-1]:.6f} µM")

        except (IndexError, KeyError, ValueError) as e:
            print(f"❌ Error al seleccionar flujo: Entrada inválida ({e}).")