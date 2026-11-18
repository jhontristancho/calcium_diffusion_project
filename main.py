import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calcium_kinetics.analizador import Analizador


def main() -> None:
    ruta_datos = ROOT / "data" / "raw"

    experimentos = {
        "transientph7-33.csv": Analizador.desde_experimento(
            ruta_csv=ruta_datos / "transientph7-33.csv",
            pH=7.33,
        ),
        "transientph6-5.csv": Analizador.desde_experimento(
            ruta_csv=ruta_datos / "transientph6-5.csv",
            pH=6.50,
        ),
    }
    print(" \n elije con que archivo quieres trabajar: \n")
    Analizador.ejecutar_sesion_interactiva(experimentos)


if __name__ == "__main__":
    main()