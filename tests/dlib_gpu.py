#!/usr/bin/env python3
# check_dlib_gpu.py
# A script to verify full dlib GPU (CUDA) compatibility.

import sys
from pathlib import Path

import numpy as np

# --- Configuration ---
# The script expects the model to be in a 'models' subfolder.
# Download from: http://dlib.net/files/mmod_human_face_detector.dat.bz2
try:
    # To this:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

MODELS_DIR = BASE_DIR / "models"
CNN_FACE_DETECTOR_PATH = MODELS_DIR / "mmod_human_face_detector.dat"

# --- Main Verification Logic ---
def check_dlib_gpu_compatibility():
    """
    Performs a series of checks to verify full dlib GPU compatibility.
    """
    print("--- Verificador de Compatibilidad de dlib con GPU ---")

    # This is a critical first step. If dlib is not available, nothing else matters.
    try:
        import dlib
    except ImportError:
        print("\n❌ FALLO CRÍTICO: La librería 'dlib' no está instalada.")
        print("   -> Solución: Instala dlib usando 'pip install dlib' o compílala desde la fuente.")
        sys.exit(1)

    # 1. Check if dlib was compiled with CUDA support
    print("\n[Paso 1/4] Verificando la compilación de dlib con CUDA...")
    if dlib.DLIB_USE_CUDA:
        print("   ✅ ÉXITO: Tu instalación de dlib fue compilada con soporte para CUDA.")
    else:
        print("\n❌ FALLO: Tu instalación de dlib NO fue compilada con soporte para CUDA.")
        print("   -> Solución: Reinstala dlib desde la fuente, asegurándote de que CMake encuentre tu kit de herramientas CUDA.")
        sys.exit(1)

    # 2. Check if dlib can detect CUDA-enabled devices
    print("\n[Paso 2/4] Detectando dispositivos GPU disponibles...")
    try:
        num_devices = dlib.cuda.get_num_devices()
        if num_devices > 0:
            print(f"   ✅ ÉXITO: dlib detectó {num_devices} dispositivo(s) GPU compatibles con CUDA.")
        else:
            print("\n❌ FALLO: dlib no detectó ningún dispositivo GPU compatible con CUDA.")
            print("   -> Solución: Asegúrate de que los drivers de NVIDIA estén instalados y actualizados.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ FALLO: Ocurrió un error al detectar dispositivos GPU: {e}")
        print("   -> Solución: Puede haber un problema con la instalación de tus drivers de NVIDIA o el kit de herramientas CUDA.")
        sys.exit(1)

    # 3. Check if the required CNN model file exists
    print("\n[Paso 3/4] Verificando la existencia del modelo CNN facial...")
    if not CNN_FACE_DETECTOR_PATH.exists():
        print(f"\n❌ FALLO: No se encontró el archivo del modelo en: {CNN_FACE_DETECTOR_PATH}")
        print(f"   -> Solución: Descarga 'mmod_human_face_detector.dat.bz2', descomprímelo y colócalo en la carpeta '{MODELS_DIR}'.")
        print("      Link de descarga: http://dlib.net/files/mmod_human_face_detector.dat.bz2")
        # Create the directory if it doesn't exist to help the user
        if not MODELS_DIR.exists():
            print(f"   [INFO] Creando el directorio '{MODELS_DIR}' para ti.")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        sys.exit(1)
    else:
        print(f"   ✅ ÉXITO: Modelo CNN encontrado en '{CNN_FACE_DETECTOR_PATH}'.")

    # 4. Attempt to load the model and perform a test inference
    print("\n[Paso 4/4] Cargando el modelo en la GPU y realizando una prueba de inferencia...")
    try:
        detector = dlib.cnn_face_detection_model_v1(str(CNN_FACE_DETECTOR_PATH))
        print("   - Modelo CNN cargado exitosamente en la memoria.")

        # Create a dummy image for a warm-up inference
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run the detector. This loads the model onto the GPU VRAM and tests it.
        detector(dummy_image, 1)
        print("   - Inferencia de prueba en la GPU completada con éxito.")

    except RuntimeError as e:
        print(f"\n❌ FALLO: Ocurrió un error en tiempo de ejecución al usar el modelo en la GPU: {e}")
        print("   -> Solución: Este error a menudo indica una incompatibilidad entre la versión de CUDA con la que se compiló dlib,")
        print("      la versión de tu driver de NVIDIA, y la capacidad de cómputo de tu GPU (Compute Capability).")
        print("      Verifica que todos los componentes (drivers, CUDA Toolkit, cuDNN) sean compatibles entre sí.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FALLO: Ocurrió un error inesperado: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("🎉 ¡FELICITACIONES! Tu entorno está correctamente configurado para usar dlib con aceleración por GPU.")
    print("="*60)

if __name__ == "__main__":
    check_dlib_gpu_compatibility()