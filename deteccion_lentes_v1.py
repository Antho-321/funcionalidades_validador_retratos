#!/usr/bin/env python3
"""
Detección de lentes en imágenes usando Autodistill + Detic.
Requisitos:
    pip install autodistill-detic supervision opencv-python torch torchvision tqdm
    (y clonar el repo oficial de Detic en una sub-carpeta ./Detic)
"""
import os, sys, urllib.request, shutil, tqdm, torch, supervision as sv, cv2
from pathlib import Path
from autodistill_detic import DETIC
from autodistill.detection import CaptionOntology

# ───────────────── 1. Pesos ──────────────────────────────────────────────────
WEIGHTS_NAME = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
WEIGHTS_URL  = f"https://dl.fbaipublicfiles.com/detic/{WEIGHTS_NAME}"
CACHE_DIR    = Path.home()/".cache"/"autodistill"/"Detic"/"models"
WEIGHTS_LOCAL= CACHE_DIR/WEIGHTS_NAME

if not WEIGHTS_LOCAL.exists():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Descargando {WEIGHTS_NAME} …")
    with urllib.request.urlopen(WEIGHTS_URL) as r, open(WEIGHTS_LOCAL,"wb") as f:
        total=int(r.headers["Content-Length"])
        with tqdm.tqdm(total=total,unit="B",unit_scale=True) as p:
            while (chunk:=r.read(8192)):
                f.write(chunk); p.update(len(chunk))
    print(f"[INFO] ✅  Pesos en {WEIGHTS_LOCAL}")
else:
    print(f"[INFO] ✅  Pesos ya presentes en {WEIGHTS_LOCAL}")

# ───────────────── 2. Rutas Detic ────────────────────────────────────────────
detic_root = Path(__file__).parent/"Detic"          # repo clonado
cfg_file   = detic_root/"configs"/WEIGHTS_NAME.replace(".pth",".yaml")
print(f"[INFO] Detic root            : {detic_root} | existe → {detic_root.is_dir()}")
print(f"[INFO] YAML original         : {cfg_file}  | existe → {cfg_file.exists()}")

# Añade Detic al PYTHONPATH
if str(detic_root) not in sys.path:
    sys.path.insert(0, str(detic_root))
    print(f"[INFO] sys.path ← {detic_root}")

# ─────────── 3. Copia YAML al directorio de trabajo ./configs ────────────────
local_cfg_dir = Path.cwd()/"configs"
local_cfg_dir.mkdir(exist_ok=True)
target_yaml   = local_cfg_dir/cfg_file.name
if not target_yaml.exists():
    shutil.copy(cfg_file, target_yaml)
    print(f"[INFO] Copiado YAML → {target_yaml}")
else:
    print(f"[INFO] YAML ya presente en {target_yaml}")

# (¡NO cambiamos de directorio! El cwd sigue siendo {Path.cwd()})
print(f"[INFO] Directorio de trabajo actual → {Path.cwd()}")

# ─────────── 4. Construir modelo Detic ───────────────────────────────────────
ontology    = CaptionOntology({"eyeglasses":"eyeglasses"})
print("[INFO] Creando modelo DETIC …")
detic_model = DETIC(ontology=ontology)              # ahora sí inicia
device      = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Modelo listo en {device}")

# ─────────── 5. Utilidades de inferencia ─────────────────────────────────────
def get_glasses_probability(ruta_imagen:str, umbral_min:float=0.0)->float:
    #print(f"[INFO] Inferencia sobre {ruta_imagen}")
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(ruta_imagen)
    dets: sv.Detections = detic_model.predict(ruta_imagen)
    confs = [c for c in (dets.confidence or []) if c>=umbral_min]
    #print(f"[INFO] Confianzas filtradas (≥{umbral_min}): {confs}")
    return max(confs, default=0.0)

def verificar_presencia_de_lentes(ruta_imagen:str, umbral:float=0.5)->str:
    prob = get_glasses_probability(ruta_imagen, umbral)
    msg  = (f"Imagen contiene lentes (prob.≈{prob:.2f})"
            if prob>=umbral else
            f"Imagen NO contiene lentes (prob.≈{prob:.2f})")
    print(f"[INFO] {msg}"); return msg

# ─────────── 6. Ejemplo rápido ───────────────────────────────────────────────
if __name__=="__main__":
    ruta = (
        r"C:\Users\Administrador\Documents\INGENIERIA_EN_SOFTWARE"
        r"\PROYECTO_FOTOGRAFIAS_ESTUDIANTES\datasets\validated_color\1726138652.jpg"
    )
    verificar_presencia_de_lentes(ruta)
