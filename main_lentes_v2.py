#!/usr/bin/env python3
import concurrent.futures
import os
import io
import sys
import traceback
from queue import Queue
from typing import List, Tuple
from tqdm import tqdm

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from deteccion_lentes_v2 import (
    get_glasses_probability,
    get_glasses_probability_batch,
    configurar_optimizaciones_gpu,
    warm_up_modelo,
    obtener_estadisticas_cache
)

from exportacion_datos_excel import (
    format_to_hyperlinks,
    normalize_dict_lengths,
    dict_to_excel,
    get_file_count,
)

# ── Constantes Globales ───────────────────────────────────────────────────────
UMBRAL_DETECCION_LENTES = 0.4486

# ── Google Drive ──────────────────────────────────────────────────────────────
SCOPES     = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"

# --- Constantes para el Caché ------------------------------------------------
# <--- NUEVO: Directorio para almacenar las imágenes y no volver a bajarlas.
CACHE_DIR = "image_cache"

def drive_service():
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    else:
        flow  = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds)

def get_folder_id_by_path(path: str, drive):
    segments  = [s for s in path.strip("/").split("/") if s and s != "Mi unidad"]
    parent_id = "root"
    for name in segments:
        resp = drive.files().list(
            q=(
                f"name = '{name}' and "
                "mimeType = 'application/vnd.google-apps.folder' and "
                f"'{parent_id}' in parents and trashed = false"
            ),
            fields="files(id)",
            pageSize=1,
        ).execute()
        items = resp.get("files", [])
        if not items:
            raise FileNotFoundError(f"Carpeta '{name}' no encontrada (parent={parent_id})")
        parent_id = items[0]["id"]
    return parent_id

def list_files_recursive(folder_id: str, drive) -> List[Tuple[str, str]]:
    results = []
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        resp = drive.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
        ).execute()
        for f in resp["files"]:
            if f["mimeType"] == "application/vnd.google-apps.folder":
                results.extend(list_files_recursive(f["id"], drive))
            else:
                results.append((f["id"], f["name"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results

def download_file_optimized(file_id: str, dest_path: str, drive, chunk_size: int = 10 * 1024 * 1024):
    request = drive.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()

def download_files_parallel(
    files: List[Tuple[str, str]],
    download_dir: str, # <--- Renombrado para mayor claridad
    drive_service_func,
    max_workers: int = 4
) -> List[str]:
    valid_ext = {'.jpg','.jpeg','.png','.bmp','.tiff','.webp'}
    # Nota: El filtrado de 'valid_files' ahora se hace en la función principal
    if not files:
        print("[WARNING] No hay imágenes válidas para descargar.")
        return []

    image_paths = []
    errors = 0

    def _download_task(file_id: str, name: str):
        drive = drive_service_func()
        local_path = os.path.join(download_dir, name)
        try:
            download_file_optimized(file_id, local_path, drive)
            return local_path
        except Exception as e:
            raise RuntimeError(f"Error al descargar '{name}': {e}") from e

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(_download_task, fid, name): name
            for fid, name in files
        }

        progress_bar = tqdm(
            concurrent.futures.as_completed(future_to_name),
            total=len(files),
            desc="Descargando",
            unit="archivo"
        )

        for future in progress_bar:
            try:
                path = future.result()
                image_paths.append(path)
            except Exception as e:
                print(f"[ERROR] {e}")
                errors += 1
            
            progress_bar.set_postfix(exitosos=len(image_paths), errores=errors)

    print(f"[INFO] Descarga completa: {len(image_paths)} éxitos, {errors} errores")
    return image_paths

# ── Versión optimizada con descarga paralela Y CACHÉ MEJORADO ──────────────────
def process_drive_folder_optimized(
    drive_folder_path: str,
    usar_batch: bool = True,
    umbral_minimo: float = 0.0,
    max_workers: int = 4,
    forzar_descarga: bool = False
) -> Tuple[List[str], List[float]]:
    """
    Procesa imágenes de una carpeta de Drive, usando un caché local para
    evitar descargas repetidas de archivos ya existentes.
    """
    print("[INFO] 🚀 Iniciando procesamiento ultra-optimizado...")
    configurar_optimizaciones_gpu()
    warm_up_modelo()

    drive = drive_service()
    folder_id = get_folder_id_by_path(drive_folder_path, drive)
    
    print("[INFO] Obteniendo lista de archivos remotos de Google Drive...")
    remote_files = list_files_recursive(folder_id, drive)
    if not remote_files:
        print("[ERROR] No se encontraron archivos en la carpeta de Drive.")
        return [], []

    print(f"[INFO] Encontrados {len(remote_files)} archivos en Drive.")
    
    # --- LÓGICA DE CACHÉ MEJORADA ---
    os.makedirs(CACHE_DIR, exist_ok=True) 

    valid_ext = {'.jpg','.jpeg','.png','.bmp','.tiff','.webp'}
    remote_image_files = [
        (fid, name) for fid, name in remote_files 
        if any(name.lower().endswith(ext) for ext in valid_ext)
    ]

    # Si se fuerza la descarga, vaciamos la carpeta de caché primero
    if forzar_descarga:
        print("[INFO] ⚠️ Forzando nueva descarga. Limpiando caché local...")
        for f in os.listdir(CACHE_DIR):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except OSError as e:
                print(f"[WARNING] No se pudo eliminar {f} del caché: {e}")

    # Creamos dos listas: una para los archivos que ya tenemos y otra para los que faltan
    files_to_download = []
    cached_image_paths = []

    print("[INFO] 🔎 Verificando caché local...")
    for file_id, file_name in remote_image_files:
        local_path = os.path.join(CACHE_DIR, file_name)
        if os.path.exists(local_path) and not forzar_descarga:
            cached_image_paths.append(local_path)
        else:
            # Si no existe localmente o se fuerza la descarga, lo añadimos a la cola
            files_to_download.append((file_id, file_name))

    if cached_image_paths:
        print(f"[INFO] ✅ {len(cached_image_paths)} archivos encontrados en el caché.")

    # Descargamos solo los archivos que no están en el caché
    if files_to_download:
        print(f"[INFO] 📥 Se descargarán {len(files_to_download)} archivos nuevos o faltantes.")
        downloaded_paths = download_files_parallel(
            files_to_download, CACHE_DIR, drive_service, max_workers
        )
        image_paths = cached_image_paths + downloaded_paths
    else:
        print("[INFO] ✅ El caché local ya está completo. No se necesitan descargas.")
        image_paths = cached_image_paths

    # Ordenamos la lista final para mantener consistencia
    image_paths.sort()

    if not image_paths:
        print("[WARNING] No hay imágenes válidas para procesar.")
        return [], []

    print(f"[INFO] ✅ Listas {len(image_paths)} imágenes para procesar.")
    print("[INFO] 🔍 Iniciando detección de lentes...")

    # --- FASE 2: Detección ---
    if usar_batch and len(image_paths) > 1:
        glasses_probs = get_glasses_probability_batch(image_paths)
        pos = sum(1 for p in glasses_probs if isinstance(p, (int, float)) and p >= UMBRAL_DETECCION_LENTES)
        print(f"[INFO] ✅ Batch completado ({pos}/{len(glasses_probs)} positivos)")
    else:
        glasses_probs: List[float] = []
        for path in tqdm(image_paths, desc="Detectando lentes", unit="imagen"):
            try:
                glasses_probs.append(get_glasses_probability(path))
            except Exception as e:
                print(f"[ERROR] {path}: {e}")
                glasses_probs.append(0.0)

    # --- FASE 3: Estadísticas finales ---
    print("\n[INFO] 📈 Estadísticas finales:")
    obtener_estadisticas_cache()

    numeric_probs = [p for p in glasses_probs if isinstance(p, (int, float))]
    total = len(numeric_probs)

    if total > 0:
        con_lentes = sum(1 for p in numeric_probs if p >= UMBRAL_DETECCION_LENTES)
        sin_lentes = total - con_lentes
        prom = sum(numeric_probs) / total
        
        porc_con_lentes = (con_lentes / total) * 100
        porc_sin_lentes = (sin_lentes / total) * 100

        print(f"👓 Con lentes: {con_lentes} ({porc_con_lentes:.1f}%)")
        print(f"👁️ Sin lentes: {sin_lentes} ({porc_sin_lentes:.1f}%)")
        print(f"📊 Promedio: {prom:.3f}")
    else:
        print("⚠️ No se procesaron imágenes de forma exitosa. No se pueden calcular estadísticas.")

    return image_paths, glasses_probs

# ── Interfaz compatible ──────────────────────────────────────────────────────
def process_drive_folder(drive_folder_path: str) -> Tuple[List[str], List[float]]:
    return process_drive_folder_optimized(drive_folder_path, usar_batch=True)

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset_drive_path = (
        "/Mi unidad/INGENIERIA_EN_SOFTWARE/5to_Semestre/"
        "PRACTICAS/Primera_Revision/"
        "validator/results/sin_procesar"
    )
    USAR_BATCH = True
    UMBRAL_MIN = 0.0
    MAX_THREADS = 6
    # <--- NUEVO: Parámetro para forzar la descarga
    FORZAR_NUEVA_DESCARGA = True

    os.makedirs("results", exist_ok=True)

    try:
        paths, probs = process_drive_folder_optimized(
            dataset_drive_path,
            usar_batch=USAR_BATCH,
            umbral_minimo=UMBRAL_MIN,
            max_workers=MAX_THREADS,
            forzar_descarga=FORZAR_NUEVA_DESCARGA # <--- Uso del nuevo parámetro
        )
        if not paths:
            sys.exit(1)

        ordered_probs = [probs[p] for p in paths] if isinstance(probs, dict) else probs

        info = {
            "Rutas": format_to_hyperlinks(paths),
            "Probabilidad": ordered_probs, # Now this is a list, as expected
            "Detección": [
                "SÍ" if isinstance(p, (int, float)) and p >= UMBRAL_DETECCION_LENTES 
                else "NO" 
                for p in ordered_probs # Iterate over the new list
            ]
        }
        normalized = normalize_dict_lengths(info)
        out = dict_to_excel(
            normalized,
            f"results/Reporte_{get_file_count('results')+1}.xlsx"
        )
        print(f"✅ Listo. Excel generado en: {out}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por usuario")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n[ERROR DE RUTA] No se pudo encontrar una carpeta en Google Drive: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR INESPERADO] Se ha producido un error. Detalles:")
        # 2. Imprimir la traza completa del error
        traceback.print_exc()
        sys.exit(1)