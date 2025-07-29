#!/usr/bin/env python3
import concurrent.futures
import os
import io
import sys
import tempfile
import threading
from queue import Queue
from typing import List, Tuple
from tqdm import tqdm

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from deteccion_lentes_v1 import (
    get_glasses_probability,
    get_glasses_probability_batch,
    configurar_optimizaciones_gpu,
    warm_up_modelo,
    obtener_estadisticas_cache,
    limpiar_cache_imagenes
)

from exportacion_datos_excel import (
    format_to_hyperlinks,
    normalize_dict_lengths,
    dict_to_excel,
    get_file_count,
)

# ── Google Drive ──────────────────────────────────────────────────────────────
SCOPES     = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
CREDS_FILE = "credentials.json"

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

# OPTIMIZACIÓN: descarga con chunks grandes
def download_file_optimized(file_id: str, dest_path: str, drive, chunk_size: int = 10 * 1024 * 1024):
    request = drive.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()

# Worker para descarga en paralelo
def download_worker(q_in: Queue, q_out: Queue, drive_service_func, temp_dir: str):
    drive = drive_service_func()
    while True:
        item = q_in.get()
        if item is None:
            break
        file_id, name = item
        local_path = os.path.join(temp_dir, name)
        try:
            download_file_optimized(file_id, local_path, drive)
            q_out.put(('success', local_path))
        except Exception as e:
            q_out.put(('error', f"{name}: {e}"))
        finally:
            q_in.task_done()

# Función de ayuda: descargas paralelas
def download_files_parallel(
    files: List[Tuple[str, str]],
    temp_dir: str,
    drive_service_func,
    max_workers: int = 4
) -> List[str]:
    """
    Downloads files from Google Drive in parallel using a modern thread pool.
    """
    valid_ext = {'.jpg','.jpeg','.png','.bmp','.tiff','.webp'}
    valid_files = [
        (fid, name) for fid, name in files
        if any(name.lower().endswith(ext) for ext in valid_ext)
    ]
    if not valid_files:
        print("[WARNING] No hay imágenes válidas")
        return []

    image_paths = []
    errors = 0

    # This self-contained worker function will be executed by each thread.
    # It performs a single download task.
    def _download_task(file_id: str, name: str):
        # Each thread creates its own service client for thread safety.
        drive = drive_service_func()
        local_path = os.path.join(temp_dir, name)
        try:
            # Re-uses your existing optimized download logic.
            download_file_optimized(file_id, local_path, drive)
            return local_path
        except Exception as e:
            # Propagate the exception, which the main thread will catch.
            # This is cleaner than passing error messages through a queue.
            raise RuntimeError(f"Error al descargar '{name}': {e}") from e

    # ThreadPoolExecutor manages the entire lifecycle of the threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks to the pool. `submit` returns a Future object.
        future_to_name = {
            executor.submit(_download_task, fid, name): name
            for fid, name in valid_files
        }

        # `as_completed` yields futures as they finish, perfect for a progress bar.
        progress_bar = tqdm(
            concurrent.futures.as_completed(future_to_name),
            total=len(valid_files),
            desc="Descargando",
            unit="archivo"
        )

        for future in progress_bar:
            try:
                # Get the result from the completed future.
                # If the task raised an exception, .result() will re-raise it here.
                path = future.result()
                image_paths.append(path)
            except Exception as e:
                print(f"[ERROR] {e}")
                errors += 1
            
            # Update the progress bar's postfix with live results.
            progress_bar.set_postfix(exitosos=len(image_paths), errores=errors)

    print(f"[INFO] Descarga completa: {len(image_paths)} éxitos, {errors} errores")
    return image_paths

# ── Versión optimizada con descarga paralela ─────────────────────────────────
def process_drive_folder_optimized(
    drive_folder_path: str,
    usar_batch: bool = True,
    umbral_minimo: float = 0.0,
    max_workers: int = 4
) -> Tuple[List[str], List[float]]:
    print("[INFO] 🚀 Iniciando procesamiento ultra-optimizado...")
    configurar_optimizaciones_gpu()
    warm_up_modelo()

    drive = drive_service()
    folder_id = get_folder_id_by_path(drive_folder_path, drive)
    files = list_files_recursive(folder_id, drive)
    if not files:
        print("[ERROR] No se encontraron archivos.")
        return [], []

    print(f"[INFO] Encontrados {len(files)} archivos")

    temp_dir = tempfile.mkdtemp(prefix="glasses_optimized_")
    print(f"[INFO] Directorio temporal: {temp_dir}")

    # FASE 1: Descarga paralela
    image_paths = download_files_parallel(files, temp_dir, drive_service, max_workers)
    if not image_paths:
        print("[WARNING] No se descargaron imágenes válidas")
        return [], []

    print(f"[INFO] ✅ Descargadas {len(image_paths)} imágenes")
    print("[INFO] 🔍 Iniciando detección de lentes...")

    # FASE 2: Detección
    if usar_batch and len(image_paths) > 1:
        glasses_probs = get_glasses_probability_batch(image_paths, umbral_minimo)
        pos = sum(1 for p in glasses_probs if p > 0.5)
        print(f"[INFO] ✅ Batch completado ({pos}/{len(glasses_probs)} positivos)")
    else:
        glasses_probs: List[float] = []
        for path in tqdm(image_paths, desc="Detectando lentes", unit="imagen"):
            try:
                glasses_probs.append(get_glasses_probability(path, umbral_minimo))
            except Exception as e:
                print(f"[ERROR] {path}: {e}")
                glasses_probs.append(0.0)

    # FASE 3: Estadísticas finales
    print("[INFO] 📈 Estadísticas finales:")
    obtener_estadisticas_cache()
    total = len(glasses_probs)
    con_lentes = sum(1 for p in glasses_probs if p >= 0.5)
    sin_lentes = total - con_lentes
    prom = sum(glasses_probs) / total if total else 0
    print(f"👓 Con lentes: {con_lentes} ({con_lentes/total*100:.1f}%)")
    print(f"👁️ Sin lentes: {sin_lentes} ({sin_lentes/total*100:.1f}%)")
    print(f"📊 Promedio: {prom:.3f}")

    return image_paths, glasses_probs

# ── Interfaz compatible ──────────────────────────────────────────────────────
def process_drive_folder(drive_folder_path: str) -> Tuple[List[str], List[float]]:
    return process_drive_folder_optimized(drive_folder_path, usar_batch=True)

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset_drive_path = (
        "/Mi unidad/INGENIERIA_EN_SOFTWARE/6to_Semestre/"
        "PRACTICAS/Practicas-FOTOS/Primera_Revision/"
        "validator/results/sin_procesar"
    )
    USAR_BATCH = True
    UMBRAL_MIN = 0.0
    MAX_THREADS = 6

    try:
        paths, probs = process_drive_folder_optimized(
            dataset_drive_path,
            usar_batch=USAR_BATCH,
            umbral_minimo=UMBRAL_MIN,
            max_workers=MAX_THREADS
        )
        if not paths:
            sys.exit(1)

        info = {
            "Rutas": format_to_hyperlinks(paths),
            "Probabilidad": probs,
            "Detección": ["SÍ" if p>=0.4486 else "NO" for p in probs]
        }
        normalized = normalize_dict_lengths(info)
        out = dict_to_excel(
            normalized,
            f"results/Reporte_{get_file_count('results')+1}.xlsx"
        )
        print(f"✅ Listo. Excel generado en: {out}")
    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por usuario")
        limpiar_cache_imagenes()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        limpiar_cache_imagenes()
        sys.exit(1)
