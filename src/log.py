import datetime
import os
import json
import traceback

def ensure_log_dir(logs_root: str = "logs") -> str:
    """Erstellt den Log-Ordner, falls nicht vorhanden, und gibt ihn zurück."""
    os.makedirs(logs_root, exist_ok=True)
    return logs_root


def log_error_once(message: str, logs_root: str = "logs", category: str = "errors"):
    """Loggt einen Fehler einmal pro Tag in logs/<category>_<YYYY-MM-DD>.log."""
    logs_root = ensure_log_dir(logs_root)
    today = datetime.date.today().isoformat()
    log_file = os.path.join(logs_root, f"{category}_{today}.log")

    # Wenn bereits enthalten → überspringen
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            if any(message in line for line in f.readlines()):
                return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def log_info(message: str, logs_root: str = "logs", category: str = "info"):
    """Einfaches Info-Log."""
    logs_root = ensure_log_dir(logs_root)
    today = datetime.date.today().isoformat()
    log_file = os.path.join(logs_root, f"{category}_{today}.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def log_exception(exc: Exception, logs_root: str = "logs"):
    """Schreibt vollständige Exception inkl. Traceback."""
    logs_root = ensure_log_dir(logs_root)
    today = datetime.date.today().isoformat()
    log_file = os.path.join(logs_root, f"errors_{today}.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trace = traceback.format_exc()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Exception: {exc}\n{trace}\n")
