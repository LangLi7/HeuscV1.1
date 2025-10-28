import sys
import os
import time
import subprocess
from datetime import datetime

from ipywidgets import IntProgress, Label, HBox, VBox
from IPython.display import display, clear_output
from tensorflow.keras.callbacks import Callback

# === Farben (ANSI) ===
COLOR_RESET  = "\033[0m"
COLOR_GREEN  = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED    = "\033[91m"
COLOR_BLUE   = "\033[94m"
COLOR_WHITE  = "\033[97m"
COLOR_GRAY   = "\033[90m"
COLOR_CYAN   = "\033[96m"
COLOR_MAGENTA = "\033[95m"

class SystemLoader:
    def __init__(self, task_name: str, total_steps: int = 100, bar_length: int = 30):
        self.task_name = task_name
        self.total_steps = total_steps
        self.bar_length = bar_length
        self.start_time = time.time()
        self.current_step = 0
        self.spinner_states = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

        print(f"\n🚀 {self.task_name} gestartet...\n")

    def update(self, step: int = None, info: str = ""):
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        progress = min(self.current_step / self.total_steps, 1)
        filled = int(self.bar_length * progress)
        bar = "█" * filled + "░" * (self.bar_length - filled)
        spinner = self.spinner_states[self.current_step % len(self.spinner_states)]
        elapsed = time.time() - self.start_time
        eta = (elapsed / progress - elapsed) if progress > 0 else 0

        sys.stdout.write(
            f"\r{spinner} {self.task_name} |{bar}| {progress*100:5.1f}% "
            f"[{self.current_step}/{self.total_steps}] ⏱ {elapsed:5.1f}s ETA:{eta:5.1f}s {info}"
        )
        sys.stdout.flush()

    def done(self, message: str = "Fertig!"):
        total = time.time() - self.start_time
        print(f"\n✅ {self.task_name} {message} ({total:.1f}s)\n")

class TrainingProgressBar:
    def __init__(self, total_epochs, bar_length=30):
        self.total_epochs = total_epochs
        self.bar_length = bar_length
        self.start_time = time.time()

    def update(self, current_epoch, logs=None):
        elapsed = time.time() - self.start_time
        progress = current_epoch / self.total_epochs
        filled = int(self.bar_length * progress)
        bar = "█" * filled + "░" * (self.bar_length - filled)

        # Optional: Show metrics
        acc = logs.get("accuracy", 0) if logs else 0
        loss = logs.get("loss", 0) if logs else 0

        sys.stdout.write(
            f"\r🧠 Training Progress |{bar}| {progress*100:5.1f}% "
            f"[Epoch {current_epoch}/{self.total_epochs}] "
            f"⏱ {elapsed:5.1f}s | acc={acc:.4f} | loss={loss:.4f}"
        )
        sys.stdout.flush()

        if current_epoch == self.total_epochs:
            print("\n✅ Training Completed! ({:.1f}s total)\n".format(elapsed))

class JupyterLoader:
    def __init__(self, title="Loading", total=100):
        self.progress = IntProgress(value=0, min=0, max=total)
        self.label = Label(f"{title}: 0%")
        self.title = title
        self.total = total
        display(VBox([self.label, self.progress]))
        self.start = time.time()

    def update(self, step, info=""):
        self.progress.value = step
        percent = (step / self.total) * 100
        elapsed = time.time() - self.start
        self.label.value = f"{self.title}: {percent:.1f}% ⏱ {elapsed:.1f}s {info}"

    def done(self):
        self.label.value = f"✅ {self.title} complete ({time.time()-self.start:.1f}s)"
        self.progress.bar_style = "success"

class TFTrainingLoader(Callback):
    """
    HEUSC Live-Loader:
    Kombinierte Anzeige von Epoch + Batch mit Live-GPU-Monitoring.
    Beispiel:
    🧩 Epoch 02 | Batch 220/3280 |━━━░░░░░░░░░░░░░░░|  6.7%  213ms/step
       acc=0.4912 loss=0.6987 prec=0.4803 rec=0.8809
       GPU0 RTX3060: 38% | 46°C | 32W | 6123/12288MiB
    """

    def __init__(self, total_epochs, total_batches=None, update_interval=0.25, bar_length=26):
        super().__init__()
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.update_interval = update_interval
        self.bar_length = bar_length
        self.start_time = None
        self.last_update = 0
        self.current_epoch = 1

    # ---------- GPU STATUS ----------
    def _gpu_status(self):
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ]).decode().strip().split("\n")

            infos = []
            for i, line in enumerate(out):
                name, util, temp, power, mu, mt = [x.strip() for x in line.split(",")]
                util, temp, power = int(util), int(temp), float(power)
                util_c = COLOR_GREEN if util < 40 else COLOR_YELLOW if util < 75 else COLOR_RED
                temp_c = COLOR_GREEN if temp < 60 else COLOR_YELLOW if temp < 75 else COLOR_RED
                infos.append(
                    f"{COLOR_BLUE}GPU{i}{COLOR_WHITE} {name}: "
                    f"{util_c}{util}%{COLOR_WHITE} | "
                    f"{temp_c}{temp}°C{COLOR_WHITE} | "
                    f"{COLOR_GRAY}{power:.0f}W{COLOR_WHITE} | "
                    f"{COLOR_CYAN}{mu}/{mt} MiB{COLOR_WHITE}"
                )
            return " | ".join(infos)
        except Exception:
            return f"{COLOR_RED}GPU Info unavailable{COLOR_RESET}"

    # ---------- TRAINING BEGIN ----------
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n{COLOR_BLUE}🚀 Training gestartet (HEUSC Live-Loader aktiviert){COLOR_RESET}\n")

    # ---------- EPOCH BEGIN ----------
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        print(f"\n🧩 Epoch {self.current_epoch:03d}/{self.total_epochs} gestartet...")

    # ---------- BATCH END ----------
    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        if now - self.last_update < self.update_interval:
            return
        self.last_update = now

        acc  = logs.get("accuracy", 0.0)
        loss = logs.get("loss", 0.0)
        prec = logs.get("precision", 0.0)
        rec  = logs.get("recall", 0.0)

        elapsed = now - self.start_time
        elapsed_min, elapsed_sec = divmod(int(elapsed), 60)
        step_ms = int((elapsed / (batch + 1)) * 1000) if batch > 0 else 0
        # Formatierte Gesamtzeit
        elapsed_fmt = f"{elapsed_min:02d}:{elapsed_sec:02d}"

        # Fortschrittsbalken
        progress = (batch + 1) / self.total_batches if self.total_batches else 0
        filled = int(self.bar_length * progress)
        bar = f"{COLOR_GREEN}{'━'*filled}{COLOR_GRAY}{'░'*(self.bar_length-filled)}{COLOR_RESET}"

        gpu_info = self._gpu_status()
        
        # === Dynamische Farbwahl basierend auf Leistung ===
        acc_c  = COLOR_RED if acc < 0.4 else COLOR_YELLOW if acc < 0.6 else COLOR_GREEN
        loss_c = COLOR_GREEN if loss < 0.4 else COLOR_YELLOW if loss < 0.7 else COLOR_RED
        prec_c = COLOR_RED if prec < 0.4 else COLOR_YELLOW if prec < 0.6 else COLOR_CYAN
        rec_c  = COLOR_RED if rec < 0.5 else COLOR_YELLOW if rec < 0.8 else COLOR_MAGENTA
        
        sys.stdout.write(
            f"\r\033[3A\033[2K"
            f"🧩 Epoch {self.current_epoch:02d}/{self.total_epochs} | "
            f"Batch {batch+1:04d}/{self.total_batches or 0:04d} |{bar}| "
            f"{progress*100:5.1f}%  {step_ms:3d}ms/step  ⏱ {COLOR_GRAY}{elapsed_fmt}{COLOR_RESET}\n"
            f"\033[2K   acc={acc_c}{acc:.4f}{COLOR_WHITE} "
            f"loss={loss_c}{loss:.4f}{COLOR_WHITE} "
            f"prec={prec_c}{prec:.4f}{COLOR_WHITE} "
            f"rec={rec_c}{rec:.4f}{COLOR_RESET}\n"
            f"\033[2K{gpu_info}\n"
        )
        sys.stdout.flush()

    # ---------- EPOCH END ----------
    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch + 1  # <── aktuelle Epoche setzen (nicht vorher!)
        val_acc = logs.get("val_accuracy", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        elapsed = time.time() - self.start_time
        elapsed_min, elapsed_sec = divmod(int(elapsed), 60)

        print(
            f"\n✅ {COLOR_BLUE}Epoch {self.current_epoch:02d}/{self.total_epochs}{COLOR_RESET} abgeschlossen "
            f"| {COLOR_GREEN}val_acc={val_acc:.4f}{COLOR_WHITE} "
            f"| {COLOR_YELLOW}val_loss={val_loss:.4f}{COLOR_RESET} "
            f"| ⏱ {COLOR_GRAY}{elapsed_min:02d}:{elapsed_sec:02d}{COLOR_RESET}\n"
        )

    def on_train_end(self, logs=None):
        print(f"\n{COLOR_GREEN}🎯 Training abgeschlossen!{COLOR_RESET}\n")