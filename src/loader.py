import sys
import time
import subprocess
from datetime import datetime

from ipywidgets import IntProgress, Label, HBox, VBox
from IPython.display import display, clear_output
from tensorflow.keras.callbacks import Callback

# ANSI-Farbcodes für Terminalfarben
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_BLUE = "\033[94m"
COLOR_WHITE = "\033[97m"
COLOR_GRAY = "\033[90m"

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
    Farbige Live-Trainingsanzeige mit GPU-Status bei jeder Batch-Aktualisierung.
    Kompatibel mit der grünen Keras-Fortschrittsleiste.
    """
    def __init__(self, total_epochs, update_interval_batches=1):
        super().__init__()
        self.total_epochs = total_epochs
        self.update_interval_batches = update_interval_batches
        self.start = None

    def _gpu_status(self):
        """Fragt nvidia-smi live ab (Name, Auslastung, Temperatur, Leistung, Speicher) und färbt Werte."""
        try:
            result = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ]).decode().strip().split("\n")

            infos = []
            for i, line in enumerate(result):
                name, util, temp, power, mem_used, mem_total = [x.strip() for x in line.split(",")]
                util = int(util)
                temp = int(temp)
                power = float(power)

                # Farben dynamisch bestimmen
                if util < 40:
                    util_color = COLOR_GREEN
                elif util < 75:
                    util_color = COLOR_YELLOW
                else:
                    util_color = COLOR_RED

                if temp < 60:
                    temp_color = COLOR_GREEN
                elif temp < 75:
                    temp_color = COLOR_YELLOW
                else:
                    temp_color = COLOR_RED

                infos.append(
                    f"{COLOR_BLUE}GPU{i}{COLOR_WHITE} {name}: "
                    f"{util_color}{util}%{COLOR_WHITE} | "
                    f"{temp_color}{temp}°C{COLOR_WHITE} | "
                    f"{COLOR_GRAY}{power:.0f}W{COLOR_WHITE} | "
                    f"{COLOR_GREEN}{mem_used}/{mem_total} MiB{COLOR_WHITE}"
                )
            return " | ".join(infos)
        except Exception:
            return f"{COLOR_RED}GPU Info unavailable{COLOR_RESET}"

    def on_train_begin(self, logs=None):
        self.start = time.time()
        print(f"\n{COLOR_BLUE}🚀 Training gestartet...{COLOR_RESET}\n")

    def on_train_batch_end(self, batch, logs=None):
        """Wird nach jedem Batch aufgerufen – zeigt GPU live."""
        if batch % self.update_interval_batches == 0:
            acc = logs.get("accuracy", 0.0)
            loss = logs.get("loss", 0.0)
            elapsed = time.time() - self.start
            gpu_info = self._gpu_status()

            sys.stdout.write(
                f"\r{gpu_info} | "
                f"{COLOR_GREEN}acc={acc:.4f}{COLOR_WHITE} | "
                f"{COLOR_YELLOW}loss={loss:.4f}{COLOR_WHITE} | "
                f"⏱ {COLOR_GRAY}{elapsed:6.1f}s{COLOR_RESET}     "
            )
            sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("accuracy", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        elapsed = time.time() - self.start

        print(
            f"\n🧠 {COLOR_BLUE}Epoch {epoch+1}/{self.total_epochs}{COLOR_RESET} | "
            f"{COLOR_GREEN}acc={acc:.4f}{COLOR_WHITE} val_acc={COLOR_GREEN}{val_acc:.4f}{COLOR_WHITE} | "
            f"{COLOR_YELLOW}loss={loss:.4f}{COLOR_WHITE} val_loss={COLOR_YELLOW}{val_loss:.4f}{COLOR_WHITE} "
            f"| ⏱ {COLOR_GRAY}{elapsed:6.1f}s{COLOR_RESET}"
        )

    def on_train_end(self, logs=None):
        print(f"\n{COLOR_GREEN}✅ Training abgeschlossen!{COLOR_RESET}\n")
