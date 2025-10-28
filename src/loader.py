import sys
import time
from datetime import datetime

from ipywidgets import IntProgress, Label, HBox, VBox
from IPython.display import display, clear_output
from tensorflow.keras.callbacks import Callback

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
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.start = None

    def on_train_begin(self, logs=None):
        self.start = time.time()
        print("\n🚀 Training started...\n")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start
        acc = logs.get("accuracy", 0)
        loss = logs.get("loss", 0)
        progress = (epoch + 1) / self.total_epochs
        bar = "█" * int(30 * progress) + "░" * (30 - int(30 * progress))
        print(
            f"\r🧠 Epoch {epoch+1}/{self.total_epochs} |{bar}| "
            f"{progress*100:.1f}% acc={acc:.4f} loss={loss:.4f} ⏱ {elapsed:.1f}s",
            end=""
        )

    def on_train_end(self, logs=None):
        print("\n✅ Training finished.\n")