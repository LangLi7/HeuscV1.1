"""Check TensorFlow GPU availability and configuration."""
import tensorflow as tf
import os

def gpu_info():
    print("🔍 Checking TensorFlow GPU configuration...\n")

    # List all physical devices
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠️  No GPU detected by TensorFlow — running on CPU.")
        print("   → Check if your Conda env has the GPU build installed:")
        print("     conda install -c conda-forge tensorflow tensorflow-gpu cudatoolkit cudnn\n")
        return

    print(f"✅ {len(gpus)} GPU(s) detected:\n")
    for gpu in gpus:
        print(f"   • {gpu.name}")

    print("\n🔧 TensorFlow build info:")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   CUDA support:       {tf.test.is_built_with_cuda()}")
    print(f"   cuDNN support:      {tf.test.is_built_with_gpu_support()}")
    print(f"   OneDNN (CPU opt):   {os.environ.get('TF_ENABLE_ONEDNN_OPTS', 'ON')}")

    # Optional: test GPU performance
    print("\n⚙️  Running simple GPU compute test...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("   ✅ GPU computation succeeded.")

if __name__ == "__main__":
    gpu_info()