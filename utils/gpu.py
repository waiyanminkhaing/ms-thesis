import platform
import tensorflow as tf
import torch

def get_device():
    os_name = platform.system()

    if os_name == "Darwin":
        # for mac
        devices = tf.config.list_physical_devices()
        print("\nDevices: ", devices)

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                details = tf.config.experimental.get_device_details(gpu)
                print("GPU details: ", details)
        else:
            print("No GPU found. Using CPU.")

        # set GPU device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        return device
    
    # for window and linux
    print("Tensorflow GPUs: ", tf.config.list_physical_devices('GPU'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using PyTorch device:", device)
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    return device