# device_info.py
import torch

def check_device():
    print("=== Device Check ===")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print("GPU is available!")
        print(f"GPU Device Name: {torch.cuda.get_device_name(current_device)}")
    else:
        print("No GPU available. Running on CPU.")

def main():
    check_device()

if __name__ == "__main__":
    main()
