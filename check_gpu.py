import torch
import sys

def check_gpu():
    print("Python execution started")
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("\nSUCCESS: CUDA is available!")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test tensor creation on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print("\nTensor test on GPU passed:")
            print(x)
        except Exception as e:
            print(f"\nERROR: Tensor creation failed: {e}")
            sys.exit(1)
            
        # Check if llama-cpp-python is compiled with BLAS
        try:
            from llama_cpp import Llama
            print("\nllama-cpp-python imported successfully.")
            # We can't easily check for CUBLAS explicitly without verbose logs during load,
            # but successful import is a good start.
        except ImportError:
            print("\nWARNING: llama-cpp-python not found.")
            
    else:
        print("\nFAILURE: CUDA is NOT available.")
        print("This container is running on CPU.")
        sys.exit(1)

if __name__ == "__main__":
    check_gpu()
