import os
import ctypes
import sys

torch_lib_path = r"D:\PythonProjects\signatureverifier\.venv\Lib\site-packages\torch\lib"

print(f"Checking for torch lib path: {torch_lib_path}")
if not os.path.exists(torch_lib_path):
    print("ERROR: torch lib path does not exist!")
    sys.exit(1)

# Add DLL directory for Python 3.8+
if hasattr(os, 'add_dll_directory'):
    print(f"Adding {torch_lib_path} to DLL search path...")
    os.add_dll_directory(torch_lib_path)
    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ['PATH']

dlls_to_check = [
    "libiomp5md.dll",
    "asmjit.dll",
    "uv.dll",
    "fbgemm.dll",
    "torch_cpu.dll",
    "torch.dll"
]

print("--- System info ---")
print(f"Python version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"PATH: {os.environ.get('PATH')[:100]}...")

for dll in dlls_to_check:
    dll_path = os.path.join(torch_lib_path, dll)
    if not os.path.exists(dll_path):
        print(f"MISSING: {dll} does not exist at {dll_path}")
        continue
    
    print(f"Attempting to load {dll} from {dll_path}...")
    try:
        # Use LoadLibraryEx flags to search for dependencies in the same directory
        # LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008
        handle = ctypes.WinDLL(dll_path, use_last_error=True)
        print(f"SUCCESS: Loaded {dll}")
    except Exception as e:
        print(f"FAILED: Could not load {dll}. Error: {e}")
        # Try to load without absolute path to see if Windows finds it in the added DLL directories
        try:
            ctypes.CDLL(dll)
            print(f"SUCCESS (relative): Loaded {dll}")
        except:
            pass

try:
    import torch
    print("SUCCESS: import torch worked!")
except Exception as e:
    print(f"FAILED: import torch still fails: {e}")
