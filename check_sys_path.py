import sys
import os

print("Current working directory:", os.getcwd())
print("Python sys.path:")
for p in sys.path:
    print(p)
