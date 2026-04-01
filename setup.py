from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import glob
import os

# Get all .py files inside 'nayanam' (excluding __init__.py)
py_files = [f for f in glob.glob("nayanam/*.py") if not f.endswith("__init__.py")]

# Define Cython extensions dynamically
ext_modules = cythonize([
    Extension(f"nayanam.{os.path.splitext(os.path.basename(file))[0]}", [file]) for file in py_files
])

# Define package data without hardcoding file names
package_data = {
    'nayanam': [
        'nayanam/best.pt',
        'nayanam/best.engine',
        'nayanam/best.onnx',
        'nayanam/best_openvino_model/**',  # Wildcard handled via MANIFEST.in
    ]
}

setup(
    name='nayanam',
    version='3.4.0',
    packages=find_packages(),
    ext_modules=ext_modules,  # Compile Cython modules
    package_data=package_data,
    include_package_data=True,  # Important to include non-Python files
    python_requires='>=3.8',
)
