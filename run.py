import importlib
from configparser import ConfigParser

conf = ConfigParser()
conf.read('/mnt/c/Documents and Settings/MANJU/Downloads/nayanam/nayanam/config.ini')
logic_id = conf.get("method", "logic")

if logic_id == "Y":
    module_name = "nayanam.tracker"
else:
    module_name = "nayanam.segmentation"

try:
    selected_module = importlib.import_module(module_name)
    
    if hasattr(selected_module, 'main'):
        selected_module.main()
    else:
        print(f"Error: Module '{module_name}' does not have a 'main()' function.")
except ImportError as e:
    print(f"Failed to import {module_name}: {e}")
