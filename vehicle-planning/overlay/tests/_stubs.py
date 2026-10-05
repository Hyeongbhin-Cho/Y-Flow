import sys
import types

for name in ("mlflow", "mmengine"):
    try:
        __import__(name)
    except ImportError:
        m = types.ModuleType(name)
        if name == "mmengine":
            m.fileio = types.SimpleNamespace(get_text=lambda p: open(p).read())
        sys.modules[name] = m
