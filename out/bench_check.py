import importlib
for name in ["torch", "tensorflow", "numpy", "scipy"]:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"{name:12} v={v}")
    except Exception:
        print(f"{name:12} MISSING")
