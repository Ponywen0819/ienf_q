# PyInstaller spec: freeze tools/electron_worker.py into a self-contained
# onedir bundle so the Electron app can run the pipeline with no Python install.
#
# The worker only uses the annotation_grow slice of neural_reconstruction, which
# pulls cv2/numpy/scipy/skimage/skan/numba/pandas/PIL/networkx — NOT the torch /
# transformers / xgboost / gudhi stack the full project declares. We collect the
# lazy-loading scientific libs explicitly and exclude the heavy unused ones so
# the bundle stays small and the freeze is reliable.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# Libs with lazy_loader / dynamic submodule imports PyInstaller can't see statically.
for pkg in ("skimage", "skan", "numba", "llvmlite"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Never bundle the heavy stack the worker doesn't touch (guards against a stray
# guarded `import torch` dragging in gigabytes).
excludes = [
    "torch", "torchvision", "torchmetrics", "transformers", "timm",
    "xgboost", "gudhi", "SimpleITK", "simpleitk", "matplotlib", "seaborn",
    "sklearn", "scikit_learn", "tensorflow", "IPython", "notebook", "pytest",
]

a = Analysis(
    ["tools/electron_worker.py"],
    pathex=["src"],  # so `neural_reconstruction` is importable
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="electron_worker",
    console=True,  # worker talks JSON-RPC over stdio; no window
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="electron_worker",
)
