# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules
from pathlib import Path

block_cipher = None
project_root = Path(SPECPATH).parent

datas = []
binaries = []
hiddenimports = []
excluded_packages = [
    "IPython",
    "altair",
    "cv2",
    "datasets",
    "googleapiclient",
    "jupyter",
    "keras",
    "matplotlib",
    "notebook",
    "pandas",
    "playwright",
    "pyarrow",
    "scipy",
    "selenium",
    "sklearn",
    "streamlit",
    "tensorflow",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
]


def is_excluded_entry(entry):
    source, target = str(entry[0]).lower(), str(entry[1]).lower()
    return any(
        f"\\{package_name.lower()}" in source
        or f"/{package_name.lower()}" in source
        or target == package_name.lower()
        or target.startswith(f"{package_name.lower()}\\")
        or target.startswith(f"{package_name.lower()}/")
        for package_name in excluded_packages
    )

for app_package_name in [
    "app.api",
    "app.core",
    "app.db",
    "app.embeddings",
    "app.llm",
    "app.retrieval",
    "app.services",
    "app.utils",
]:
    hiddenimports += collect_submodules(app_package_name)

for package_name in [
    "llama_cpp",
    "fastembed",
    "qdrant_client",
]:
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
    except Exception:
        pass

datas = [entry for entry in datas if not is_excluded_entry(entry)]
binaries = [entry for entry in binaries if not is_excluded_entry(entry)]
hiddenimports = [
    name
    for name in hiddenimports
    if not any(
        name == package_name or name.startswith(f"{package_name}.")
        for package_name in excluded_packages
    )
]

hiddenimports += [
    "fastapi",
    "uvicorn",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "langchain_core.callbacks",
    "langchain_core.documents",
    "langchain_core.embeddings",
    "langchain_core.language_models",
    "langchain_core.prompts",
    "langchain_community.llms",
    "langchain_community.llms.llamacpp",
    "langchain_google_genai",
    "google.genai",
    "google.genai.types",
    "nltk",
    "nltk.tokenize",
    "rank_bm25",
]

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_packages,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

a.binaries = TOC([entry for entry in a.binaries if not is_excluded_entry(entry)])
a.datas = TOC([entry for entry in a.datas if not is_excluded_entry(entry)])
a.pure = TOC([
    entry
    for entry in a.pure
    if not any(
        str(entry[0]) == package_name or str(entry[0]).startswith(f"{package_name}.")
        for package_name in excluded_packages
    )
])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="oracles-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="oracles-backend",
)
