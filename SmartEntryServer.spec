
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all


datas = [
    ('Web-UI/app.py', 'Web-UI'),
    ('Web-UI/templates', 'Web-UI/templates'),
    ('Web-UI/static/app.js', 'Web-UI/static'),
    ('Web-UI/static/style.css', 'Web-UI/static'),
    ('Web-UI/static/default-avatar.svg', 'Web-UI/static'),

    ('Face-Recognition', 'Face-Recognition'),

    (
        'smile-env/Lib/site-packages/insightface/data/objects',
        'objects'
    ),
]

binaries = []

hiddenimports = [
    'mediapipe',
    'peewee',
    'pgvector',
    'fastapi',
    'starlette',
    'uvicorn',
    'jinja2',
    'cv2',
    'insightface',
    'onnxruntime',
]

packages = [
    'mediapipe',
    'insightface',
    'onnxruntime',
    'peewee',
    'pgvector',
    'fastapi',
    'starlette',
    'uvicorn',
    'jinja2',
    'cv2',
]

for package in packages:
    tmp_ret = collect_all(package)

    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]


a = Analysis(
    ['server_main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SmartEntryServer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
