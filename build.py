import subprocess
import pyinstaller_versionfile
from aidia import __appname__, __version__, ADD_AI

ADD_AI = False

pyinstaller_versionfile.create_versionfile(
    output_file="version.txt",
    version=__version__,
    company_name="Tokushima University",
    file_description="AI development and Image Annotation",
    internal_name=__appname__,
    legal_copyright="Copyright (C) 2021-2024 Kohei Torii",
    original_filename=f"{__appname__}.exe",
    product_name=__appname__,
    translations=[1033, 1041, 1200]  # English, Japanese, Unicode
)

subprocess.run(["pyinstaller", "aidia.spec"], encoding="utf-8")