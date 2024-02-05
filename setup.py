import distutils.spawn
import os
import re
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup
from aidia import __version__


# def get_version():
#     filename = "aidia/__init__.py"
#     with open(filename) as f:
#         match = re.search(
#             r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
#         )
#     if not match:
#         raise RuntimeError("{} doesn't contain __version__".format(filename))
#     version = match.groups()[0]
#     return version


# def get_install_requires():
#     install_requires = [
#         "opencv-python-headless==4.4.0.46",
#         "PySide2==5.15.2",
#         "PyQt5==5.15.2",
#         "QtPy",
#         "pydicom",
#         "python-gdcm",
#         "pyinstaller==4.10",
#         "pyinstaller-versionfile",
#         "PyYAML",
#         "matplotlib",
#         "numpy",
#         "pandas",
#         "openpyxl",
#         "tensorflow==2.10.0",
#         "onnxruntime==1.14.1",
#         "tf2onnx==1.14.0",
#         "imgaug==0.4.0 --no-binary imgaug",
#     ]

    # Find python binding for qt with priority:
    # PyQt5 -> PySide2
    # and PyQt5 is automatically installed on Python3.
    # QT_BINDING = None

    # try:
    #     import PyQt5  # NOQA

    #     QT_BINDING = "pyqt5"
    # except ImportError:
    #     pass

    # if QT_BINDING is None:
    #     try:
    #         import PySide2  # NOQA

    #         QT_BINDING = "pyside2"
    #     except ImportError:
    #         pass

    # if QT_BINDING is None:
    #     # PyQt5 can be installed via pip for Python3
    #     # 5.15.3, 5.15.4 won't work with PyInstaller
    #     install_requires.append("PyQt5!=5.15.3,!=5.15.4")
    #     QT_BINDING = "pyqt5"

    # del QT_BINDING

    # if os.name == "nt":  # Windows
    #     install_requires.append("colorama")

    # return install_requires


# def get_long_description():
#     with open("README.md") as f:
#         long_description = f.read()
#     try:
#         # when this package is being released
#         import github2pypi

#         return github2pypi.replace_url(
#             slug="wkentaro/labelme", content=long_description, branch="main"
#         )
#     except ImportError:
#         # when this package is being installed
#         return long_description


def main():
    # version = get_version()

    # if sys.argv[1] == "release":
    #     try:
    #         import github2pypi  # NOQA
    #     except ImportError:
    #         print(
    #             "Please install github2pypi\n\n\tpip install github2pypi\n",
    #             file=sys.stderr,
    #         )
    #         sys.exit(1)

    #     if not distutils.spawn.find_executable("twine"):
    #         print(
    #             "Please install twine:\n\n\tpip install twine\n",
    #             file=sys.stderr,
    #         )
    #         sys.exit(1)

    #     commands = [
    #         "git push origin main",
    #         "git tag v{:s}".format(version),
    #         "git push origin --tags",
    #         "python setup.py sdist",
    #         "twine upload dist/labelme-{:s}.tar.gz".format(version),
    #     ]
    #     for cmd in commands:
    #         print("+ {:s}".format(cmd))
    #         subprocess.check_call(shlex.split(cmd))
    #     sys.exit(0)

    setup(
        name="aidia",
        version=__version__,
        # version=version,
        packages=find_packages(),
        description="AI Development and Image Annotation",
        # long_description=get_long_description(),
        # long_description_content_type="text/markdown",
        author="Kohei Torii",
        author_email="work.torii@gmail.com",
        url="https://github.com/wt501/Aidia",
        install_requires=open("requirements.txt").read().splitlines(),
        # install_requires=get_install_requires(),
        license="GPLv3",
        keywords="Image Annotation, Machine Learning, Medical Images",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            # "Programming Language :: Python :: 3.5",
            # "Programming Language :: Python :: 3.6",
            # "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3 :: Only",
        ],
        package_data={"aidia": [
            "icons/*",
            "config/*.yaml",
            "translate/ja_JP.qm"]},
        entry_points={
            "console_scripts": [
                "aidia=aidia.__main__:main",
            ],
        },
    )


if __name__ == "__main__":
    main()