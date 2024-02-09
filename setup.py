
from setuptools import find_packages
from setuptools import setup
from aidia import __version__


def main():

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