import pathlib
from setuptools import setup, find_packages


base_packages = ["scikit-learn>=1.0.0", "scikit-partial>=0.1.0", "mmh3"]

test_packages = [
    "interrogate>=1.5.0",
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "black>=19.3b0",
    "pre-commit>=2.2.0",
    "flake8-print>=4.0.0",
]

all_packages = base_packages
dev_packages = all_packages + test_packages


setup(
    name="scikit-bloom",
    version="0.1.1",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Bloom tricks for text pipelines in scikit-learn.",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://koaning.github.io/scikit-bloom/",
    project_urls={
        "Documentation": "https://koaning.github.io/scikit-bloom/",
        "Source Code": "https://github.com/koaning/scikit-bloom/",
        "Issue Tracker": "https://github.com/koaning/scikit-bloom/issues",
    },
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
    license_files=("LICENSE",),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
