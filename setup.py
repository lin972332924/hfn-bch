"""
Setup script for the breast cancer classification package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hfn-bch",
    version="1.0.0",
    author="Breast Cancer Classification Team",
    description="Breast Cancer Pathology Image Classification using Local and Global Approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lin972332924/hfn-bch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hfn-train=train:main",
            "hfn-evaluate=evaluate:main",
            "hfn-predict=predict:main",
        ],
    },
)
