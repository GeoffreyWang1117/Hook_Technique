from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-hooks",
    version="0.1.0",
    author="LLM Hooks Team",
    author_email="",
    description="A pluggable LLM hook analysis framework for inference optimization and interpretability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeoffreyWang1117/Hook_Technique",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "cuda": [
            "py3nvml>=0.2.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-hooks=llm_hooks.cli:main",
        ],
    },
)
