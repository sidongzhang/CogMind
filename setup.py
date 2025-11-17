from setuptools import setup, find_packages

setup(
    name="cogmind",
    version="0.1.0",
    description="A lightweight LLM training framework from scratch",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "tqdm",
        "pytest"
    ],
    python_requires='>=3.8',
    author="sidongzhang",
    author_email="tiancaizhaozhao618@example.com"
)