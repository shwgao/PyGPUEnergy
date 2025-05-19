from setuptools import setup, find_packages

setup(
    name="PyGPUEnergy",
    version="0.1.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "matplotlib",
        "torch", 
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    python_requires=">=3.9",
    author="Shouwei Gao",
    author_email="gaosho@oregonstate.edu",
    description="A package for monitoring GPU power and energy consumption",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shwgao/PyGPUEnergy",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
) 