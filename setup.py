from setuptools import setup

setup(
    name="icae",
    version="0.1",
    description="Max Pernklau's master thesis",
    author="Max Pernklau",
    author_email="max.pernklau@udo.edu",
    packages=["icae"],
    zip_safe=False,
    test_suite="pytest",
    install_requires=[
        "ordered_set",
        "matplotlib",
        "numpy",
        "pandas",
        "sklearn",
        "tqdm",
        "scipy",
        "IPython",
        "torch",
        "torchvision",
        "ray",
        "joblib",
        "ConfigSpace",
        "hilbertcurve",
        "python-box",
        "jupytext",
        "jupyterlab",
        "tables",
        "requests",
        "pytest",
        "black",
        "hpbandster"
    ],
)