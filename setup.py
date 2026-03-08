from setuptools import setup, find_packages

setup(
    name="cbct-segmentation",
    version="1.0.0",
    description="3D CBCT Tooth Segmentation with FDI labeling",
    author="Ashmeet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "nibabel>=5.1.0",
        "SimpleITK>=2.3.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "scikit-image>=0.22.0",
    ],
    entry_points={
        "console_scripts": [
            "cbct-train=training.train:main",
            "cbct-predict=inference.predict:main",
            "cbct-preprocess=preprocessing.preprocess:main",
        ]
    },
)
