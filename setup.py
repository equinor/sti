from setuptools import setup, find_packages

setup(
        name="sti",
        packages=find_packages(),
        install_requires=[
            'sklearn',
            'numpy',
            'scipy',
        ],
        extras_require={
            'train' : ['pandas'],
        },
        test_require=['pytest'],
)
