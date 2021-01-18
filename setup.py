from setuptools import setup, find_packages
import os

__version__ = '1.0.4'
url = 'https://github.com/bzhanglab/proms'

install_requires = [
    'numpy',
    'scipy',
    'sklearn',
    'pandas',
    'xgboost',
    'matplotlib',
    
]


setup(
    name='proms',
    version=__version__,
    description='Protein Markers Selection',
    author='Zhiao Shi',
    author_email='zhiao.shi@gmail.com',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'machine learning', 'feature selection', 'proteomics',
        'multiomics', 'biomarker', 'genomics', 'bioinformatics'
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=find_packages(),
    entry_points={"console_scripts": ["proms_train=proms.__main__:main", 
                  "proms_predict=proms.predict:main"]}
)
