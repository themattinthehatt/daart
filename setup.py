from distutils.core import setup


VERSION = '1.1.0'

# add the README.md file to the long_description
with open('README.md', 'r') as fh:
    long_description = fh.read()

install_requires = [
    'ipykernel',
    'jupyter',
    'matplotlib',
    'numpy',
    'opencv-python-headless',
    'pandas',
    'pytest',
    'pyyaml',
    'scikit-learn',
    'scipy>=1.2.0',
    'seaborn',
    'tables',
    'test-tube',
    'torch',
    'tqdm',
    'typeguard',
]

extras_require = {
    'dev': {
        'flake8',
        'sphinx',
        'sphinx_rtd_theme',
        'sphinx-rtd-dark-mode',
        'sphinx-automodapi',
        'sphinx-copybutton',
    }
}

setup(
    name='daart',
    packages=['daart'],
    version=VERSION,
    description='a collection of action segmentation tools for analyzing behavioral data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='matt whiteway',
    author_email='',
    url='http://www.github.com/themattinthehatt/daart',
    install_requires=install_requires,
    extras_require=extras_require,
    keywords=['machine learning', 'action segmentation', 'computer vision'],
)
