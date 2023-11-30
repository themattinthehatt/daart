from distutils.core import setup


VERSION = '1.0.1'

# add the README.md file to the long_description
with open('README.md', 'r') as fh:
    long_description = fh.read()

extras_require = {
    'dev': {
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
    extras_require=extras_require,
    keywords=['machine learning', 'action segmentation', 'computer_vision'],
)
