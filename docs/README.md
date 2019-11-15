# Hotdocs -- Golden Retriever Documentation

## Build

To build the docs, provision the docs environment with `conda`:

```bash
# Optionally, keep your build environment clean
# by working in a docker container
docker run -ti --rm -v $(pwd)/<repo_location>:/usr/src continuumio/anaconda3 bash
cd /usr/src

# In the main repository
conda env create -f docs-environment.yml
conda activate hotdoc-docs
```

Build the docs:

```bash
cd docs
sphinx-apidoc -o source/api ../src # generate reStructuredText from Python source directory
make html
make latexpdf # to generate pdf documentation. You must have laTeX installed.
```

You'll find the built files in `/docs/build`

## Supported Docstring Styles

Sphinx can parse docstrings using different style guides.

The following style guides are supported by the current config:

- [PEP 287 — reStructuredText Docstring Format](https://www.python.org/dev/peps/pep-0287/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Style Guide](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)

## Hide unused functions and classes

Sphinx generates documentation for all classes and functions,
unless they are set as private with a `_` prefix. E.g.,

```python
class _ThisWontBePublished:
    def __init__(self):
        print('code here')
```

Optional:

Sphinx by default ignores all private classes and functions.
But can set Sphinx to document private classes that have docstrings:

```python
def _included(self):
    """
    This will be included in the docs because it has a docstring
    """
    pass

def _skipped(self):
    # This will NOT be included in the docs
    pass
```

To enable this behaviour, edit `conf.py` and set this line to `True`:

```python
napoleon_include_private_with_doc = True
```

See https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#confval-napoleon_include_private_with_doc
for more information.

## Other notes

The API documentation is generated from the `/src` folder
in this repository. This means you'll need to activate
the `conda` environment needed to run the libraries
in order for the docs to properly read the modules in
`/src`.

