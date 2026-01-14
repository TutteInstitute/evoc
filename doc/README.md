# EVoC Documentation

This directory contains the Sphinx documentation for EVoC.

## Structure

```
doc/
├── build/                  # Generated documentation (HTML, PDF, etc.)
├── source/                 # Source files for documentation  
│   ├── _static/           # Static files (CSS, images, etc.)
│   ├── _templates/        # Custom Sphinx templates
│   ├── api/               # API documentation files
│   ├── notebooks/         # Jupyter notebook examples
│   ├── tutorials/         # Step-by-step tutorials
│   ├── conf.py           # Sphinx configuration
│   ├── index.rst         # Main documentation page
│   └── *.rst             # Other documentation pages
├── requirements.txt       # Documentation dependencies
├── Makefile              # Build commands (Unix)
├── build_docs.sh         # Automated build script (Unix)
├── build_docs.bat        # Automated build script (Windows)  
└── README.md             # This file
```

## Building the Documentation

### Prerequisites

1. Python 3.8 or later
2. Git (for development installation)

### Quick Build

**Unix/macOS:**
```bash
cd doc
./build_docs.sh
```

**Windows:**
```cmd
cd doc
build_docs.bat
```

### Manual Build

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install EVoC in development mode:
```bash
pip install -e ../..
```

3. Build documentation:
```bash
make html
```

4. Open `build/html/index.html` in your browser

### Advanced Options

**Clean build:**
```bash
make clean html
```

**Check links:**
```bash
make linkcheck
```

**Run doctests:**
```bash
make doctest
```

**Live reload during development:**
```bash
pip install sphinx-autobuild
make livehtml
```

## Features

- **Sphinx RTD Theme**: Professional appearance matching ReadTheDocs
- **Numpydoc**: Automatic parsing of NumPy-style docstrings
- **Nbsphinx**: Integration of Jupyter notebooks as documentation
- **Autodoc**: Automatic API documentation generation
- **ReadTheDocs Ready**: Configured for automatic deployment

## Adding Content

### New Documentation Pages

1. Create `.rst` files in `source/`
2. Add them to the `toctree` in `index.rst`
3. Rebuild documentation

### Jupyter Notebooks

1. Add `.ipynb` files to `source/notebooks/`
2. Add them to `source/notebooks/index.rst`
3. Notebooks are automatically converted during build

### API Documentation

API documentation is automatically generated from docstrings. To add new modules:

1. Add the module to `source/api/index.rst`
2. Create a dedicated `.rst` file if needed
3. Rebuild documentation

## ReadTheDocs Integration

This documentation is configured for ReadTheDocs deployment:

- Configuration: `.readthedocs.yaml` in project root
- Requirements: `doc/requirements.txt`
- Python version: 3.11 (configurable in `.readthedocs.yaml`)

## Troubleshooting

**Import errors during build:**
- Ensure EVoC is installed in development mode: `pip install -e ../..`
- Check that all dependencies are installed: `pip install -r requirements.txt`

**Missing modules in API docs:**
- Verify the module paths in `source/api/index.rst`
- Check that modules are importable from the documentation directory

**Notebook execution errors:**
- Notebooks are not executed by default (`nbsphinx_execute = 'never'`)
- To execute notebooks during build, change to `nbsphinx_execute = 'always'` in `conf.py`

**Theme or styling issues:**
- Check `source/_static/custom.css` for customizations
- Verify `sphinx_rtd_theme` is installed

## Contributing

When adding new documentation:

1. Follow reStructuredText formatting
2. Use NumPy-style docstrings for API documentation  
3. Include code examples where appropriate
4. Test build locally before submitting
5. Keep notebook outputs clear for examples

For more details, see the main EVoC contributing guidelines.
