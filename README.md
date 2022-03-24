## About

A python package for the calculation of ellipticity corrections for seismic phases in elliptical planetary models.

Authors:
- Stuart Russell
- Jessica C. E. Irving
- John F. Rudge
- Sanne Cottaar

## Structure

### README.md

This file.

### requirements.txt

Software requirements of this package.

### LICENSE

License for this package - standard MIT license.

### pyproject.toml and setup.cfg

Files that enable the package to be installed using pip. These should not be edited.

### src/ellipticity/main.py

Main functions for users to interact with.

### src/ellipticity/tools.py

Other functions that are required by main.py but that the user shouldn't need to interact with during normal usage.

### src/ellipticity/epsilon/*txt

Text files containing the epsilon profiles with radius for models that have been initialised for the package. Inlcuded are PREM, iasp91 and ak135 and the package will create these for other models as and when they are required.
