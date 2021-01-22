pystatplottools
=================

A Python library that simplifies working with and plotting of statistical data and high-dimensional distributions. The library utilizes standard numpy operations in a smart way for an easy processing of more complicated data evaluation methods. Further, the pytorch_data_generation module simplifies the generation and the storage of custom datasets. The pystatsplottools library currently consists of the following main modules:

- **distributions** - Convenient computation of joint and marginal distributions. In addition, binned statistics can be evaluated.
- **expectation_values** - Computation of expectation values.
- **plotting** - Wrapper for plotting 2D contour plots with linear and logarithmic scales.
- **pytorch_data_generation** - Tools for an easy generation and storage of a custom pytorch datasets. The data can be pregenerated and stored as a .pt file. Alternatively, data can be generated in real time.
- **visualization** - Contains a class for visualizing samples and batches from the dataset and a decorator for handling figures
- **pdf_env** - Adapted tool for an easy saving of plots as pdfs and pngs. The original code can be found on http://bkanuka.com/posts/native-latex-plots/.

Examples
--------

Examples to the different python modules can be found in the examples/ folder. A more detailed example which covers almost all functionalities of the library can be found here: https://github.com/statphysandml/pystatplottools/blob/master/examples/cheat_sheet.ipynb.

Integration
-----------

So far, the library needs to be build locally. This can be done by

```bash
cd path_to_pystatplottools/

python setup.py sdist
pip install -e .
```

For virtual enviroments, the library needs to be activate beforehand.

After this step, the different modules of the library can be used, for example, by

```python
import pystatplottools

from pystatplottools.distributions.joint_distribution import JointDistribution
```

Dependencies
------------

- matplotlib
- numpy
- pandas
- scipy
- (jupyter lab)

Projects using the pystatplottools library
------------------------------------------

- MCMCEvalutionLib (https://github.com/statphysandml/MCMCEvaluationLib)

Support and development
----------------------

For bug reports/suggestions/complaints please file an issue on GitHub.

Or start a discussion on our mailing list: statphysandml@thphys.uni-heidelberg.de

