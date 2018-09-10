# Dimensionality-Reduction
Experimenting with dimensionality reduction through Principal Component Analysis for semi-correlated data spaces.

The PCA.py script implements a class called Basis, which can analyze a dataset (either from CSV file or from a pandas dataframe) and store a new orthogonal basis in terms of the original basis.
This new basis will minimize correlation between axes, possibly even with a lower dimensionality in the basis-space.

The Basis class must be initialized separately from its creation, using one of two initializer functions: setBasisFromCSV(csv_fileName) or setBasisFromDataframe(data). These functions will override any previously saved basis.
There are two ways to project data onto this new basis: projectCSV(csv_fileName, output_csv_fileName) and projectData(data). projectCSV() reads a CSV file and writes to another CSV file. projectData() reads from a pandas dataframe object and returns another dataframe.

TODO: Command-line support, performance optimizations
