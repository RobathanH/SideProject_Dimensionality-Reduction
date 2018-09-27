# Dimensionality-Reduction
Experimenting with dimensionality reduction through Principal Component Analysis for semi-correlated data spaces.

The PCA.py script implements a class called Basis, which can analyze a dataset (either from CSV file or from a pandas dataframe) and store a new orthogonal basis in terms of the original basis.
This new basis will minimize correlation between axes, possibly even with a lower dimensionality in the basis-space. Currently, if the data exhibits less dimensions (d) of variability than the number of variables (n), then the basis will end with n-d 0 vectors.

The Basis class must be initialized separately from its creation, using one of two initializer functions: setBasisFromCSV(csv_fileName) or setBasisFromDataframe(data). These functions will override any previously saved basis.
There are two ways to project data onto this new basis: projectCSV(csv_fileName, output_csv_fileName) and projectData(data). projectCSV() reads a CSV file and writes to another CSV file. projectData() reads from a pandas dataframe object and returns another dataframe.

The Basis class includes a method saveBasis(fileName) which will, as the name implies, save the basis to a given file in raw csv format, with no column or row labels. The inverse of this function is loadBasis(fileName) which will load a basis stored in this format.

Command-line Interface: python3 PCA.py csv_file
In this format, this program will find the optimized basis for the given csv_file and save the new basis in the file basis.csv and save the given data in terms of the new basis in the file newdata.csv.
