import pandas
import numpy
import sys


# Constants
MAX_EIGEN_ERROR = 0.0001 #sensitivity of eigenvector measurements. Lower value -> higher sensitivity
MAX_EIGEN_ITER = 1000 #number of times the vector is multiplied by the covar matrix before it is given up on and assumed to be impossible
MIN_COVARIANCE_VALS = 1e-10 #if there are no covariances greater than this value, then all relevant eigenvalues have been found

# The basis class must be created and separately initialized with either setBasisWithCSV(csvFile) or setBasisFromDataframe(panda_data)
class Basis:
    
    def __init__(self):
        self.initialized = False
        
    # INITIALIZING FUNCTIONS
        
    # loads basis directly from saved csv file
    def loadBasis(self, basis_fileName):
        self.basis = pandas.read_csv(basis_fileName, header=None)
    
    # creates and stores a new basis from a csv file
    def setBasisFromCSV(self, csv_fileName):
        data = pandas.read_csv(csv_fileName, header=None)
        self.setBasisFromDataframe(data)
    
    # creates and stores a new basis from a panda dataframe
    def setBasisFromDataframe(self, data):
        covar = self.createCoVarMtrx(data)

        basis = self.findMainEigen(covar)
	    
        covar = self.removeEigenComponents(covar, basis)
	   
        while (basis.shape[1] < basis.shape[0]) & (covar.max().max() >= MIN_COVARIANCE_VALS):
            vec = self.findMainEigen(covar)
            covar = self.removeEigenComponents(covar, vec)
            basis[basis.shape[1]] = vec
	   
        while basis.shape[1] < basis.shape[0]:
            basis[basis.shape[1]] = [0] * basis.shape[0]
            
        self.basis = basis
        self.initialized = True
        
    
    
    # PROJECTING FUNCTIONS
        
    # projects csv into the basis and returns a csv string in the new basis
    def projectCSV(self, csv_fileName, output_csv_fileName):
        data = pandas.read_csv(csv_fileName)
        newData = self.projectData(data)
        return newData.to_csv(output_csv_fileName, header=None, index=False)
        
        
    # projects a dataframe into the basis and returns a dataframe in the new basis
    def projectData(self, data):
        if not self.initialized:
            print("Error! No basis has been initialized, so there is nothing to project data onto.")
            sys.exit(1)
        
            
        newData = pandas.DataFrame()

        for row in range(data.shape[0]):
            coords = [[]]
            for col in range(data.shape[1]):
                coords[0].append(numpy.dot(data.loc[data.index[row], : ], self.basis[col]))
            newData = newData.append(coords, True)

        return newData        
        
        
    # saves the basis in simple csv form
    def saveBasis(self, fileName):
        try:
            with open(fileName, 'w') as f:
                for i in range(len(self.basis)):
                    for j in range(len(self.basis[i])):
                        f.write(str(self.basis[i][j]))
                        if j == len(self.basis[i]) - 1:
                            f.write("\n")
                        else:
                            f.write(",")
                
        except IOError:
            print("Error! Couldn't open file: " + fileName)
            sys.exit(1)
        
        
        
    # HELPER FUNCTIONS
        
    #data should be a DataFrame formatted with each column representing a variable, and each row a separate datapoint
    #data should be normalized
    def createCoVarMtrx(self, data):
        testNum, varNum = data.shape
        
        means = pandas.DataFrame(data.mean(0))
        
        temp = pandas.DataFrame([1] * testNum)
        
        variance = data - numpy.dot(temp, means.transpose())
        
        cv = pandas.DataFrame(numpy.dot(variance.transpose(), variance) / (testNum - 1))
        
        return cv
        
        
    #takes covar matrix, returns column eigenvector, both dataframes
    def findMainEigen(self, cv):
        #create arbitrary beginning vector
        vec = pandas.DataFrame([1] * cv.shape[0])
        vec /= numpy.sqrt(numpy.square(vec).sum(0))[0] #normalize
            
        error = MAX_EIGEN_ERROR + 1 #making sure that the while loop runs at least once
        count = 0
        while((error > MAX_EIGEN_ERROR) & (count < MAX_EIGEN_ITER)):
            oldVec = vec
            vec = pandas.DataFrame(numpy.dot(cv, vec))
            if vec.sum(0)[0] != 0:
                vec /= numpy.sqrt(numpy.square(vec).sum(0))[0]
            error = numpy.sqrt(numpy.square(vec - oldVec).sum(0))[0]
            count += 1

        if count >= MAX_EIGEN_ITER:
            print("Critical Error: Eigenvector didn't stabilize.")
            sys.exit(1)

        return vec


    #takes covar matrix and column vector, and returns new covar matrix, all dataframes
    def removeEigenComponents(self, cv, vec):	    
        for col in range(cv.shape[0]):
            mask = [[]]
            for i in range(cv.shape[0]):
                if (i == col):
                    mask[0].append(1)
                else:
                    mask[0].append(0)
            mask = pandas.DataFrame(mask)
            
            
            parallelComponent = vec * numpy.dot(vec.transpose(), cv[col])[0] 
            
            print(parallelComponent)            
            
            cv = cv - pandas.DataFrame(numpy.dot(parallelComponent, mask))
            
        return cv







# command line support - python3 PCA.py csv_file
# creates basis from given csv file, saves new basis and csv data in terms of new basis in other files: basis.csv and newdata.csv
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect Format: python3 PCA.py csv_file")
        sys.exit(1)
    
    basis = Basis()
    basis.setBasisFromCSV(sys.argv[1])
    basis.saveBasis("basis.csv")
    basis.projectCSV(sys.argv[1], "newdata.csv")