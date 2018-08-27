import pandas
import numpy
import sys


# Constants
MAX_EIGEN_ERROR = 0.0001 #sensitivity of eigenvector measurements. Lower value -> higher sensitivity
MAX_EIGEN_ITER = 1000 #number of times the vector is multiplied by the covar matrix before it is given up on and assumed to be impossible
MIN_COVARIANCE_VALS = 1e-10 #if there are no covariances greater than this value, then all relevant eigenvalues have been found


#data should be a DataFrame formatted with each column representing a variable, and each row a separate datapoint
#data should be normalized
def createCoVarMtrx(data):
    testNum, varNum = data.shape
    
    means = pandas.DataFrame(data.mean(0))
    
    temp = [1] * testNum

    variance = data - numpy.dot(temp, means.transpose())
    
    cv = pandas.DataFrame(numpy.dot(variance.transpose(), variance) / (testNum - 1))

    return cv


#takes covar matrix, returns column eigenvector, both dataframes
def findMainEigen(cv):
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
def removeEigenComponents(cv, vec):
    cvCopy = cv
    
    for col in range(cv.shape[0]):
        mask = [[]]
        for i in range(cv.shape[0]):
            if (i == col):
                mask[0].append(1)
            else:
                mask[0].append(0)
        mask = pandas.DataFrame(mask)

        parallelComponent = vec * numpy.dot(vec.transpose(), cv[col])[0]

        cvCopy = cvCopy - pandas.DataFrame(numpy.dot(parallelComponent, mask))

    return cvCopy


def findNewBasis(data):
    covar = createCoVarMtrx(data)

    basis = findMainEigen(covar)
    
    covar = removeEigenComponents(covar, basis)
   
    while (basis.shape[1] < basis.shape[0]) & (covar.max().max() >= MIN_COVARIANCE_VALS):
        vec = findMainEigen(covar)
        covar = removeEigenComponents(covar, vec)
        basis[basis.shape[1]] = vec
   
    while basis.shape[1] < basis.shape[0]:
        basis[basis.shape[1]] = [0] * basis.shape[0]

    return basis


def projectOntoNewBasis(data, basis):
    newData = pandas.DataFrame()
    
    for row in range(data.shape[0]):
        coords = [[]]
        for col in range(data.shape[1]):
            coords[0].append(numpy.dot(data.loc[data.index[row], : ], basis[col]))
        newData = newData.append(coords, True)

    return newData

    
