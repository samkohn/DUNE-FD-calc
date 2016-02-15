import numpy as np

class SimulationComponent(np.matrix):
    def __new__(cls, location): # TODO **kwargs
        """
        Read in data from location and assign it to the np.matrix
        (inherited) data structure.

        This base class method reads in the data, and each subclass must
        define how to convert that data into an np.matrix.

        """
        #TODO read in data
        data = location
        data = cls._getMatrixForm(data)
        obj = np.matrix.__new__(cls, data)
        obj.description = None
        obj.bins = None # Instance of Binning object
        obj.dataFileLocation = None
        obj.views = [] # any views into the data, e.g. 'nue flux'
        # Store the data in the underlying np.matrix structure
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.description = getattr(obj, 'description', None)
        self.bins = getattr(obj, 'bins', None)
        self.dataFileLocation = getattr(obj, 'dataFileLocation', None)
        self.views = getattr(obj, 'views', None)

    @staticmethod
    def _getMatrixForm(data):
        """
        Format the data in the appropriate matrix representation.

        E.g. matrix, diagonal matrix, or column vector.

        """
        raise NotImplementedError("Must override this method in " +
        " a subclass.")

    # Various views on the data: for example
    # ['nue flux', 'numu flux', ...]
    # or ['nueCC spectrum','nueNC spectrum', ...]
    # This may require more variables to keep track of the state of the
    # object.
    def extract(self, viewName):
        """
        Extract a particular part of the data, for example the nue
        spectrum.

        If no special views exist, simply returns self in all cases.

        """
        return self

class Binning(object):
    def __init__(self, edges):
        self.edges = np.array(edges)
        self.centers = np.empty(self.edges.shape[0]-1)
        self.start = self.edges[0]
        self.end = self.edges[-1]
        for ind, (i, j) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            self.centers[ind] = ((i + j)/2.0)


class BeamFlux(SimulationComponent):
    def __new__(cls, location):
        obj = SimulationComponent.__new__(cls, location)
        obj.views.extend(['nue flux', 'numu flux', 'nutau flux'])
        obj.bins = Binning(np.arange(0, 10.25, 0.25))
        return obj

    def __array_finalize__(self, obj):
        pass

    @staticmethod
    def _getMatrixForm(data):
        """
        Ensure that the data is a column vector and return it.

        No guarantee on whether the return value is a copy or a
        reference to the original data.

        """
        data = np.asarray(data)
        if data.ndim == 1:
            return data
        else:
            raise ValueError("Data not a column vector")

    def zipWithEnergy(self):
        return zip(self.bins.centers, self)

    def extract(self, viewName):
        nbins = len(self.bins.centers)
        start = 0
        end = 0
        if viewName == 'nue flux':
            start = 0
            end = nbins
        elif viewName == 'numu flux':
            start = nbins
            end = 2 * nbins
        elif viewName == 'nutau flux':
            start = 2 * nbins
            end = 3 * nbins
        else:
            raise ValueError("Bad viewName")
        return self[start:end]

class OscillationProbability(SimulationComponent):
    def _getMatrixForm(self, data):
        """
        Ensure that the data is either a column vector or a diagonal
        matrix and return it.

        No guarantee on whether the return value is distinct from the
        original data.

        """
        data = np.asarray(data)
        if data.ndim == 1:
            return np.diag(data)
        elif data.ndim == 2:
            data_diagonal = np.diag(self.diagonal())
            if np.alltrue(data_diagonal == data):
                return data
            else:
                raise ValueError("Bad format for data")
        else:
            raise ValueError("Bad format for data")

    def zipWithEnergy(self):
        return zip(self.bins.centers, self.diagonal())


class A(object):
    def __init__(self):
        print "Init A"

class B(A):
    def __init__(self):
        print "Init B"
if __name__ == "__main__":
    # Example run
    print "Computing oscillated flux"
    flux = BeamFlux('flux.csv')
    oscprob = OscillationProbability('prob.csv')
    oscflux = oscprob * flux
    print "nue flux\n", oscflux.view('nue flux')
    print "numu flux\n", oscflux.view('numu flux')

    print "Fetch change in flux from one delta-CP to another,",
    print "as a function of energy"
    oscprob_newdCP = OscillationProbability('prob2.csv')
    oscflux2 = oscprob2 * flux
    diff = oscflux2 - oscflux
    print diff.view('nue flux', withEnergy=True)
