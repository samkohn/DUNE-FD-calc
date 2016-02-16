import numpy as np
import csv

class SimulationComponent(np.matrix):
    def __new__(cls, location): # TODO **kwargs
        """
        Read in data from location and assign it to the np.matrix
        (inherited) data structure.

        This base class method reads in the data, and each subclass must
        define how to convert that data into an np.matrix.

        """
        # read in data
        data = []
        with open(location) as fin:
            reader = csv.reader(fin)
            for row in reader:
                data.append(map(float, row))
        # This conditional remedies various csv formatting styles (e.g.
        # all one row or all one column)
        if len(data) == 1: # all one row
            data = data[0]
        elif len(data[0]) == 1: # all one column
            data = zip(*data)[0]
        data = cls._getMatrixForm(data)
        # Store the data in the underlying np.matrix structure
        obj = np.matrix.__new__(cls, data)
        obj.description = None
        obj.bins = None # Instance of Binning object
        obj.dataFileLocation = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.description = getattr(obj, 'description', None)
        self.bins = getattr(obj, 'bins', None)
        self.dataFileLocation = getattr(obj, 'dataFileLocation', None)

    def evolve(self, other):
        result = other * self
        newDataFormat = other.nextFormat
        return result.view(newDataFormat)

    @staticmethod
    def _getMatrixForm(data):
        """
        Format the data in the appropriate matrix representation.

        E.g. matrix, diagonal matrix, or column vector.

        """
        raise NotImplementedError("Must override this method in " +
        " a subclass.")

class Binning(object):
    def __init__(self, edges):
        self.edges = np.array(edges)
        self.centers = np.empty(self.edges.shape[0]-1)
        self.start = self.edges[0]
        self.end = self.edges[-1]
        for ind, (i, j) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            self.centers[ind] = ((i + j)/2.0)
        self.n = len(self.centers)


class BeamFlux(SimulationComponent):
    def __new__(cls, location):
        obj = SimulationComponent.__new__(cls, location)
        obj.bins = Binning(np.arange(0, 10.25, 0.25))
        return obj

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
        return zip(np.tile(self.bins.centers, 3), self)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.description = getattr(obj, 'description', None)
        self.bins = getattr(obj, 'bins', None)
        self.dataFileLocation = getattr(obj, 'dataFileLocation', None)
        if self.bins is None:
            self.bins = Binning(np.arange(0, 10.25, 0.25))


    def extract(self, name, withEnergy=False):
        thing = None
        if withEnergy:
            thing = self.zipWithEnergy()
        else:
            thing = self
        if name == 'nue flux':
            return np.asarray(thing[0:self.bins.n])
        if name == 'numu flux':
            return np.asarray(thing[self.bins.n:2*self.bins.n])
        if name == 'nutau flux':
            return np.asarray(thing[2*self.bins.n:3*self.bins.n])
        raise ValueError("Bad name")

class OscillationProbability(SimulationComponent):
    nextFormat = BeamFlux
    def __new__(cls, location):
        obj = SimulationComponent.__new__(cls, location)
        obj.bins = Binning(np.arange(0, 10.25, 0.25))
        return obj

    @staticmethod
    def _getMatrixForm(data):
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
        return zip(np.tile(self.bins.centers, 3), self.diagonal())

    def __array_finalize__(self, obj):
        if obj is None: return
        self.description = getattr(obj, 'description', None)
        self.bins = getattr(obj, 'bins', None)
        self.dataFileLocation = getattr(obj, 'dataFileLocation', None)
        if self.bins is None:
            self.bins = Binning(np.arange(0, 10.25, 0.25))

    def extract(self, name, withEnergy=False):
        thing = None
        if withEnergy:
            thing = self.zipWithEnergy()
        else:
            thing = self.diagonal()
        if name == 'nue2nue':
            return np.asarray(thing[0:self.bins.n])
        if name == 'nue2numu':
            return np.asarray(thing[self.bins.n:2*self.bins.n])
        if name == 'nue2nutau':
            return np.asarray(thing[2*self.bins.n:3*self.bins.n])
        raise ValueError("Bad name")

if __name__ == "__main__":
    # Example run
    print "Computing oscillated flux"
    flux = \
    BeamFlux('../Fast-Monte-Carlo/Flux-Configuration/nuflux_nueflux_nue40.csv')
    oscprob = \
    OscillationProbability('../Fast-Monte-Carlo/Oscillation-Parameters/numu_nue40.csv')
    oscflux = flux.evolve(oscprob)
    print "nue flux\n", oscflux.extract('nue flux')
    print "numu flux\n", oscflux.extract('numu flux')

    print "Fetch change in flux from one delta-CP to another,",
    print "as a function of energy"
    oscprob2 = np.diag(oscprob.diagonal() + 0.01)#OscillationProbability('prob2.csv')
    oscflux2 = flux.evolve(oscprob2)
    diff = oscflux2 - oscflux
    print diff.extract('nue flux', withEnergy=True)
