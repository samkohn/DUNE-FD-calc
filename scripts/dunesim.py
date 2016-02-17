import numpy as np
import csv

def setEnergyBins(bins):
    SimulationComponent.defaultBinning = Binning(bins)

class SimulationComponent(np.matrix):
    defaultBinning = None
    def __new__(cls, arg): # TODO **kwargs
        """
        Read in data from either an array-like object or a file location
        and assign it to the np.matrix (inherited) data structure.

        This base class method reads in the data, and each subclass must
        define how to convert that data into an np.matrix via the
        _getMatrixForm method.

        """
        if cls.defaultBinning is None:
            raise Exception("Must define " +
                "SimulationComponent.defaultBinning first")
        data = None
        try: # assume data is a numpy array or array-like
            data = np.asanyarray(arg, dtype=np.float64)
        except ValueError: # maybe data is a location of a file
            location = arg
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
        except: # maybe it's unknown
            raise
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
        if self.bins is None:
            self.bins = self.defaultBinning

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
    def __new__(cls, arg):
        obj = SimulationComponent.__new__(cls, arg)
        obj.bins = cls.defaultBinning
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

class Spectrum(SimulationComponent):
    def __new__(cls, arg):
        obj = SimulationComponent.__new__(cls, arg)
        obj.bins = cls.defaultBinning
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

    def extract(self, name, withEnergy=False):
        thing = None
        if withEnergy:
            thing = self.zipWithEnergy()
        else:
            thing = self
        if name == 'nue spectrum':
            return np.asarray(thing[0:self.bins.n])
        if name == 'numu spectrum':
            return np.asarray(thing[self.bins.n:2*self.bins.n])
        if name == 'nutau spectrum':
            return np.asarray(thing[2*self.bins.n:3*self.bins.n])
        raise ValueError("Bad name")

class OscillationProbability(SimulationComponent):
    nextFormat = BeamFlux
    def __new__(cls, arg):
        obj = SimulationComponent.__new__(cls, arg)
        obj.bins = cls.defaultBinning
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

class CrossSection(SimulationComponent):
    nextFormat = Spectrum
    def __new__(cls, arg):
        obj = SimulationComponent.__new__(cls, arg)
        obj.bins = cls.defaultBinning
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

    def extract(self, name, withEnergy=False):
        thing = None
        if withEnergy:
            thing = self.zipWithEnergy()
        else:
            thing = self.diagonal()
        if name == 'nue':
            return np.asarray(thing[0:self.bins.n])
        if name == 'numu':
            return np.asarray(thing[self.bins.n:2*self.bins.n])
        if name == 'nutau':
            return np.asarray(thing[2*self.bins.n:3*self.bins.n])
        raise ValueError("Bad name")

class DetectorResponse(SimulationComponent):
    """
    Detector Response matrix.

    The true energy increases along a row (i.e. the second index gives
    the true energy). The reconstructed energy increases down a column
    (i.e. the first index gives the reconstructed energy). The
    normalization should be that a true particle ends up somewhere
    (unitarity), so that the sum down a column is 1.
    I.e., sum(obj[:,i]) == 1 for all i.

    """
    nextFormat = Spectrum
    def __new__(cls, arg):
        obj = SimulationComponent.__new__(cls, arg)
        obj.bins = cls.defaultBinning
        return obj

    @staticmethod
    def _getMatrixForm(data):
        """
        Ensure that the data is a matrix of the appropriate dimensions
        and return it.

        No guarantee on whether the return value is a reference to or a
        copy of the original data.

        """
        data = np.asarray(data)
        n = SimulationComponent.defaultBinning.n
        if data.ndim != 2:
            raise ValueError("Input is not a 2-D matrix")
        else:
            return data

    def zipWithEnergy(self, form='list'):
        """
        Pair up the energy values corresponding to the entries in the
        matrix.

        If form == 'list' (default), return an n^2-length array of the form:
            [((true_e[0], reco_e[0]), value[0, 0]),
             ((true_e[1], reco_e[0]), value[0, 1]),
             ...
             ((true_e[n-1], reco_e[n-1]), value[n-1, n-1])]


        If form == 'matrix', then return an n x n x 3 array with the
        following structure:

        If x = obj.zipWithenergy(), then
        x[i][j] = [ith energy bin, jth energy bin, value of obj[i][j]]

        """
        energylist = np.tile(self.bins.centers, 3)
        if form == 'matrix':
            energymatrix = np.array([[(e1, e2) for e2 in energylist]
                for e1 in energylist])
            return np.dstack((energymatrix, self))
        if form == 'list':
            energymatrix = np.array([(e1, e2) for e2 in energylist for
                e1 in energylist])
            drm = self.flatten()
            return zip(energymatrix, drm)
        raise ValueError("Did not recognize form " + str(form))

class Efficiency(SimulationComponent):
    """
    The efficiency for detecting and recognizing a particle of a given
    energy and flavor as a function of all of the spectra.

    Allows for cross-contamination/background, e.g. a nu-mu being
    mistaken as a nu-e, through the use of off-diagonal entries.

    The true particle increases along the rows, and the recognized
    particle increases down the columns. This way, obj[j, k] is the
    fraction of particle k's which are labeled as a particle j.

    Note: unitarity is not required here, as it is possible (likely)
    that some particles are simply lost.

    """
    nextFormat = Spectrum
    def __new__(cls, arg):
        obj = SimulationComponent.__new__(cls, arg)
        obj.bins = cls.defaultBinning
        return obj

    @staticmethod
    def _getMatrixForm(data):
        """
        Interpret a 1-D array as a diagonal matrix, and any other matrix
        as itself.

        """
        data = np.asarray(data)
        if data.ndim == 1:
            return np.diag(data)
        elif data.ndim == 2:
            return data

    def zipWithEnergy(self, form='list'):
        """
        Pair up the energy values corresponding to the entries in the
        matrix.

        If form == 'list' (default), return an n^2-length array of the form:
            [((in_particle_e[0], out_particle_e[0]), efficiency[0, 0]),
             ((in_particle_e[1], out_particle_e[0]), efficiency[0, 1]),
             ...
             ((in_particle_e[n-1], out_particle_e[n-1]), efficiency[n-1, n-1])]


        If form == 'matrix', then return an n x n x 3 array with the
        following structure:

        If x = obj.zipWithenergy(), then
        x[i][j] = [ith energy bin, jth energy bin, value of obj[i][j]]

        """
        energylist = np.tile(self.bins.centers, 3)
        if form == 'matrix':
            energymatrix = np.array([[(e1, e2) for e2 in energylist]
                for e1 in energylist])
            return np.dstack((energymatrix, self))
        if form == 'list':
            energymatrix = np.array([(e1, e2) for e2 in energylist for
                e1 in energylist])
            efficiencies = self.flatten()
            return zip(energymatrix, drm)
        raise ValueError("Did not recognize form " + str(form))


if __name__ == "__main__":
    # Example run
    setEnergyBins(np.arange(0, 10.25, 0.25))
    print "Computing oscillated flux"
    flux = \
    BeamFlux('../Fast-Monte-Carlo/Flux-Configuration/nuflux_numuflux_numu40.csv')
    oscprob = \
    OscillationProbability('../Fast-Monte-Carlo/Oscillation-Parameters/numu_nue40.csv')
    xsec = \
    CrossSection('../Fast-Monte-Carlo/Cross-Sections/nu_e_Ar40__tot_cc40.csv')
    detectorresponse = \
    DetectorResponse('../Fast-Monte-Carlo/Detector-Response/DetRespMat-nuflux_numuflux_nue.csv')
    efficiency = \
    Efficiency('../Fast-Monte-Carlo/Efficiencies/nueCCsig_efficiency.csv')
    oscflux = flux.evolve(oscprob)
    print "nue flux\n", oscflux.extract('nue flux')
    print "numu flux\n", oscflux.extract('numu flux')
    signalspec = (flux.evolve(oscprob)
            .evolve(xsec)
            .evolve(detectorresponse)
            .evolve(efficiency))
    print "Python type of signal spectrum = ", type(signalspec)
    print "nue spectrum = "
    print signalspec.extract('nue spectrum')

    print "Fetch change in flux from one delta-CP to another,",
    print "as a function of energy"
    oscprob2 = OscillationProbability(np.diag(oscprob.diagonal() +
            0.01))#OscillationProbability('prob2.csv')
    oscflux2 = flux.evolve(oscprob2)
    diff = oscflux2 - oscflux
    print diff.extract('nue flux', withEnergy=True)
