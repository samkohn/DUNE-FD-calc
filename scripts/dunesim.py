import numpy as np
import csv

def setEnergyBins(bins):
    SimulationComponent.defaultBinning = Binning(bins)

class SimulationComponent(np.matrix):
    """
    This is the base class for all of the data structures used in the
    simulation.

    To create a new component, provide one of the following:
      - the location of a .csv file
      - a python/numpy data structure such as a list or ndarray
      - an array of file locations in "block matrix" format, where the
        contents of the files will fill in the resulting matrix in the
        positions they appear in the array. E.g.:
           ['nue_flux.csv', 'numu_flux.csv', 'nutau_flux.csv']
        will result in a single column vector with first the nue flux,
        then the numu flux, and then the nutau flux. Check with the
        documentation for each class for the appropriate format.

    The internal of the data is column vectors and matrices. For something
    like a flux, a column vector should be supplied (a text file with
    one entry on each line, or a 1D list or ndarray). For something like
    the detector response matrix or efficiency matrix, a matrix should
    be supplied (a comma-separated text file with rows corresponding to
    matrix rows and columns corresponding to matrix columns). Each
    subclass's method _getMatrixForm has a docstring specifying how the
    input data is converted to the appropriate data structure.

    """
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
                "binning with setEnergyBins() first")
        data = np.asanyarray(arg)
        dtype = data.dtype
        # Different cases for different argument types
        if (np.issubdtype(dtype, np.float) or
                np.issubdtype(dtype, np.int)):
            # then a matrix was supplied
            data = np.asanyarray(data, dtype=np.float64)
        elif ('S' in str(dtype) or 'a' in str(dtype) or 'U' in
                str(dtype)):
            # Then a string or array of strings was supplied
            if data.shape == ():
                # Then the supplied argument is a file location
                data = cls._parseFile(arg)
                data = cls._getMatrixForm(data)
            else:
                # Then the supplied argument is a set of file locations
                # describing a block matrix. E.g. a file for each
                # flavor combination's oscillation probabilities.
                # First read in the files and store the results in a 2d
                # list
                try:
                    dataarray = [[cls._parseFile(name) for name in row]
                            for row in arg]
                    # Re-format each element of the list
                    blockmatrix = [[cls._getMatrixForm(chunk) for chunk
                        in row] for row in dataarray]
                    # Convert from "block" matrix to real matrix
                    data = np.bmat(blockmatrix)
                except IOError: # Happens if supplied list is 1-D
                    print "Input is 1D array"
                    dataarray = [cls._parseFile(name) for name in arg]
                    blockmatrix = [cls._getMatrixForm(chunk) for chunk
                            in dataarray]
                    data = np.array(blockmatrix).flatten()
        else:
            raise ValueError('Bad argument to constructor.')
        # Store the data in the underlying np.matrix structure
        obj = np.matrix.__new__(cls, data.view(cls))
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

    @staticmethod
    def _parseFile(location):
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
        else: # No problem
            pass
        return data

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
        Ensure that the data is either a column vector or a matrix and
        return it.

        If the data is a column vector, convert it into a diagonal
        matrix.

        No guarantee on whether the return value is distinct from the
        original data.

        """
        data = np.asarray(data)
        if data.ndim == 1:
            return np.diag(data)
        elif data.ndim == 2:
            return data
        else:
            raise ValueError("Bad format for data")

    def zipWithEnergy(self):
        """
        Pair up the energy values corresponding to the entries in the
        matrix.

        Return an n x n x 3 array with the following structure:

        If x = obj.zipWithenergy(), then
        x[i][j] = [ith energy bin, jth energy bin, value of obj[i][j]]

        """
        energylist = np.tile(self.bins.centers, 3)
        energymatrix = np.array([[(e1, e2) for e2 in energylist]
            for e1 in energylist])
        return np.dstack((energymatrix, self))

    def extract(self, name, withEnergy=False):
        thing = None
        n = self.bins.n
        if withEnergy:
            thing = self.zipWithEnergy()
        else:
            thing = self
        if name == 'nue2nue':
            return np.asarray(thing[0:n, 0:n])
        if name == 'nue2numu':
            return np.asarray(thing[n:2*n, 0:n])
        if name == 'nue2nutau':
            return np.asarray(thing[2*n:3*n, 0:n])
        if name == 'numu2nue':
            return np.asarray(thing[0:n, n:2*n])
        if name == 'numu2numu':
            return np.asarray(thing[n:2*n, n:2*n])
        if name == 'numu2nutau':
            return np.asarray(thing[2*n:3*n, n:2*n])
        if name == 'nutau2nue':
            return np.asarray(thing[0:n, 2*n:3*n])
        if name == 'nutau2numu':
            return np.asarray(thing[n:2*n, 2*n:3*n])
        if name == 'nutau2nutau':
            return np.asarray(thing[2*n:3*n, 2*n:3*n])
        raise ValueError("Bad name")

class CrossSection(SimulationComponent):
    nextFormat = Spectrum
    def __new__(cls, arg, units=1e-38):
        obj = SimulationComponent.__new__(cls, arg)
        obj.bins = cls.defaultBinning
        obj *= units
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

    def extract(self, name, withEnergy=False):
        thing = None
        n = self.bins.n
        if withEnergy:
            thing = self.zipWithEnergy('matrix')
        else:
            thing = self
        if name == 'nue2nue':
            return np.asarray(thing[0:n, 0:n])
        if name == 'nue2numu':
            return np.asarray(thing[n:2*n, 0:n])
        if name == 'nue2nutau':
            return np.asarray(thing[2*n:3*n, 0:n])
        if name == 'numu2nue':
            return np.asarray(thing[0:n, n:2*n])
        if name == 'numu2numu':
            return np.asarray(thing[n:2*n, n:2*n])
        if name == 'numu2nutau':
            return np.asarray(thing[2*n:3*n, n:2*n])
        if name == 'nutau2nue':
            return np.asarray(thing[0:n, 2*n:3*n])
        if name == 'nutau2numu':
            return np.asarray(thing[n:2*n, 2*n:3*n])
        if name == 'nutau2nutau':
            return np.asarray(thing[2*n:3*n, 2*n:3*n])
        raise ValueError("Bad name")


if __name__ == "__main__":
    print """WARNING: the files used do not contain full data on all three
    neutrino flavors. As a consequence, some of the outputs are empty
    arrays. As this is just an example anyways, do not trust the
    numerical results of this output.\n\n\n"""
    # Example run
    setEnergyBins(np.arange(0, 10.25, 0.25))

    # Set up block matrices for the flux, oscillation probabilities, and
    # cross sections. Note that the flux is vectorlike and so comes as a
    # 1-D array, but the oscillation probabilities and cross sections
    # are matrix-like, so they come in 2D arrays.
    pre = '../Fast-Monte-Carlo/Flux-Configuration/numode_'
    fluxfiles = map(lambda x: pre + x, ['nue_flux40.csv', 'numu_flux40.csv',
        'nutau_flux40.csv'])
    pre = '../Fast-Monte-Carlo/Oscillation-Parameters/nu'
    oscfiles = [['e_nue40.csv', 'mu_nue40.csv', 'tau_nue40.csv'],
             ['e_numu40.csv', 'mu_numu40.csv', 'tau_numu40.csv'],
             ['e_nutau40.csv', 'mu_nutau40.csv', 'tau_nutau40.csv']]
    oscfiles = [[pre + name for name in row] for row in oscfiles]
    pre = '../Fast-Monte-Carlo/Cross-Sections/nu_'
    xsecfiles = [map(lambda x: pre + x, ['e_Ar40__tot_cc40.csv',
        'mu_Ar40__tot_cc40.csv', 'tau_Ar40__tot_cc40.csv'])]
    flux = BeamFlux(fluxfiles)
    oscprob = OscillationProbability(oscfiles)
    xsec = CrossSection(xsecfiles)
    detectorresponse = \
    DetectorResponse('../Fast-Monte-Carlo/Detector-Response/DetRespMat-nuflux_numuflux_nue.csv')
    efficiency = \
    Efficiency('../Fast-Monte-Carlo/Efficiencies/nueCCsig_efficiency.csv')
    oscflux = flux.evolve(oscprob)
    print "oscillated flux\nnue flux\n", oscflux.extract('nue flux')
    print "numu flux\n", oscflux.extract('numu flux')
    print "nutau flux\n", oscflux.extract('nutau flux')
    print "\n\n\n"
    # Get the total beam flux for 10^20 POT
    NUM_POT = 1e20
    NUM_AR_ATOMS = 6e30
    flux *= NUM_POT
    detectorspec = (flux
            .evolve(oscprob)
            .evolve(xsec))
    detectorspec *= NUM_AR_ATOMS
    print "True spectrum of CC nue events at detector"
    print detectorspec.extract('nue spectrum')
    # (signalspec = detectorspec
            # .evolve(detectorresponse)
            # .evolve(efficiency))
    # print "Python type of signal spectrum = ", type(signalspec)
    # print "nue spectrum = "
    # print signalspec.extract('nue spectrum')

    print "\n\n\n"

    print "Fetch change in flux from one delta-CP to another,",
    print "as a function of energy"
    oscprob2 = OscillationProbability(np.diag(oscprob.diagonal() +
            0.01))#OscillationProbability('prob2.csv')
    oscflux2 = flux.evolve(oscprob2)
    diff = oscflux2 - oscflux
    print diff.extract('nue flux', withEnergy=True)
