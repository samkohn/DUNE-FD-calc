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
                    data = cls._getMatrixForm(data)
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
    """
    A representation of the neutrino beam flux.

    This class keeps track of three flavors of neutrinos. The data
    should be supplied in the order [e flux, mu flux, tau flux] in a
    column vector.

    """
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
    """
    A representation of the spectrum of neutrinos which interact with a
    detector.

    This class keeps track of three flavors of neutrinos and the way
    they interacted (via charged current or neutral current). The data
    should be supplied in the order [eCC, eNC, muCC, muNC, tauCC,
    tauNC] for true spectra and [eCC-like, muCC-like, NC-like] for
    reconstructed spectra.

    """
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
    """
    A representation of the oscillation probability.

    This class keeps track of the oscillation probabilities for a
    particular set of oscillation parameters, baseline, etc. It is used
    to transform one BeamFlux into another BeamFlux that represents the
    oscillated flux.

    The data should be supplied in the following matrix form:
    [[nue->nue, numu->nue, nutau->nue],
     [nue->numu, numu->numu, nutau->numu],
     [nue->nutau, numu->nutau, nutau->nutau]]

    """
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
    """
    A representation of the interaction cross section.

    This class keeps track of the interaction cross sections for three
    flavors of neutrinos and two interaction channels (CC and NC). It
    converts a BeamFlux into a Spectrum (in particular, a true
    spectrum).

    The data should be supplied in one of the two following ways:

     - A column vector of the form [nueCC, nueNC, numuCC, numuNC,
       nutauCC, nutauNC]
     - A matrix of the form
         [[nueCC, 0, 0],
          [nueNC, 0, 0],
          [0, numuCC, 0],
          [0, numuNC, 0],
          [0, 0, nutauCC],
          [0, 0, nutauNC]]

    """
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
        elif data.ndim == 2 and data.shape[0] == data.shape[1]:
            data_diagonal = np.diag(self.diagonal())
            if np.alltrue(data_diagonal == data):
                return data
            else:
                raise ValueError("Bad format for data")
        elif data.ndim == 2:
            rows = data.shape[0]
            cols = data.shape[1]
            if rows > cols:
                stride = cols
                result = np.zeros((rows, rows), dtype=data.dtype)
                empty = np.zeros_like(data[:stride,:stride]) # empty square
                ratio = rows/cols
            else:
                stride = rows
                result = np.zeros((cols, cols), dtype=data.dtype)
                empty = np.zeros_like(data[:stride, :stride])
                ratio = cols/rows
            for row in range(ratio):
                startrow = row * stride
                endrow = startrow + stride
                for col in range(ratio):
                    startcol = col * stride
                    endcol = startcol + stride
                    if row == col:
                        if rows > cols:
                            assignment = data[startrow:endrow, :]
                        else:
                            assignment = data[:, startcol:endcol]
                    else:
                        assignment = empty
                    result[startrow:endrow, startcol:endcol] = assignment
            return result

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

    The true energy increases along a row (i.e. the second index
    gives the true energy). The reconstructed energy increases down a
    column (i.e. the first index gives the reconstructed energy). The
    normalization should be that a true particle ends up somewhere
    (unitarity), so that the sum down a column is 1. I.e., sum(obj[:,i])
    == 1 for all i. The exception is if some events are rejected as not
    a neutrino event at all.

    The detector response matrix converts a true spectrum into a
    reconstructed spectrum. Its input should be broken down into
    interactions by flavor and channel (e.g. nueCC, nutauNC). Its output
    will be broken down either in the same way, or by reconstructed
    channel (namely nueCC-like, numuCC-like, NC-like), depending on the
    form of the matrix (see next paragraph).

    The matrix should be supplied in one of the following forms:
     - More precise: if the detector response, including event channel
       ID, is to be used all together, use a matrix:

       [[eCC->eCC-like, eNC->eCC-like, ..., tauNC->eCC-like],
        [eCC->muCC-like, ..., tauNC->muCC-like],
        [eCC->NC-like, ..., tauNC->NC-like]]

       Output is a reconstructed spectrum (eCC-like, muCC-like,
       NC-like).

     - Less precise: if the energy response and the event
       classification are to be used separately (approximately true),
       use a column vector, which will preserve the event channel
       information.

       [eCC, eNC, muCC, muNC, tauCC, tauNC]

       Output is still a "true" spectrum format (eCC, eNC, ..., tauNC).
       The event classification must be performed later using the
       Efficiency object.

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

    This should only be used if the detector response matrix is the
    "less precise" kind, where the energy reconstruction and event
    channel ID are separate. If a full DRM is used, this class is not
    necessary as the DRM handles both together. The output is a
    reconstructed spectrum of the form (nueCC-like, numuCC-like,
    NC-like).

    The supplied data should be of the form:

     [[eCC->eCC-like, eNC->eCC-like, ..., tauNC->eCC-like],
      [eCC->muCC-like, ..., tauNC->muCC-like],
      [eCC->NC-like, ..., tauNC->NC-like]]

    To ensure the separation between energy reconstruction and
    interaction channel ID, each sub-block of the matrix (corresponding
    to one of the blocks in the description above (e.g. eCC->eCC-like)
    should be diagonal, so that no particles of a particular energy end
    up in a bin of a different energy. Ultimately, though, it's on you
    to decide how you want to use this functionality.

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
