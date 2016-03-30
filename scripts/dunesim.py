"""
Module dunesim
Author: Samuel Kohn, skohn@lbl.gov
License: MIT License (c) 2016 Samuel Kohn

This module provides an interface into the calculation of the
neutrino spectrum detected at the DUNE Far Detector. The module is
composed of two parts: one is a set of classes that represent
parametrizations of the components of the calculation (spectrum, flux,
detector response, etc.). The other part is a small set of convenience
functions for setting the bin spacing and loading default data sets into
classes.

An example of how to use this module is in the python file
`dunesimexample.py`.

"""

import numpy as np
import csv
import os
import sys
import pdb


def setEnergyBins(bins):
    """
    Configure the energy bins used in the files loaded into the dunesim
    classes.

    `bins` should be an array of the N+1 bin edges, e.g. for bins of 1
    GeV covering 0-10GeV, `bins` should be `[0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10]`. An easy way to do this is to use `numpy.linspace`:

    ```
    setEnergyBins(numpy.linspace(0, 10, 11, endpoint=True))
    ```

    passes in the exact array specified above. 0 and 10 are the
    endpoints, 11 is the desired length of the array (= N+1), and
    endpoint=True tells numpy to include 10 (the endpoint) as one of the
    entries. `numpy.arange` also works but the documentation notes that
    using it with floating-point arguments can be finnicky, and that
    linspace works better.

    """
    SimulationComponent.defaultBinning = Binning(bins)


def _setUpRepositoryDir():
    """
    Fetch the file path of the repository this module is located in.

    This function is used to provide absolute paths to the locations of
    data files. It is only called in the `default...()` methods.

    """
    if 'repositorydir' not in globals().keys():
        try:
            global repositorydir
            repositorydir = os.environ['DUNECONFIGSROOT']
        except KeyError:
            print "ERROR: Cannot find environment variable `$DUNECONFIGSROOT`"
            print "ERROR: Source the setup script:"
            print "       $ source setup"
            print "ERROR: and try again."
            print "INFO: Aborting..."
            sys.exit()


def defaultBeamFlux(neutrinomode=True):
    """
    Create a BeamFlux object from the flux data stored in this git
    repository.

    If `neutrinomode` is False, return the flux used by the antineutrino
    mode.

    """
    _setUpRepositoryDir()
    if neutrinomode:
        directory = 'CD1-CDR-FHC/'
        suffix = '_flux40.csv'
    else:
        directory = 'CDR-RHC/'
        suffix = '_flux40_anumode.csv'
    pre = (repositorydir +
           '/Fast-Monte-Carlo/Flux-Configuration/' + directory)
    fluxfiles = map(
            lambda x: pre + x + suffix,
            ['nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'])
    flux = BeamFlux(fluxfiles)
    return flux


def defaultOscillationProbability():
    """
    Create an OscillationProbability object from the files stored in
    this git repository.

    These parameters are the Nu-Fit 2014 best fit values, normal
    ordering, with delta CP = 0.

    """
    _setUpRepositoryDir()
    pre = repositorydir + '/Fast-Monte-Carlo/Oscillation-Parameters/nu'
    oscfiles = [['e_nue40.csv', 'mu_nue40.csv', 'tau_nue40.csv'],
                ['e_numu40.csv', 'mu_numu40.csv', 'tau_numu40.csv'],
                ['e_nutau40.csv', 'mu_nutau40.csv', 'tau_nutau40.csv'],
                ['ebar_nuebar40.csv', 'mubar_nuebar40.csv', 'taubar_nuebar40.csv'],
                ['ebar_numubar40.csv', 'mubar_numubar40.csv', 'taubar_numubar40.csv'],
                ['ebar_nutaubar40.csv', 'mubar_nutaubar40.csv', 'taubar_nutaubar40.csv']]
    oscfiles = [[pre + name for name in row] for row in oscfiles]
    oscprob = OscillationProbability(oscfiles)
    return oscprob


def defaultCrossSection():
    """
    Create a CrossSection object from the files stored in this git
    repository.

    Includes the total cross section broken down by NC and CC for
    neutrinos and antineutrinos on Argon 40.

    """
    _setUpRepositoryDir()
    pre = repositorydir + '/Fast-Monte-Carlo/Cross-Sections/nu_'
    xsecfiles = map(
        lambda x: pre + x,
        ['e_Ar40__tot_cc40.csv', 'e_Ar40__tot_nc40.csv',
         'mu_Ar40__tot_cc40.csv', 'mu_Ar40__tot_nc40.csv',
         'tau_Ar40__tot_cc40.csv', 'tau_Ar40__tot_nc40.csv',
         'e_bar_Ar40__tot_cc40.csv', 'e_bar_Ar40__tot_nc40.csv',
         'mu_bar_Ar40__tot_cc40.csv', 'mu_bar_Ar40__tot_nc40.csv',
         'tau_bar_Ar40__tot_cc40.csv', 'tau_bar_Ar40__tot_nc40.csv'])
    xsec = CrossSection(xsecfiles)
    return xsec


def defaultDetectorResponse(factored=True):
    """
    Create a DetectorResponse object from the files stored in this
    repository.

    The values are extracted from the Fast Monte Carlo.

    If `factored` is True, the returned object will only perform an
    energy reconstruction without assigning reconstructed interaction
    channel labels to the events. I.e. the energy distribution may
    change, but the number of events in each interaction type will be
    conserved. If `factored` is False, the returned object will assign
    both a new energy and a reconstructed interaction label to events.
    I.e. a true nue CC event at 1.5 GeV may end up as a reconstructed
    NC-like event at 0.5 GeV.

    """
    _setUpRepositoryDir()
    pre = repositorydir + '/Fast-Monte-Carlo/Detector-Response/nuflux_numu'
    if factored:
        drmfiles = ['flux_nue_trueCC', 'flux_nue_trueNC',
                    'flux_numu_trueCC', 'flux_numu_trueNC',
                    'flux_nutau_trueCC', 'flux_nutau_trueNC',
                    'barflux_nuebar_trueCC', 'barflux_nuebar_trueNC',
                    'barflux_numubar_trueCC', 'barflux_numubar_trueNC',
                    'barflux_nutaubar_trueCC', 'barflux_nutaubar_trueNC']
        drmfiles = [pre + name + '40.csv' for name in drmfiles]
    else:
        drmfiles = [
                ['flux_nue_nueCC-like_trueCC', 'flux_nue_nueCC-like_trueNC',
                 'flux_numu_nueCC-like_trueCC', 'flux_numu_nueCC-like_trueNC',
                 'flux_nutau_nueCC-like_trueCC', 'flux_nutau_nueCC-like_trueNC',
                 'barflux_nuebar_nueCC-like_trueCC', 'barflux_nuebar_nueCC-like_trueNC',
                 'barflux_numubar_nueCC-like_trueCC', 'barflux_numubar_nueCC-like_trueNC',
                 'barflux_nutaubar_nueCC-like_trueCC', 'barflux_nutaubar_nueCC-like_trueNC'],
                ['flux_nue_numuCC-like_trueCC', 'flux_nue_numuCC-like_trueNC',
                 'flux_numu_numuCC-like_trueCC', 'flux_numu_numuCC-like_trueNC',
                 'flux_nutau_numuCC-like_trueCC', 'flux_nutau_numuCC-like_trueNC',
                 'barflux_nuebar_numuCC-like_trueCC', 'barflux_nuebar_numuCC-like_trueNC',
                 'barflux_numubar_numuCC-like_trueCC', 'barflux_numubar_numuCC-like_trueNC',
                 'barflux_nutaubar_numuCC-like_trueCC', 'barflux_nutaubar_numuCC-like_trueNC'],
                ['flux_nue_NC-like_trueCC', 'flux_nue_NC-like_trueNC',
                 'flux_numu_NC-like_trueCC', 'flux_numu_NC-like_trueNC',
                 'flux_nutau_NC-like_trueCC', 'flux_nutau_NC-like_trueNC',
                 'barflux_nuebar_NC-like_trueCC', 'barflux_nuebar_NC-like_trueNC',
                 'barflux_numubar_NC-like_trueCC', 'barflux_numubar_NC-like_trueNC',
                 'barflux_nutaubar_NC-like_trueCC', 'barflux_nutaubar_NC-like_trueNC']
            ]
        drmfiles = [[pre + name + '40.csv' for name in row]
                    for row in drmfiles]
    drm = DetectorResponse(drmfiles)
    return drm


def defaultEfficiency(neutrinomode=True):
    _setUpRepositoryDir()
    modestr = '' if neutrinomode else 'a'
    pre = repositorydir + '/Fast-Monte-Carlo/Efficiencies/nuflux_numu'
    files = [
            ['flux_nue_nueCC-like_trueCC', 'flux_nue_nueCC-like_trueNC',
             'flux_numu_nueCC-like_trueCC', 'flux_numu_nueCC-like_trueNC',
             'flux_nutau_nueCC-like_trueCC', 'flux_nutau_nueCC-like_trueNC',
             'barflux_nuebar_nueCC-like_trueCC', 'barflux_nuebar_nueCC-like_trueNC',
             'barflux_numubar_nueCC-like_trueCC', 'barflux_numubar_nueCC-like_trueNC',
             'barflux_nutaubar_nueCC-like_trueCC', 'barflux_nutaubar_nueCC-like_trueNC'],
            ['flux_nue_numuCC-like_trueCC', 'flux_nue_numuCC-like_trueNC',
             'flux_numu_numuCC-like_trueCC', 'flux_numu_numuCC-like_trueNC',
             'flux_nutau_numuCC-like_trueCC', 'flux_nutau_numuCC-like_trueNC',
             'barflux_nuebar_numuCC-like_trueCC', 'barflux_nuebar_numuCC-like_trueNC',
             'barflux_numubar_numuCC-like_trueCC', 'barflux_numubar_numuCC-like_trueNC',
             'barflux_nutaubar_numuCC-like_trueCC', 'barflux_nutaubar_numuCC-like_trueNC'],
            ['flux_nue_NC-like_trueCC', 'flux_nue_NC-like_trueNC',
             'flux_numu_NC-like_trueCC', 'flux_numu_NC-like_trueNC',
             'flux_nutau_NC-like_trueCC', 'flux_nutau_NC-like_trueNC',
             'barflux_nuebar_NC-like_trueCC', 'barflux_nuebar_NC-like_trueNC',
             'barflux_numubar_NC-like_trueCC', 'barflux_numubar_NC-like_trueNC',
             'barflux_nutaubar_NC-like_trueCC', 'barflux_nutaubar_NC-like_trueNC']
        ]
    files = [[pre + name + '40.csv' for name in row] for row in files]
    efficiency = Efficiency(files)
    return efficiency


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

    The internal structure of the data is column vectors and matrices.
    For something like a flux, a column vector should be supplied (a
    text file with one entry on each line, or a 1D list or ndarray). For
    something like the detector response matrix or efficiency matrix,
    a matrix should be supplied (a comma-separated text file with rows
    corresponding to matrix rows and columns corresponding to matrix
    columns). Each subclass's method _getMatrixForm has a docstring
    specifying how the input data is converted to the appropriate data
    structure.

    """
    defaultBinning = None

    def __new__(cls, arg):
        """
        Read in data from an array-like object of data, a file location,
        or an array-like object of file locations and assign it to the
        np.matrix (inherited) data structure.

        This base class method reads in the data, and each subclass must
        define how to convert that data into an np.matrix via the
        _getMatrixForm and _getBlockMatrixForm methods.

        """
        if cls.defaultBinning is None:
            raise Exception("Must define binning with setEnergyBins " +
                            "first")
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
                blocknames = cls._getBlockMatrixForm(arg)
                data = cls._translateBlockMatrixToMatrix(blocknames)
        else:
            raise ValueError('Bad argument to constructor.')
        # Store the data in the underlying np.matrix structure
        obj = np.matrix.__new__(cls, data.view(cls))
        obj.bins = None  # Instance of Binning object
        return obj

    def __array_finalize__(self, obj):
        """
        Finish configuring new objects by following numpy instructions.

        This method is important when creating a SimulationComponent
        object as a "view" of an existing object (i.e. without
        explicitly constructing a new one). Documentation for this
        procedure is online [here]
        (https://docs.scipy.org/doc/numpy/user/basics.subclassing.html)

        """
        if obj is None: return
        self.bins = getattr(obj, 'bins', None)
        if self.bins is None:
            self.bins = self.defaultBinning

    @classmethod
    def _translateBlockMatrixToMatrix(cls, blocknames):
        """
        Convert a block matrix of file names into a normal matrix of
        data from the specified files.

        """
        blocksize = cls.defaultBinning.n
        blockshape = blocknames.shape
        fullshape = np.array(blockshape) * blocksize
        result = np.zeros(fullshape, dtype=np.float64)
        if len(blockshape) == 1:
            # Column vector
            for blockrow in range(blockshape[0]):
                fullrow = blockrow * blocksize
                nextrow = fullrow + blocksize
                filename = blocknames[blockrow]
                if len(filename) > 0:
                    rawdata = cls._parseFile(filename)
                    data = cls._getMatrixForm(rawdata)
                    result[fullrow:nextrow] = data
                else:
                    pass  # leave as zeros
        elif len(blockshape) == 2:
            # Block matrix
            for blockrow in range(blockshape[0]):
                fullrow = blockrow * blocksize
                nextrow = fullrow + blocksize
                for blockcol in range(blockshape[1]):
                    fullcol = blockcol * blocksize
                    nextcol = fullcol + blocksize
                    filename = blocknames[blockrow, blockcol]
                    if len(filename) > 0:
                        rawdata = cls._parseFile(filename)
                        data = cls._getMatrixForm(rawdata)
                        result[fullrow:nextrow, fullcol:nextcol] = data
                    else:
                        pass  # leave as zeros
        return result

    @staticmethod
    def _parseFile(location):
        """
        Read in the given CSV file and convert it to a (possibly nested)
        python list.

        """
        data = []
        with open(location) as fin:
            # Filter out comments in the CSV file (row starts with #)
            goodrows = (row for row in fin if not row.startswith('#'))
            reader = csv.reader(goodrows)
            for row in reader:
                data.append(map(float, row))
        # This conditional remedies various csv formatting styles (e.g.
        # all one row or all one column)
        if len(data) == 1:  # all one row
            data = data[0]
        elif len(data[0]) == 1:  # all one column
            data = zip(*data)[0]
        else:  # No problem
            pass
        return data

    def evolve(self, other):
        """
        Apply the effect of `other`.

        This method multiplies the two matrices in the correct
        orientation and formats the output according to the correct
        class. (e.g. myFlux.evolve(myXsec) -> mySpectrum of type
        Spectrum)

        """
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
                                  "a subclass.")


class Binning(object):
    """
    This class keeps track of the energy bins represented by rows and
    columns in the SimulationComponent matrices.

    """
    def __init__(self, edges):
        """
        Create a new Binning object by specifying the bin edges.

        N + 1 edges are required to create an N-bin configuration. The
        easiest way to do this is with the numpy command
        `numpy.linspace(lowedge, highedge, n+1)`.

        """
        self.edges = np.array(edges)
        self.centers = np.empty(self.edges.shape[0]-1)
        self.start = self.edges[0]
        self.end = self.edges[-1]
        for ind, (i, j) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            self.centers[ind] = ((i + j)/2.0)
        self.n = len(self.centers)

    _namedict = {key: np.asarray(val) for key, val in {
            'nue': (0, 1),
            'numu': (1, 2),
            'nutau': (2, 3),
            'nuebar': (3, 4),
            'numubar': (4, 5),
            'nutaubar': (5, 6),
            'nueCC': (0, 1),
            'nueNC': (1, 2),
            'numuCC': (2, 3),
            'numuNC': (3, 4),
            'nutauCC': (4, 5),
            'nutauNC': (5, 6),
            'nuebarCC': (6, 7),
            'nuebarNC': (7, 8),
            'numubarCC': (8, 9),
            'numubarNC': (9, 10),
            'nutaubarCC': (10, 11),
            'nutaubarNC': (11, 12),
            'nueCC-like': (0, 1),
            'numuCC-like': (1, 2),
            'NC-like': (2, 3)
        }.iteritems()}
    """
    This dict translates between the name of a section of the matrix
    and the index of that section, up to a factor of Nbins.

    """

    def index(self, name):
        """
        Retrieve a slice object over the indexes of the region of the
        matrix represented by the given name.

        """
        return slice(*(self._namedict[name] * self.n))


class BeamFlux(SimulationComponent):
    """
    A representation of the neutrino beam flux.

    This class keeps track of three flavors of neutrinos and
    antineutrinos. The data should be supplied in the order [e flux,
    mu flux, tau flux, ebar flux, mubar flux, taubar flux] in a column
    vector.

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

    @staticmethod
    def _getBlockMatrixForm(data):
        """
        Ensure the file names are in a column vector.

        """
        data = np.asarray(data)
        shape = data.shape
        if shape == (6,):
            return data
        else:
            raise ValueError("Incorrect shape (6,) != " + str(shape))

    def zipWithEnergy(self):
        return zip(np.tile(self.bins.centers, 6), self)

    def extract(self, name, withEnergy=False):
        thing = None
        smallslice = self.bins.index(name)
        if withEnergy:
            thing = self.zipWithEnergy()
            return np.asarray(thing[smallslice])
        else:
            thing = self
            return np.asarray(thing[smallslice]).reshape(self.bins.n)


class Spectrum(SimulationComponent):
    """
    A representation of the spectrum of neutrinos which interact with a
    detector.

    This class keeps track of three flavors of neutrinos and
    antineutrinos and the way they interacted (via charged current
    or neutral current). The data should be supplied in the order
    [eCC, eNC, muCC, muNC, tauCC, tauNC, ebarCC, ebarNC, ...] for
    true spectra and [eCC-like, muCC-like, NC-like] for reconstructed
    spectra. Reconstructed spectra combine eCC- and ebarCC-like events
    (and similar for muCC and NC) because there is no charge sign
    discrimination.

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

    @staticmethod
    def _getBlockMatrixForm(data):
        """
        Ensure the data is a column vector of the appropriate size.

        """
        data = np.asarray(data)
        shape = data.shape
        if shape == (3,) or shape == (12,):
            return data
        else:
            raise ValueError("Incorrect shape " + str(shape))

    def zipWithEnergy(self):
        numBlocks = len(self)/self.bins.n
        return zip(np.tile(self.bins.centers, numBlocks), self)

    def extract(self, name, withEnergy=False):
        thing = None
        smallslice = self.bins.index(name)
        if withEnergy:
            thing = self.zipWithEnergy()
            return np.asarray(thing[smallslice])
        else:
            thing = self
            return np.array(thing[smallslice]).reshape(self.bins.n)


class OscillationProbability(SimulationComponent):
    """
    A representation of the oscillation probability.

    This class keeps track of the oscillation probabilities for a
    particular set of oscillation parameters, baseline, etc. It is used
    to transform one BeamFlux into another BeamFlux that represents the
    oscillated flux.

    The data should be supplied in the following block matrix form:
    [[nue->nue, numu->nue, nutau->nue],
     [nue->numu, numu->numu, nutau->numu],
     [nue->nutau, numu->nutau, nutau->nutau],
     [nuebar->nuebar, numubar->nuebar, nutaubar->nuebar],
     [nuebar->numubar, numubar->numubar, nutaubar->numubar],
     [nuebar->nutaubar, numubar->nutaubar, nutaubar->nutaubar]]

    Or in the following matrix form:
    [[nue->nue, numu->nue, nutau->nue, 0, 0, 0],
     [nue->numu, numu->numu, nutau->numu, 0, 0, 0],
     [nue->nutau, numu->nutau, nutau->nutau, 0, 0, 0],
     [0, 0, 0, nuebar->nuebar, numubar->nuebar, nutaubar->nuebar],
     [0, 0, 0, nuebar->numubar, numubar->numubar, nutaubar->numubar],
     [0, 0, 0, nuebar->nutaubar, numubar->nutaubar, nutaubar->nutaubar]]

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

    @staticmethod
    def _getBlockMatrixForm(data):
        """
        Ensure the filenames are in a 6x3 matrix and convert it into a
        6x6 block-diagonal matrix (blocks of 3x3).

        """
        data = np.asarray(data)
        shape = data.shape
        if shape == (6, 3):
            result = np.zeros((6, 6), dtype=data.dtype)
            result[0:3, 0:3] = data[0:3, 0:3]
            result[3:6, 3:6] = data[3:6, 0:3]
            return result
        else:
            raise ValueError("Incorrect shape " + str(shape))

    def zipWithEnergy(self):
        """
        Pair up the energy values corresponding to the entries in the
        matrix.

        Return a 6n x 6n x 3 array with the following structure:

        If x = obj.zipWithenergy(), then
        x[i][j] = [ith energy bin, jth energy bin, value of obj[i][j]]

        """
        energylist = np.tile(self.bins.centers, 6)
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
        # Split the given name into two parts, the "from" and the "to."
        colname, rowname = name.split('2')
        colslice = self.bins.index(colname)
        rowslice = self.bins.index(rowname)
        return np.asarray(thing[rowslice, colslice])


class CrossSection(SimulationComponent):
    """
    A representation of the interaction cross section.

    This class keeps track of the interaction cross sections for three
    flavors of neutrinos and three of antineutrinos, and two interaction
    channels (CC and NC). It converts a BeamFlux into a Spectrum (in
    particular, a true spectrum).

    The data should be supplied in one of the two following ways:

     - A block column vector of the form [nueCC, nueNC, numuCC, numuNC,
       nutauCC, nutauNC, nuebarCC, nuebarNC, numubarCC, numubarNC,
       nutaubarCC, nutaubarNC]
     - A matrix of the form
         [[nueCC, 0, 0, 0, 0, 0],
          [nueNC, 0, 0, 0, 0, 0],
          [0, numuCC, 0, 0, 0, 0],
          [0, numuNC, 0, 0, 0, 0],
          [0, 0, nutauCC, 0, 0, 0],
          [0, 0, nutauNC, 0, 0, 0]
          [0, 0, 0, nuebarCC, 0, 0],
          [0, 0, 0, nuebarNC, 0, 0],
          [0, 0, 0, 0, numubarCC, 0],
          [0, 0, 0, 0, numubarNC, 0],
          [0, 0, 0, 0, 0, nutaubarCC],
          [0, 0, 0, 0, 0, nutaubarNC]]
       NOTE: the software will NOT verify that the matrix is in this
       form.

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
        else:
            raise ValueError("Bad format for data")

    @staticmethod
    def _getBlockMatrixForm(data):
        data = np.asarray(data)
        shape = data.shape
        if shape == (12,):
            result = np.zeros((12, 6), dtype=data.dtype)
            result[0, 0] = data[0]
            result[1, 0] = data[1]
            result[2, 1] = data[2]
            result[3, 1] = data[3]
            result[4, 2] = data[4]
            result[5, 2] = data[5]
            result[6, 3] = data[6]
            result[7, 3] = data[7]
            result[8, 4] = data[8]
            result[9, 4] = data[9]
            result[10, 5] = data[10]
            result[11, 5] = data[11]
            return result
        else:
            raise ValueError("Incorrect shape " + str(shape))

    def linearize(self):
        """
        Convert the matrix into a column vector.

        Reasonable and meaningful since the matrix is composed of block
        matrices, each of which is diagonal. The order is the same as
        the block column vector order specified in the constructor.

        """
        n = self.bins.n
        slices = zip(range(0, 12*n, n), np.repeat(range(0, 6*n, n), 2))
        blocks = [self[i:i+n, j:j+n] for i, j in slices]
        xsecs = np.concatenate(map(lambda x: x.diagonal(), blocks))
        return xsecs

    def zipWithEnergy(self):
        """
        Pair up each cross section bin with the mean energy of the bin.

        The return value is a 12*nbins-length array of (E, xsec) tuples,
        where the 12 n-bin groups correspond to the block column vector
        order of cross sections specified in the constructor.

        """
        return zip(np.tile(self.bins.centers, 12), self.linearize())

    def extract(self, name, withEnergy=False):
        thing = None
        smallslice = self.bins.index(name)
        if withEnergy:
            thing = self.zipWithEnergy()
            return np.asarray(thing[smallslice])
        else:
            thing = self.linearize()
            return np.asarray(thing[smallslice]).reshape(self.bins.n)


class DetectorResponse(SimulationComponent):
    """
    Detector Response matrix.

    The true energy increases along a row (i.e. the second index
    gives the true energy). The reconstructed energy increases down a
    column (i.e. the first index gives the reconstructed energy). The
    normalization should be that a true particle ends up somewhere
    (unitarity), so that the sum down a column is 1. I.e., sum(obj[:,i])
    == 1 for all i. This normalization is enforced in the constructor.

    The detector response matrix converts a true spectrum into a
    reconstructed spectrum. Its input should be broken down into
    interactions by flavor and channel (e.g. nueCC, nutauNC). Its output
    will be broken down either in the same way, or by reconstructed
    channel (namely nueCC-like, numuCC-like, NC-like), depending on the
    form of the matrix (see next paragraph).

    The matrix should be supplied in one of the following forms:
     - More precise: if the detector response, including event channel
       ID, is to be used all together, use a matrix:

       [[eCC->eCC-like, eNC->eCC-like, ..., tauNC->eCC-like,
       ebarCC->eCC-like, ebarNC->eCC-like, ..., taubarNC->eCC-like],
        [eCC->muCC-like, ..., taubarNC->muCC-like],
        [eCC->NC-like, ..., taubarNC->NC-like]]

       Output is a reconstructed spectrum (eCC-like, muCC-like,
       NC-like).

     - Less precise: if the energy response and the event
       classification are to be used separately (approximately true),
       use a column vector, which will preserve the event channel
       information.

       [eCC, eNC, muCC, muNC, tauCC, tauNC, ebarCC, ..., taubarNC]

       Output is still a "true" spectrum format (eCC, eNC, ..., taubarNC).
       The event classification must be performed later using the
       Efficiency object.

    """
    nextFormat = Spectrum

    def __new__(cls, arg):
        obj = SimulationComponent.__new__(cls, arg)
        obj.bins = cls.defaultBinning
        # Ignore divide by 0 errors and save the old error system
        old_error_state = np.seterr(invalid='ignore')
        # Normalize
        obj = np.divide(obj, np.sum(obj, axis=0))
        obj = np.nan_to_num(obj)
        # Reset numpy error system
        np.seterr(**old_error_state)
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
        if data.ndim != 2:
            raise ValueError("Input is not a 2-D matrix")
        else:
            return data

    @staticmethod
    def _getBlockMatrixForm(data):
        data = np.asarray(data)
        shape = data.shape
        if shape == (3, 12):  # full DRM
            return data
        elif shape == (12,):  # factored DRM + channel ID
            # Create a block-diagonal matrix
            return np.diag(data)
        else:
            raise ValueError("Incorrect shape " + str(shape))

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
        raise NotImplementedError("This method is not yet implemented. Sorry.")
        energylist = np.tile(self.bins.centers, 3)
        if form == 'matrix':
            energymatrix = np.array([[(e1, e2) for e2 in energylist]
                                     for e1 in energylist])
            return np.dstack((energymatrix, self))
        if form == 'list':
            energymatrix = np.array([(e1, e2) for e2 in energylist
                                     for e1 in energylist])
            drm = self.flatten()
            return zip(energymatrix, drm)
        raise ValueError("Did not recognize form " + str(form))

    def extract(self, name, withEnergy=False):
        thing = None
        n = self.bins.n
        if withEnergy:
            thing = self.zipWithEnergy()
        else:
            thing = self
        # Split the given name into two parts, the "from" and the "to."
        colname, rowname = name.split('2')
        colslice = self.bins.index(colname)
        rowslice = self.bins.index(rowname)
        return np.asarray(thing[rowslice, colslice])


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

     [[eCC->eCC-like, eNC->eCC-like, ..., taubarNC->eCC-like],
      [eCC->muCC-like, ..., taubarNC->muCC-like],
      [eCC->NC-like, ..., taubarNC->NC-like]]

    If the files supplied in "block" form are column vectors, they will
    be interpreted as diagonal matrices to satisfy the requirement in
    the following paragraph.

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
        Ensure the data is a matrix of appropriate dimensions and return
        it.

        """
        data = np.asarray(data)
        if data.ndim == 2:
            return data
        elif data.ndim == 1:
            return np.diag(data)
        else:
            raise ValueError("Incorrect ndim ", data.ndim, "!= 2")

    @staticmethod
    def _getBlockMatrixForm(data):
        data = np.asarray(data)
        shape = data.shape
        if shape == (3, 12):
            return data
        else:
            raise ValueError("Incorrect shape " + str(shape))

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
            energymatrix = np.array([(e1, e2) for e2 in energylist
                                     for e1 in energylist])
            efficiencies = self.flatten()
            return zip(energymatrix, drm)
        raise ValueError("Did not recognize form " + str(form))

    def extract(self, name, withEnergy=False):
        thing = None
        n = self.bins.n
        if withEnergy:
            thing = self.zipWithEnergy()
        else:
            thing = self
        # Split the given name into two parts, the "from" and the "to."
        colname, rowname = name.split('2')
        colslice = self.bins.index(colname)
        rowslice = self.bins.index(rowname)
        return np.asarray(thing[rowslice, colslice])
