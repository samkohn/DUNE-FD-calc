import sys
sys.path.append('/Users/skohn/Documents/DUNE/configs/scripts')
from dunesim import *
import matplotlib.pyplot as plt
import argparse

def plot(nuespecs, ratio, plotChiSquare):
    nominalspec = nuespecs[10]

    plotspeckey = {
            0: r"$\theta_{23}=40^{\circ}$",
            11: r"$\delta_{CP}=0.15\pi$",
            12: r"$\delta_{CP}=\pi/2$",
            13: r"$\delta_{CP}=-\pi/2$",
            14: r"$\delta_{CP}=\pi$",
            15: r"IO",
            20: r"$\theta_{23}=50^{\circ}$",
        }

    specstoplot = nuespecs[np.asarray(plotspeckey.keys()),:]

    bins = Spectrum.defaultBinning.centers
    plotkwargs = {
            'linewidth': 2.0,
        }
    labelkwargs = {
            'fontsize': 20,
        }
    fig = plt.figure(figsize=(16, 9))
    if ratio:
        plt.subplot(212)
        plt.plot(bins,((specstoplot-nominalspec) / nominalspec).T,
                **plotkwargs)
        plt.errorbar(bins, [0]*40, fmt='k--',yerr=np.sqrt(nominalspec)/nominalspec,
                **plotkwargs)
        plt.ylabel(r'$\Delta S/S_{nominal}$', **labelkwargs)
        plt.legend(plotspeckey.values() + ['nominal'])
        plt.xlabel('Energy [GeV]', **labelkwargs)
        plt.subplot(211)
    else:
        plt.xlabel('Energy [GeV]', **labelkwargs)

    plt.plot(bins, specstoplot.T, **plotkwargs)
    plt.errorbar(bins, nominalspec, fmt='k--', yerr=np.sqrt(nominalspec),
            **plotkwargs)
    plt.ylabel('Neutrinos per $0.25$ GeV', **labelkwargs)
    plt.title(r'Spectrum for 150 kt-MW-yr', **labelkwargs)
    if plotChiSquare:
        # Calculate the chi square for each non-nominal curve as an
        # attempted fit to the nominal curve.
        # Model = nominal curve
        # data = non-nominal curve
        # uncertainty = root(N) of non-nominal curve (data)
        chisquares = np.square(specstoplot-nominalspec) / specstoplot
        chisquares = np.sum(chisquares, axis=1)
        legendextras = [r", $\chi^{2}/NDF = " + str(cs)  + "/39$" for
                cs in chisquares] + ['']
    else:
        legendextras = [''] * (len(plotspeckey) + 1)
    legendnames = plotspeckey.values() + ['nominal']
    plt.legend([legendnames[i] + legendextras[i] for i in
        range(len(legendnames))])
    if outfilename == '':
        plt.show()
    else:
        plt.savefig(outfilename, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", action="store_true", help="plot dS/S")
    parser.add_argument("--factored-drm", action="store_true",
            help="use factored DRM/efficiency")
    parser.add_argument("--x2", action="store_true", help="include" +
            "chi square")
    parser.add_argument("--bar", action="store_true",
            help="antineutrino mode")
    parser.add_argument("-f", "--flavor", default=None,
            help="neutrino flavor whose spectrum will be plotted")
    parser.add_argument("-o", "--output", type=str, help="output location",
            default='', metavar="FILE")
    args = parser.parse_args()
    ratio = args.ratio
    factored = args.factored_drm
    chiSquare = args.x2
    neutrinomode = not args.bar
    outfilename = args.output
    # Some nontrivial rules for which flavor to plot:
    # Default to electron neutrino in neutrino mode, electron
    # antineutrino in antineutrino mode. Otherwise use the flavor
    # provided.
    if args.flavor is None:
        if neutrinomode:
            flavor = "e"
        else:
            flavor = "ebar"
    else:
        flavor = args.flavor


    # Compute the variation in the spectrum based on neutrino oscillation
    # parameters

    # CDR Vol 2 figure 3.5 uses 150 kt-MW-yr and 120 GeV protons
    # For 40 kt that's 3.75 MW-yr, at 1.2 MW that's 3.125yr, and
    # there's 1.1e21 POT/yr
    NUM_POT = 1.1e21 * 3.125 # POT/yr * yrs
    FLUX_FILE_BIN_WIDTH = 0.125  # GeV
    MY_BIN_WIDTH = 0.25  # GeV
    NUM_TARGET_ATOMS = 6e32 # 40kt argon
    XSEC_UNITS_CMSQ2MSQ = 1e-4

    setEnergyBins(np.linspace(0, 10, num=41, endpoint=True))

    folderbase = ('/Users/skohn/Documents/DUNE/configs/Fast-Monte-Carlo/' +
        'Oscillation-Parameters/local/set2/oscvectors_')
    foldersuffixes = [str(i) for i in range(1, 31)]
    folders = [folderbase + suffix + '/' for suffix in foldersuffixes]

    filenames = [['nue_nue40.csv', 'numu_nue40.csv', 'nutau_nue40.csv'],
            ['nue_numu40.csv', 'numu_numu40.csv', 'nutau_numu40.csv'],
            ['nue_nutau40.csv', 'numu_nutau40.csv', 'nutau_nutau40.csv'],
            ['nuebar_nuebar40.csv', 'numubar_nuebar40.csv', 'nutaubar_nuebar40.csv'],
            ['nuebar_numubar40.csv', 'numubar_numubar40.csv', 'nutaubar_numubar40.csv'],
            ['nuebar_nutaubar40.csv', 'numubar_nutaubar40.csv', 'nutaubar_nutaubar40.csv']]

    flux = defaultBeamFlux(neutrinomode) * NUM_POT * MY_BIN_WIDTH / FLUX_FILE_BIN_WIDTH
    xsec = defaultCrossSection() * NUM_TARGET_ATOMS * XSEC_UNITS_CMSQ2MSQ
    drm = defaultDetectorResponse(factored)

    nuespecs = np.empty((len(folders), flux.bins.n), flux.dtype)
    if factored:
            spectoextract = 'nu%sCC' % flavor
    else:
        spectoextract = '%s-like' % flavor
    for i, folder in enumerate(folders):
        oscfiles = [[folder + name for name in row] for row in filenames]
        oscprob = OscillationProbability(oscfiles)
        spectrum = (flux
                .evolve(oscprob)
                .evolve(xsec)
                .evolve(drm)
        )
        nuespecs[i, :] = spectrum.extract(spectoextract)

    plot(nuespecs, ratio, chiSquare)
