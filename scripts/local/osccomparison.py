from dunesim import *
import matplotlib.pyplot as plt
import argparse

def plot(nuespecs, plotspeckey, nominalspec, bar, ratio, specrange, plotChiSquare, plotN, hardcodeaxes, outfilename):

    specstoplot = nuespecs[np.asarray(plotspeckey.keys()),specrange]

    bins = Spectrum.defaultBinning.centers[specrange]
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

    lines = plt.plot(bins, specstoplot.T, **plotkwargs)
    plt.errorbar(bins, nominalspec, fmt='k--', yerr=np.sqrt(nominalspec),
            **plotkwargs)
    plt.ylabel(('Antin' if bar else 'N') + 'eutrinos per $0.25$ GeV', **labelkwargs)
    if hardcodeaxes == 'neutrinomode':
        plt.ylim([0, 120])
    elif hardcodeaxes == 'antineutrinomode':
        plt.ylim([0,35])
    elif not hardcodeaxes:
        pass # hardcodeaxes is just False
    else:
        raise ValueError("Bad hardcodeaxes", hardcodeaxes)
    plt.title(('Antin' if bar else 'N') + r'eutrino spectrum for 150 kt-MW-yr', **labelkwargs)
    legendextras = [''] * (len(specstoplot) + 1)
    if plotN:
        sums = [sum(spec) for spec in specstoplot]
        legendextras = [r", $N=%.0f$" % N for N in sums +
                [sum(nominalspec)]]
    if plotChiSquare:
        # Calculate the chi square for each non-nominal curve as an
        # attempted fit to the nominal curve.
        # Model = nominal curve
        # data = non-nominal curve
        # uncertainty = root(N) of non-nominal curve (data)
        chisquares = np.square(specstoplot-nominalspec) / specstoplot
        chisquares = np.sum(chisquares, axis=1)
        lastentry = legendextras[-1]
        legendextras = [le + r", $\chi^{2}/NDF = %.1f/39$" % cs
                for (le, cs) in zip(legendextras[:-1], chisquares)]
        legendextras += [lastentry]
    else:
        pass
    legendnames = plotspeckey.values() + ['nominal']
    plt.legend([legendnames[i] + legendextras[i] for i in
        range(len(legendnames))])
    plt.setp(lines, linestyle='steps-mid')
    if outfilename == '':
        plt.show()
    else:
        plt.savefig(outfilename, bbox_inches='tight')

def varyBackgroundType(CLargs, physicsparams):
    flux = defaultBeamFlux(not CLargs.bar) * physicsparams['fluxweight']
    originalflux = flux.copy()
    osc = defaultOscillationProbability()
    originalosc = osc.copy()
    xsec = defaultCrossSection() * physicsparams['xsecweight']
    originalxsec = xsec.copy()
    drm = defaultDetectorResponse(CLargs.factored_drm)
    originaldrm = drm.copy()
    if CLargs.factored_drm:
        spectoextract = 'nu%sCC' % CLargs.flavor
        efficiency = defaultEfficiency(not CLargs.bar)
    else:
        spectoextract = 'nu%s-like' % CLargs.flavor
        efficiency = None

    # Different types of background:
    # - Beam nue and nuebar CC: set all other fluxes to 0 and only preserve
    #   the nue->nue and nuebar->nuebar oscillations
    # - NC: set all of the CC cross sections to 0
    # - nutau and nutaubar CC: zero all cross sections except for those
    # - numu and numubar CC: zero all cross sections except for those
    sub = flux.bins.index
    def zero(obj, *flavors):
        slices = map(sub, flavors)
        obj[slices] = 0
    def sig_nue():
        zero(flux, 'nue')
        zero(flux, 'nutau')
        zero(flux, 'nuebar')
        zero(flux, 'numubar')
        zero(flux, 'nutaubar')
        zero(osc, 'numu', 'numu')
        zero(osc, 'nutau', 'numu')
    def sig_nuebar():
        zero(flux, 'nue')
        zero(flux, 'numu')
        zero(flux, 'nutau')
        zero(flux, 'nuebar')
        zero(flux, 'nutaubar')
        zero(osc, 'numubar', 'numubar')
        zero(osc, 'nutaubar', 'numubar')
    def bg_beamnue():
        zero(flux, 'numu')
        zero(flux, 'nutau')
        zero(flux, 'numubar')
        zero(flux, 'nutaubar')
        # Order reversed from oscillation (nue->numu => [numu, nue])
        zero(osc, 'numu', 'nue')
        zero(osc, 'nutau', 'nue')
        zero(osc, 'numubar', 'nuebar')
        zero(osc, 'nutaubar', 'nuebar')
    def bg_nc():
        zero(xsec, 'nueCC')
        zero(xsec, 'numuCC')
        zero(xsec, 'nutauCC')
        zero(xsec, 'nuebarCC')
        zero(xsec, 'numubarCC')
        zero(xsec, 'nutaubarCC')
    def bg_tauCC():
        zero(xsec, 'nueCC')
        zero(xsec, 'nueNC')
        zero(xsec, 'numuCC')
        zero(xsec, 'numuNC')
        zero(xsec, 'nutauNC')
        zero(xsec, 'nuebarCC')
        zero(xsec, 'nuebarNC')
        zero(xsec, 'numubarCC')
        zero(xsec, 'numubarNC')
        zero(xsec, 'nutaubarNC')
    def bg_muCC():
        zero(xsec, 'nueCC')
        zero(xsec, 'nueNC')
        zero(xsec, 'numuNC')
        zero(xsec, 'nutauCC')
        zero(xsec, 'nutauNC')
        zero(xsec, 'nuebarCC')
        zero(xsec, 'nuebarNC')
        zero(xsec, 'numubarNC')
        zero(xsec, 'nutaubarCC')
        zero(xsec, 'nutaubarNC')
    variations = [lambda:None, sig_nue, sig_nuebar, bg_beamnue, bg_nc, bg_tauCC, bg_muCC]
    nuespecs = np.empty((len(variations), flux.bins.n), flux.dtype)
    for i, variation in enumerate(variations):
        flux = originalflux.copy()
        osc = originalosc.copy()
        xsec = originalxsec.copy()
        drm = originaldrm.copy()
        variation()
        for badflavor in CLargs.suppress:
            col, row = badflavor.split('2')
            zero(osc, row, col)
        spectrum = (flux
                .evolve(osc)
                .evolve(xsec)
                .evolve(drm)
        )
        if CLargs.factored_drm:
            spectrum = spectrum.evolve(efficiency)
        else:
            pass
        nuespecs[i, :] = spectrum.extract(spectoextract)
    nominalspec = nuespecs[0, CLargs.binstoplot]
    plotspeckey = {
            1: r"signal $\nu_{e}$",
            2: r"signal $\bar{\nu}_{e}$",
            3: r"beam $\nu_{e}$ and $\bar{\nu}_{e}$",
            4: r"NC",
            5: r"$\nu_{\tau}$ and $\bar{\nu}_{\tau}$ CC",
            6: r"$\nu_{\mu}$ and $\bar{\nu}_{\mu}$ CC",
    }
    return (nuespecs, nominalspec, plotspeckey)

def varyFluxNormalizations(CLargs, physicsparams):
    flux = defaultBeamFlux(not CLargs.bar) * physicsparams['fluxweight']
    originalflux = flux.copy()
    osc = defaultOscillationProbability()
    xsec = defaultCrossSection() * physicsparams['xsecweight']
    drm = defaultDetectorResponse(CLargs.factored_drm)
    if CLargs.factored_drm:
        spectoextract = 'nu%sCC' % CLargs.flavor
        efficiency = defaultEfficiency(not CLargs.bar)
    else:
        spectoextract = 'nu%s-like' % CLargs.flavor
        efficiency = None
    variations = [
            (flux.bins.index('nue'), 1.0),  # no change
            (flux.bins.index('nue'), 1.1),
            (flux.bins.index('nue'), 0.9),
            (flux.bins.index('numu'), 1.1),
            (flux.bins.index('numu'), 0.9),
            (flux.bins.index('nuebar'), 1.1),
            (flux.bins.index('nuebar'), 0.9),
    ]
    nuespecs = np.empty((len(variations), flux.bins.n), flux.dtype)
    for i, (bins, change) in enumerate(variations):
        flux = originalflux.copy()
        flux[bins] *= change
        for badflavor in CLargs.suppress:
            col, row = badflavor.split('2')
            osc[osc.bins.index(row), osc.bins.index(col)] = 0
        spectrum = (flux
                .evolve(osc)
                .evolve(xsec)
                .evolve(drm)
        )
        if CLargs.factored_drm:
            spectrum = spectrum.evolve(efficiency)
        else:
            pass
        nuespecs[i, :] = spectrum.extract(spectoextract)

    nominalspec = nuespecs[0, CLargs.binstoplot]

    plotspeckey = {
            1: r"beam $\nu_{e} \uparrow 10\%$",
            2: r"beam $\nu_{e} \downarrow 10\%$",
            3: r"beam $\nu_{\mu} \uparrow 10\%$",
            4: r"beam $\nu_{\mu} \downarrow 10\%$",
            5: r"beam $\bar{\nu}_{e} \uparrow 10\%$",
            6: r"beam $\bar{\nu}_{e} \downarrow 10\%$",
        }
    return (nuespecs, nominalspec, plotspeckey)

def varyOscillationParameters(CLargs, physicsparams):
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

    flux = defaultBeamFlux(not CLargs.bar) * physicsparams['fluxweight']
    xsec = defaultCrossSection() * physicsparams['xsecweight']
    drm = defaultDetectorResponse(CLargs.factored_drm)
    if CLargs.factored_drm:
        spectoextract = 'nu%sCC' % CLargs.flavor
        efficiency = defaultEfficiency(not CLargs.bar)
    else:
        spectoextract = 'nu%s-like' % CLargs.flavor
        efficiency = None

    nuespecs = np.empty((len(folders), flux.bins.n), flux.dtype)
    for i, folder in enumerate(folders):
        oscfiles = [[folder + name for name in row] for row in filenames]
        oscprob = OscillationProbability(oscfiles)
        for badflavor in CLargs.suppress:
            col, row = badflavor.split('2')
            oscprob[oscprob.bins.index(row), oscprob.bins.index(col)] = 0
        spectrum = (flux
                .evolve(oscprob)
                .evolve(xsec)
                .evolve(drm)
        )
        if CLargs.factored_drm:
            spectrum = spectrum.evolve(efficiency)
        else:
            pass
        nuespecs[i, :] = spectrum.extract(spectoextract)

    nominalspec = nuespecs[10, CLargs.binstoplot]

    plotspeckey = {
            0: r"$\theta_{23}=40^{\circ}$",
            11: r"$\delta_{CP}=0.15\pi$",
            12: r"$\delta_{CP}=\pi/2$",
            13: r"$\delta_{CP}=-\pi/2$",
            14: r"$\delta_{CP}=\pi$",
            15: r"IO",
            20: r"$\theta_{23}=50^{\circ}$",
        }
    return (nuespecs, nominalspec, plotspeckey)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", action="store_true", help="plot dS/S")
    parser.add_argument("dataset", type=str,
            choices=["oscparam", "norm", "bg"],
            help="the data set to plot: varying oscillation " +
            "parameters, normalization, or background source")
    parser.add_argument("--factored-drm", action="store_true",
            help="use factored DRM/efficiency")
    parser.add_argument("--x2", action="store_true", help="include " +
            "chi square")
    parser.add_argument("-N", "--total", action="store_true",
            help="include the integral total of each spectrum")
    parser.add_argument("--bar", action="store_true",
            help="antineutrino mode")
    parser.add_argument("-f", "--flavor", default=None,
            help="neutrino flavor whose spectrum will be plotted")
    parser.add_argument("--suppress", default=[], type=str, nargs='+',
            help="oscillations to ignore at the detector (e.g. to " +
            "investigate background)", metavar="nu_x2nu_y")
    parser.add_argument("-r", "--range", nargs=2, type=int,
            help="min and max bin numbers to plot", metavar=("MIN",
            "MAX"))
    parser.add_argument("--standard-axes", action="store_true",
            help="use hard-coded axis range")
    parser.add_argument("-o", "--output", type=str, help="output location",
            default='', metavar="FILE")
    args = parser.parse_args()
    ratio = args.ratio
    dataset = args.dataset
    factored = args.factored_drm
    chiSquare = args.x2
    plotN = args.total
    neutrinomode = not args.bar
    suppress = args.suppress
    binstoplot = args.range
    hardcodeaxes = args.standard_axes
    outfilename = args.output
    # Some nontrivial rules for which flavor to plot:
    # Default to electron neutrino in neutrino mode, electron
    # antineutrino in antineutrino mode. Otherwise use the flavor
    # provided.
    if args.flavor is None:
        if factored:
            if neutrinomode:
                args.flavor = "e"
            else:
                args.flavor = "ebar"
        else:
            args.flavor = "eCC"
    else:
        flavor = args.flavor

    # Some nontrivial rules for hardcodeaxes value:
    # If it's specified, then let it encode the neutrino mode. If it's
    # unspecified, leave it as False.
    if hardcodeaxes:
        if neutrinomode:
            hardcodeaxes = 'neutrinomode'
        else:
            hardcodeaxes = 'antineutrinomode'
    else:
        hardcodeaxes = False

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
    physicsparams = {
            'fluxweight': NUM_POT * MY_BIN_WIDTH/FLUX_FILE_BIN_WIDTH,
            'xsecweight': NUM_TARGET_ATOMS * XSEC_UNITS_CMSQ2MSQ,
    }


    setEnergyBins(np.linspace(0, 10, num=41, endpoint=True))
    if binstoplot is None:
        args.binstoplot = slice(0,
                SimulationComponent.defaultBinning.n)
    else:
        args.binstoplot = slice(binstoplot[0], binstoplot[1])

    if dataset == "oscparam":
        fntoplot = varyOscillationParameters
    elif dataset == "norm":
        fntoplot = varyFluxNormalizations
    elif dataset == "bg":
        fntoplot = varyBackgroundType
    nuespecs, nominalspec, plotspeckey = fntoplot(args,
            physicsparams)
    plot(nuespecs, plotspeckey, nominalspec, args.bar, ratio, args.binstoplot, chiSquare, plotN, hardcodeaxes, outfilename)

