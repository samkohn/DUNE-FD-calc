from dunesim import *
import sigmas
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ratio', action='store_true',
        help='plot dS/S')
args = parser.parse_args()

bins = Spectrum.defaultBinning.centers
quantities = {
        'flux': sigmas.flux,
        r'oscillation parameters ($\theta_{23}$)': sigmas.oscprob,
        'cross section': sigmas.xsec,
        'energy response/reconstruction': sigmas.drm,
        'interaction channel ID efficiency': sigmas.eff,
        r'$\delta_{CP}$': sigmas.dcp
}
quantitieslist = [
        sigmas.flux,
        sigmas.oscprob,
        sigmas.xsec,
        sigmas.drm,
        sigmas.eff,
        sigmas.dcp
]

pot_per_year = 1.1e21
years = 3.125
flux_file_bin_width=0.125
my_bin_width = 10.0/SimulationComponent.defaultBinning.n
num_targets = 6e32
xsec_units = 1e-4
physicsfactor = pot_per_year * years * my_bin_width/flux_file_bin_width
physicsfactor *= num_targets * xsec_units

defaultspectrum = physicsfactor * (sigmas.flux['default']
        .evolve(sigmas.oscprob['default'])
        .evolve(sigmas.xsec['default'])
        .evolve(sigmas.drm['default'])
        .evolve(sigmas.eff['default'])).extract('nue')
defaultspectrumerror = np.sqrt(defaultspectrum)

fig = plt.figure(1)
fig.suptitle(r"$\nu_{e}$ spectra for 3.125 years at $1.1\times 10^{2" +
        r"1}$ POT/yr for 40 kt FD ($\approx 150$ kt-MW-yr)", fontsize=20)
for plotnum, (name, quantity_to_adjust) in enumerate(quantities.iteritems()):
    ax = plt.subplot(2, 3, plotnum+1)
    if quantity_to_adjust is sigmas.flux:
        spectrum_plus = physicsfactor * sigmas.flux['+1sigma']
        spectrum_minus = physicsfactor * sigmas.flux['-1sigma']
    else:
        spectrum_plus = physicsfactor * sigmas.flux['default']
        spectrum_minus = physicsfactor * sigmas.flux['default']
    if quantity_to_adjust is sigmas.dcp:
        # Swap out the theta-23 variation for the delta-cp variation
        quantitieslist[1], quantitieslist[-1] = (quantitieslist[-1],
            quantitieslist[1])
    for nextquantity in quantitieslist[1:-1]:
        if quantity_to_adjust is nextquantity:
            spectrum_plus = spectrum_plus.evolve(
                nextquantity['+1sigma'])
            spectrum_minus = spectrum_minus.evolve(
                nextquantity['-1sigma'])
        else:
            spectrum_plus = spectrum_plus.evolve(
                nextquantity['default'])
            spectrum_minus = spectrum_minus.evolve(
                nextquantity['default'])
    if quantity_to_adjust is sigmas.dcp:
        # Swap back the theta-23 variation for the delta-cp variation
        quantitieslist[1], quantitieslist[-1] = (quantitieslist[-1],
            quantitieslist[1])
    if args.ratio:
        plusplot = (spectrum_plus.extract('nue') -
                defaultspectrum)/defaultspectrum
        minusplot = (spectrum_minus.extract('nue') -
                defaultspectrum)/defaultspectrum
        errorplot_up = defaultspectrumerror/defaultspectrum
        errorplot_low = -errorplot_up
        ax.set_ylim([-1, 1])
    else:
        im = ax.plot(bins, defaultspectrum, 'b')
        plusplot = spectrum_plus.extract('nue')
        minusplot = spectrum_minus.extract('nue')
        errorplot_up = defaultspectrum + defaultspectrumerror
        errorplot_low = defaultspectrum - defaultspectrumerror
        ax.set_ylim([0, 35])
    ax.set_xlim([0, 10])
    errorcolors = {
            'facecolor': 'b',
            'alpha': 0.2,
            'edgecolor': 'none'
    }
    errors_im = ax.fill_between(bins, errorplot_up, errorplot_low,
            **errorcolors)
    flux_plus_im = ax.plot(bins, plusplot, 'b--')
    flux_minus_im = ax.plot(bins, minusplot, 'b:')
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Neutrino events per 0.083 GeV")
    ax.set_title(name)
    if name == r'oscillation parameters ($\theta_{23}$)':
        legendlabels = ["nominal", r"$+3\sigma$", r"$-3\sigma$"]
    elif name == r'$\delta_{CP}$':
        legendlabels = [r'$\delta_{CP} = 0$', r'$\delta_{CP} = \pi/2$',
            r'$\delta_{CP} = -\pi/2$']
    else:
        legendlabels = ["Nominal", r"$+1\sigma$", r"$-1\sigma$"]
    legendlabels.append("stat. error")
    if args.ratio:
        legendlabels.pop(0)
    ax.legend(legendlabels)
plt.show()
