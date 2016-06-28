from dunesim import *
import sigmas
import numpy as np
import matplotlib.pyplot as plt

bins = Spectrum.defaultBinning.centers
quantities = {
        'flux': sigmas.flux,
        r'oscillation parameters ($\pm3\sigma$ on $\theta_{23}$)': sigmas.oscprob,
        'cross section': sigmas.xsec,
        'energy response/reconstruction': sigmas.drm,
        'interaction channel ID efficiency': sigmas.eff
}
quantitieslist = [
        sigmas.flux,
        sigmas.oscprob,
        sigmas.xsec,
        sigmas.drm,
        sigmas.eff
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
        .evolve(sigmas.eff['default']))

fig = plt.figure(1)
for plotnum, (name, quantity_to_adjust) in enumerate(quantities.iteritems()):
    ax = plt.subplot(2, 3, plotnum+1)
    if quantity_to_adjust is sigmas.flux:
        spectrum_plus = physicsfactor * sigmas.flux['+1sigma']
        spectrum_minus = physicsfactor * sigmas.flux['-1sigma']
    else:
        spectrum_plus = physicsfactor * sigmas.flux['default']
        spectrum_minus = physicsfactor * sigmas.flux['default']
    for nextquantity in quantitieslist[1:]:
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
    im = ax.plot(bins, defaultspectrum.extract('nue'))
    flux_plus_im = ax.plot(bins, spectrum_plus.extract('nue'),
            'b--')
    flux_minus_im = ax.plot(bins, spectrum_minus.extract('nue'),
            'b:')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 35])
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Neutrino events per 0.083 GeV")
    ax.set_title(name)
    ax.legend(["Nominal", r"$+1\sigma$", r"$-1\sigma$"])
plt.show()
