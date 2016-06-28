from dunesim import *
import sigmas
import numpy as np
import matplotlib.pyplot as plt

bins = Spectrum.defaultBinning.centers
quantities = {
        'flux': sigmas.flux,
        'oscillation parameters': sigmas.oscprob,
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


defaultspectrum = (sigmas.flux['default']
        .evolve(sigmas.oscprob['default'])
        .evolve(sigmas.xsec['default'])
        .evolve(sigmas.drm['default'])
        .evolve(sigmas.eff['default']))

fig = plt.figure(1)
for plotnum, (name, quantity_to_adjust) in enumerate(quantities.iteritems()):
    ax = plt.subplot(2, 3, plotnum+1)
    if quantity_to_adjust is sigmas.flux:
        spectrum_plus = sigmas.flux['+1sigma']
        spectrum_minus = sigmas.flux['-1sigma']
    else:
        spectrum_plus = sigmas.flux['default']
        spectrum_minus = sigmas.flux['default']
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
    ax.set_title(name)
plt.show()
