"""
This module generates +/- 1-sigma adjustments to the cross section,
flux, DRM, and efficiency given by the default* methods in dunesim.

"""
from dunesim import *

setEnergyBins(np.linspace(0, 10, 121))

flux0 = defaultBeamFlux(neutrinomode=True)
anuflux0 = defaultBeamFlux(neutrinomode=False)
oscprob0 = defaultOscillationProbability()
xsec0 = defaultCrossSection()
drm0 = defaultDetectorResponse()
eff0 = defaultEfficiency()

flux = {'default': flux0}
anuflux = {'default': anuflux0}
oscprob = {'default': oscprob0}
xsec = {'default': xsec0}
drm = {'default': drm0}
eff = {'default': eff0}

flux['+1sigma'] = flux['default'] * 1.1
flux['-1sigma'] = flux['default'] * 0.9

anuflux['+1sigma'] = anuflux['default'] * 1.3
anuflux['-1sigma'] = anuflux['default'] * 0.7

xsec['+1sigma'] = xsec['default'].copy()
xsec['-1sigma'] = xsec['default'].copy()
xsec['+1sigma'] *= 1.3
xsec['-1sigma'] *= 0.7
tmp = xsec['+1sigma'].extract('nueCC')
tmp *= 1.3
tmp = xsec['-1sigma'].extract('nueCC')
tmp *= 0.7
# xsec['+1sigma'].extract('nueNC') *= 1.5
# xsec['-1sigma'].extract('nueNC') *= 0.5

oscprob['+1sigma'] = oscprob['default'].copy()
oscprob['-1sigma'] = oscprob['default'].copy()

# Define a "focusing" function that reduces the spread
# I choose a Gaussian
def focus(n, center, width, low=None, up=None):
    """
    Specify the center and width in real coordinates (energy).

    If low and up are provided, they are used as the range of energy
    being considered. If they are not provided, then the default binning
    from the dunesim module is used.

    """
    if low is None:
        low = SimulationComponent.defaultBinning.start
    if up is None:
        up = SimulationComponent.defaultBinning.end
    centerBin = int((center-low)/(up-low) * n)
    widthInBins = int(width/(up-low) * n)
    lowBin = -centerBin/widthInBins
    upBin = (n-centerBin)/widthInBins
    x = np.linspace(lowBin, upBin, n)
    exponent = -np.square(x)/2
    return np.exp(exponent)


drm['+1sigma'] = drm['default'].copy()
drm['-1sigma'] = drm['default'].copy()

eff['+1sigma'] = eff['default'].copy()
eff['-1sigma'] = eff['default'].copy()
