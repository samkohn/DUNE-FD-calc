"""
This module generates +/- 1-sigma adjustments to the cross section,
flux, DRM, and efficiency given by the default* methods in dunesim.

"""
from dunesim import *
import math
import os

__all__=['dcp', 'flux', 'anuflux', 'oscprob', 'xsec', 'drm', 'eff', 'smear', 'focus']

setEnergyBins(np.linspace(0, 10, 121))

dcp0 = defaultOscillationProbability()
flux0 = defaultBeamFlux(neutrinomode=True)
anuflux0 = defaultBeamFlux(neutrinomode=False)
oscprob0 = defaultOscillationProbability()
xsec0 = defaultCrossSection()
drm0 = defaultDetectorResponse()
eff0 = defaultEfficiency()

dcp = {'default': dcp0}
flux = {'default': flux0}
anuflux = {'default': anuflux0}
oscprob = {'default': oscprob0}
xsec = {'default': xsec0}
drm = {'default': drm0}
eff = {'default': eff0}

# Fetch variations in delta-cp from the oscvectorsets
oscparamsdir = os.path.join(os.environ['DUNECONFIGSROOT'],
        'Fast-Monte-Carlo/Oscillation-Parameters/Parameter-Sets/')
dcp['+1sigma'] = defaultOscillationProbability(loc=os.path.join(
        oscparamsdir, 'oscvectors_13'))
dcp['-1sigma'] = defaultOscillationProbability(loc=os.path.join(
        oscparamsdir, 'oscvectors_14'))

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

# Fetch the oscprob changes from the oscvectorsets
oscprob['+1sigma'] = defaultOscillationProbability(loc=os.path.join(
        oscparamsdir, 'oscvectors_21'))
oscprob['-1sigma'] = defaultOscillationProbability(loc=os.path.join(
        oscparamsdir, 'oscvectors_1'))

# Define a "focusing" function that reduces the spread
# I choose a Gaussian
def focus(peak, width, n=None, low=None, up=None):
    """
    Create an array whose entries are proportional to the Gaussian
    distribution.

    Specify the peak and width in real coordinates (energy), not in
    bin indices.

    If low and up are provided, they are used as the range of energy
    being considered. If they are not provided, then the default binning
    from the dunesim module is used.

    """
    if low is None:
        low = SimulationComponent.defaultBinning.start
    if up is None:
        up = SimulationComponent.defaultBinning.end
    if n is None:
        n = SimulationComponent.defaultBinning.n
    peakIndex = math.floor((peak-low)/(up-low) * n)
    widthInBins = math.floor(width/(up-low) * n)
    lowBinValue = -peakIndex/widthInBins
    upBinValue = (n-peakIndex)/widthInBins
    x = np.linspace(lowBinValue, upBinValue, n)
    exponent = -np.square(x)/2
    return np.exp(exponent)

# For the smearer, I want to divide the center by a larger number, and
# then decrease the reduction factor down to 1 as we move away from the
# center. The function that accomplishes this is division by (1 +
# gaussian), or multiplication by 1/(1 + gaussian).
# I will use the focusing function as a generic "gaussian" function.
def smear(normalization, peak, width, n=None, low=None, up=None):
    """
    Create an array whose entries are 1/(1 + prop-to-gaussian).

    The normalization gives the value of the peak of the gaussian, so
    that min(smear(x, ...)) == 1/(1 + x). A suggested value is 1./3.

    Specify the peak and width in real coordinates (energy), not in
    bin indices.

    If low and up are provided, they are used as the range of energy
    being considered. If they are not provided, then the default binning
    from the dunesim module is used.

    """
    focuser = focus(peak, width, n, low, up)
    return 1/(1 + normalization * focuser)

blockstofocus = ['nueCC2nueCC',
        'numuCC2numuCC',
        'nuebarCC2nuebarCC',
        'numubarCC2numubarCC']

drm['+1sigma'] = drm['default'].copy()
drm['-1sigma'] = drm['default'].copy()

# tighten up the energy response centered on the true energy
focus_width = 0.5 # width in GeV
bins = SimulationComponent.defaultBinning.centers
for blockname in blockstofocus:
    focus_block = drm['+1sigma'].extract(blockname)
    focuser = np.zeros_like(focus_block)
    for i, energy in enumerate(bins):
        focuser[:, i] = focus(energy, focus_width)
    focus_block *= focuser

# worsen the -1 sigma case by allowing for the possibility of missing
# energy. approximate this by adding the nueNC energy response
for blockname in blockstofocus:
    smear_block = drm['-1sigma'].extract(blockname)
    smear_block += drm['default'].extract('nueNC2nueNC') * 0.1

drm['+1sigma'].normalize()
drm['-1sigma'].normalize()

# Increase and decrease the selection efficiency for signal and
# background channels
eff['+1sigma'] = eff['default'].copy()
eff['-1sigma'] = eff['default'].copy()
correct_ids = [
        'nueCC2nueCC-like',
        'numuCC2numuCC-like',
        'nutauCC2NC-like',
        'nueNC2NC-like',
        'numuNC2NC-like',
        'nutauNC2NC-like',
        'nuebarCC2nueCC-like',
        'numubarCC2numuCC-like',
        'nutaubarCC2NC-like',
        'nuebarNC2NC-like',
        'numubarNC2NC-like',
        'nutaubarNC2NC-like',
    ]
# For better ID, assume the false negative rate decreases by half for
# each true event type. The actual change will not be truly half because
# I do not correctly normalize.
for selection in correct_ids:
    block = eff['+1sigma'].extract(selection)
    data = block.diagonal()
    false_negative = 1 - data
    correction = false_negative / 2
    np.fill_diagonal(block, data + correction)
eff['+1sigma'].normalize()
# For worse ID, assume the following sources of background are
# amplified. I chose these backgrounds based on which forecast
# improvements seem hardest to implement.
backgrounds = [
        'nueNC2nueCC-like',
        'numuNC2nueCC-like',
        'nutauCC2nueCC-like',
        'nutauNC2nueCC-like',
        'nuebarNC2nueCC-like',
        'numubarNC2nueCC-like',
        'nutaubarCC2nueCC-like',
        'nutaubarNC2nueCC-like',
    ]
for background in backgrounds:
    block = eff['-1sigma'].extract(background)
    data = block.diagonal()
    correction = data
    np.fill_diagonal(block, data + correction)
eff['-1sigma'].normalize()
