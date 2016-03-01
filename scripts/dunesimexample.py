from dunesim import *
print """WARNING: This is just an example. Do not trust the
numerical results of this output. [enter to continue]\n\n\n"""
raw_input()
# Example run
setEnergyBins(np.arange(0, 10.25, 0.25))

# Set up block matrices for the flux, oscillation probabilities, and
# cross sections. Specifications for the formats are given in the class
# docstrings.
pre = '../Fast-Monte-Carlo/Flux-Configuration/numode_'
fluxfiles = map(lambda x: pre + x, ['nue_flux40.csv', 'numu_flux40.csv',
    'nutau_flux40.csv'])
pre = '../Fast-Monte-Carlo/Oscillation-Parameters/nu'
oscfiles = [['e_nue40.csv', 'mu_nue40.csv', 'tau_nue40.csv'],
         ['e_numu40.csv', 'mu_numu40.csv', 'tau_numu40.csv'],
         ['e_nutau40.csv', 'mu_nutau40.csv', 'tau_nutau40.csv']]
oscfiles = [[pre + name for name in row] for row in oscfiles]
pre = '../Fast-Monte-Carlo/Cross-Sections/nu_'
xsecfiles = map(lambda x: pre + x,
    ['e_Ar40__tot_cc40.csv', 'e_Ar40__tot_nc40.csv',
     'mu_Ar40__tot_cc40.csv', 'mu_Ar40__tot_nc40.csv',
     'tau_Ar40__tot_cc40.csv', 'tau_Ar40__tot_nc40.csv'])
pre = '../Fast-Monte-Carlo/Detector-Response/nuflux_numuflux_nu'
drmfiles = ['e_trueCC40.csv', 'e_trueNC40.csv',
            'mu_trueCC40.csv', 'mu_trueNC40.csv',
            'tau_trueCC40.csv', 'tau_trueNC40.csv']
drmfiles = [pre + name for name in drmfiles]
flux = BeamFlux(fluxfiles)
oscprob = OscillationProbability(oscfiles)
xsec = CrossSection(xsecfiles)
detectorresponse = \
DetectorResponse(drmfiles)
efficiency = \
Efficiency('../Fast-Monte-Carlo/Efficiencies/nueCCsig_efficiency.csv')
oscflux = flux.evolve(oscprob)
print "oscillated flux\nnue flux\n", oscflux.extract('nue flux')
print "numu flux\n", oscflux.extract('numu flux')
print "nutau flux\n", oscflux.extract('nutau flux')
print "\n\n\n"
# Get the total beam flux for 10^20 POT
# The units of the flux are #/GeV/POT/m^2. Our binning is 0.25 GeV so
# must also multiply by 0.25. The cross-sectional area of the 4 far
# detectors combined is approximately 4 * 10m * 10m = 400m^2.
NUM_POT = 1e20
BIN_WIDTH = 0.25
AREA = 400
NUM_AR_ATOMS = 6e30
flux *= NUM_POT * BIN_WIDTH * AREA
detectorspec = (flux
        .evolve(oscprob)
        .evolve(xsec))
detectorspec *= NUM_AR_ATOMS
print "True spectrum of CC nue events at detector"
print detectorspec.extract('nueCC')
signalspec = detectorspec.evolve(detectorresponse)
print "Python type of signal spectrum = ", type(signalspec)
print "nue CC spectrum = "
print signalspec.extract('nueCC')

print "integrated true spectrum =", sum(detectorspec.extract('nueCC'))
print "integrated reco spectrum =", sum(signalspec.extract('nueCC'))


print "\n\n"
