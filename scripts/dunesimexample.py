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
# Get the total beam flux for 2*10^21 POT/year * 10 years (POT rate
# taken from the CDR [arXiv: 1601.05823] table 6-2)
# The units of the flux are #/GeV/POT/m^2. Our binning is 0.25 GeV so
# must also multiply by 0.25.
NUM_POT = 2e21 * 10
BIN_WIDTH = 0.25
# The new units are #/m^2
flux *= NUM_POT * BIN_WIDTH
oscprob = OscillationProbability(oscfiles)
# The cross section takes a units argument. The files use 1e-38 cm^2
# which is 1e-42 m^2
xsec = CrossSection(xsecfiles, units=1e-42)
detectorresponse = \
DetectorResponse(drmfiles)
efficiency = \
Efficiency('../Fast-Monte-Carlo/Efficiencies/nueCCsig_efficiency.csv')
oscflux = flux.evolve(oscprob)
print "oscillated flux\nnue flux\n", oscflux.extract('nue flux')
print "numu flux\n", oscflux.extract('numu flux')
print "nutau flux\n", oscflux.extract('nutau flux')
print "\n\n\n"
detectorspec = (flux
        .evolve(oscprob)
        .evolve(xsec))
NUM_AR_ATOMS = 6e32 # 40kt * 6e23/40g * 1000g/kg * 1000kg/t * 1000t/kt
detectorspec *= NUM_AR_ATOMS
print "True spectrum of CC nue events at detector"
print detectorspec.extract('nueCC')
signalspec = detectorspec.evolve(detectorresponse)
print "Python type of signal spectrum = ", type(signalspec)
print "nue CC spectrum = "
print signalspec.extract('nueCC', withEnergy=True)

print "integrated true spectrum =", sum(detectorspec.extract('nueCC'))
print "integrated reco spectrum =", sum(signalspec.extract('nueCC'))


print "\n\n"
