from dunesim import *
print """WARNING: This is just an example. Do not trust the
numerical results of this output. [enter to continue]\n\n\n"""
raw_input()
# Example run
setEnergyBins(np.arange(0, 10.25, 0.25))

flux = defaultBeamFlux()
# Get the total beam flux for 2*10^21 POT/year * 10 years (POT rate
# taken from the CDR [arXiv: 1601.05823] table 6-2)
# The units of the flux are #/GeV/POT/m^2. Our binning is 0.25 GeV so
# must also multiply by 0.25.
NUM_POT = 2e21 * 10
MY_BIN_WIDTH = 0.25
SOURCE_BIN_WIDTH = 0.125
# The new units are #/m^2
flux *= NUM_POT * MY_BIN_WIDTH / SOURCE_BIN_WIDTH
oscprob = defaultOscillationProbability()
# The cross section takes a units argument. The files use 1e-38 cm^2
# which is 1e-42 m^2
xsec = defaultCrossSection() * 1e-4
detectorresponse = defaultDetectorResponse()
# efficiency = \
# Efficiency('../Fast-Monte-Carlo/Efficiencies/nueCCsig_efficiency.csv')
oscflux = flux.evolve(oscprob)
print "oscillated flux\nnue flux\n", oscflux.extract('nue')
print "numu flux\n", oscflux.extract('numu')
print "nutau flux\n", oscflux.extract('nutau')
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
