Signal Efficiencies
==========

### For the DUNE Fast Monte Carlo

-----------

#### Description

The reconstruction procedure must determine whether an event is a CC
nu-e interaction (signal) or not (background). This directory contains
various probabilities (as a function of energy) for making the correct
determination.

Each file contains probabilities (as a function of reconstructed
energy) for a particular true interaction type to be identified as
a particular reconstructed interaction type. For example, the file
`nuflux_nueflux_nue_nueCC-like_trueCC40.csv` contains the probabilities
for a nue coming from beam nues (i.e. did not oscillate) in neutrino
mode that interacted via CC to be identified as nueCC-like. The binning
in reconstructed energy is from 0 to 10 GeV in 40 bins of 250 MeV.
