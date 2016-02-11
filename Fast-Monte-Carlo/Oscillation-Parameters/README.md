Oscillation Parameters
==========

### For the DUNE Fast Monte Carlo

-----------

#### Description

The oscillation parameters used for the Fast Monte Carlo are stored in
the XML file. For different run configurations, different reported
values of the oscillation parameters are used. Most of the
configurations use the parameters reported in the "default" section, but
the GENIE 2.10.0 version uses a different, newer set of parameters.

#### Probability files

The .csv files contain pre-calculated probabilities for all possible
neutrino oscillations. The oscillation parameters used are from Nu-Fit's
2014 results (JHEP 11 (2014) 052 [arXiv:1409.5439]) using the normal
hierarchy parameters, delta-cp = 0, a baseline of 1300km, and earth
density a constant 2.7 g/cm^3. The oscillation calculator is the Prob3++
Barger Propagator.

As usual, the bins are from 0 to 10 GeV, and the "40" at the end of each
file indicates 40 equal bins (250 MeV each). The precise value of the
energy used for each bin is the midpoint.
