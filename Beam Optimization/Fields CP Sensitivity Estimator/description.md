CP Sensitivity Estimator
==========

### For Laura Fields's Beam Optimization

-----

#### Description

A set [^1] of eighty-four varied neutrino fluxes were processed through
the Fast MC, and the 75% CP sensitivity after 6 years of exposure was
evaluated for each variation. In each of these fluxes, one energy bin
of one neutrino flavor was increased from the default beam flux (g4lbne
v3r2p4b) by 10%. Six neutrino flavors (muon neutrinos in neutrino mode,
muon antineutrinos in neutrino mode, electron neutrinos in neutrino
mode, muon neutrinos in antineutrino mode, muon antineutrinos in
antineutrino mode and electron antineutrinos in antineutrino mode) and
fourteen energy bins, bounded by [0,0.5,1,2,3,4,5,6,7,8,9,10,15,20,120]
GeV, were considered. The process was then repeated, extracting CP
sensitivities with the default flux in a particular energy bin altered
to 0%, 50%, 90%, 110%, 200%, 500%, and 1000% of itself.


To estimate the CP sensitivity of a beam design, we first simulate the
beam and produce spectra of muon neutrino and antineutrino and electron
neutrino fluxes in neutrino mode and muon neutrino and antineutrino and
electron antineutrino fluxes in antineutrino mode. Each bin of neutrino
energy is then compared to the default flux, and the information in
Figures 2 - 2 is then interpolated to estimate the expect change in CP
sensitivity given a change in flux of one neutrino flavor in one energy
bin. The total sensitivity of the beam in question is then estimated as
the default beam sensitivity plus the sum over all neutrino fluxes and
all energy bins of the individual changes in CP sensitivity.

[^1]: Adapted from DUNE-doc-56-v3
