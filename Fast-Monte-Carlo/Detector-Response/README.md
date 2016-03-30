Detector Response
=======

### For the DUNE Fast Monte Carlo

--------

#### Folder contents
This folder contains:

 * Histograms of the relationship between true and reconstructed
   neutrino energy
 * Detector response matrices which allow you to convert a true spectrum
   into a reconstructed spectrum
 * The specific configuration used to generate the detector response
   matrix, in the form of an XML configuration file that is used by the
   Fast Monte Carlo

#### Detector Response Matrix
The `.csv` files contain detector response matrices for the various
neutrino fluxes. The matrices convert between true and reconstructed
neutrino energy. Each file's name has three parts. In order, they
specify:

 * Whether the beam is in neutrino (nu) or antineutrino (anu) mode
 * The flavor of neutrino that leaves the beam (i.e. pre-oscillation)
 * The flavor of neutrino that is incident on the Far Detector (i.e.
   post-oscillation)

In order to use the file, you probably will want the following
information:

 * The energy range is 0 to 10 GeV, in 40 steps of 250 MeV
 * Each matrix was generated from a set of approximately 1e6 neutrinos
   in the Fast Monte Carlo
 * As you move along a row, the true energy increases
 * As you move down a column, the reconstructed energy increases
 * To go from true to reconstructed, construct a column vector of the
   true spectrum using the same 40 bins of 250 MeV. Multiply the column
   vector by the matrix to get the reconstructed energy in those same 40
   bins of 250 MeV.

Note: The matrix should be normalized so that probability is conserved. This
means each column should be normalized to 1. If the DRMs are to be
stacked together to create a large DRM for all neutrino types, the
normalization should happen *after* the large matrix is assembled.

#### Algorithm Description
The basic detector response [^1] is the energy/momentum/angle smearing, and
detection thresholds for final-state particles. Particles that produce
tracks rather than EM showers have two sets of reconstruction behavior,
one for contained tracks that appear the range out, and a second for
exiting tracks and/or particles that appear to shower. Particle energy
smearing for EM shower producing particles is a determined as a function
of energy, while tracking particles are smeared as a function of
momentum, and whether the particle is contained in the active detector
and the 'fate' of the particle.

[^1]: Adapted from the FastMC [wiki](https://cdcvs.fnal.gov/redmine/projects/fast_mc/wiki/Physics_and_Detector_Inputs_and_Assumptions#The-Detector-simulation "").
