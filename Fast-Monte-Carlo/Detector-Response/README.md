Detector Response
=======

### For the DUNE Fast Monte Carlo

--------

#### Description
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
