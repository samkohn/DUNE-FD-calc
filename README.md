# DUNE-configs

This repository holds all the information I can find about experimental,
theoretical and software settings, parameters, and configurations
related to calculations of the DUNE experiment's sensitivity to CP
violation and the neutrino mass ordering.

It also contains some scripts to aid manipulation of the data using
python/numpy/matplotlib. There is also some legacy code in the form of
ROOT macros that performs some primitive calculations.

### Scripts
To run the scripts, first clone or download this repository. Then,
ensure that your system is configured correctly by executing the
`bootstrap` file with `./bootstrap`. This only needs to be done once. To
set up the correct python path information, source the `setup` file with
`source setup`. This needs to be done every time a new shell session is
started or the code in the dunesim module will not work.

### Physics and Detector Configurations
Each root-level directory contains all of the settings related to a
particular subsection of DUNE work. Subsections can be working groups,
individual people, detector subsystems, analysis tasks, and many other
logical groupings.

### Information contained in this repository
I have adopted a standard layout for all of the information. For each
configuration/setting, there is a folder in the repository which
contains a metadata file with resources related to the configuration
(e.g. contact person); textual description and explanation of the settings; a
plaintext file containing the configuration data; histograms and plots
of representative quantities related to the configuration; and at least
one data file containing the configuration in its "native" state. The
data file may be a ROOT file, an XML file, or something else. You may
need other software (e.g. LArSoft or GENIE) in order to read this
"native" file.
