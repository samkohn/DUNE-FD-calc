# DUNE-configs

This repository holds all the information I can find about experimental,
theoretical and software settings, parameters, and configurations
related to preparations for the DUNE experiment.

### Catalog
Each root-level directory contains all of the settings related to a
particular subsection of DUNE work. Subsections can be working groups,
individual people, detector subsystems, analysis tasks, and many other
logical groupings. The contents of the entire repository can be found in
the CATALOG.md file.

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
