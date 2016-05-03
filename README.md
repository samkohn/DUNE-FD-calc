# DUNE-configs

This repository holds all the information I can find about experimental,
theoretical and software settings, parameters, and configurations
related to calculations of the DUNE experiment's sensitivity to CP
violation and the neutrino mass ordering.

It also contains some scripts to aid manipulation of the data using
python/numpy/matplotlib. There is also some legacy code in the form of
ROOT macros that performs some primitive calculations.

To generate new input .csv files, check out my Extractor repository,
[DUNE-FMC-Extract](https://github.com/samkohn/DUNE-FMC-Extract).

### Scripts
To run the scripts, first clone or download this repository. Then,
ensure that your system is configured correctly and set up the correct
python path information. Source the `setup` file with `source setup`.
This needs to be done every time a new shell session is started or the
code in the dunesim module will not work.

The easiest script to start with is the plotter.py script. You can run
`python plotter.py --help` to get a summary of the available commands.
Here are some examples:

```
# Plot variations in nue spectrum due to uncertainty in mixing angles
# Include integrated number of events (-N) and chi-square (--x2)
$ python plotter.py oscparam -N --x2

# Antineutrino mode (automatically plots nuebar spectrum)
$ python plotter.py oscparam --bar

# Plot numu CC-like spectrum
$ python plotter.py oscparam --flavor muCC

# Suppress signal and only look at background
$ python plotter.py oscparam --suppress numu2nue numubar2nuebar

# Suppress nues from all sources
$ python plotter.py oscparam --suppress nue2nue numu2nue nutau2nue

# Plot variations due to normalization uncertainty
$ python plotter.py norm -N
```

Each plot is labeled with the git commit description representing the
version used to make the plot. Also included is the specific python
command that was called.

The plotter module can be extended to include new analyses. Pull
requests are welcome!


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
