# DUNE-FD-calc

This repository contains human- and machine-readable data related to the
DUNE neutrino oscillation measurement, including detector response
information, cross sections, and beam flux simulation results.

It also contains a python module for performing calculations with this
data and a flexible yet powerful plotting program for performing common
calculations and comparisons. *The only dependencies for this software
are numpy and matplotlib*. If you are not sure if you have these
python packages installed, the provided `setup` script will check for
you as it sets up your environment. It will even try to install them for
you using [pip](https://pypi.python.org/pypi/pip/) if they are missing.

To generate new custom data (.csv) files, check out my Extractor
repository,
[DUNE-FMC-Extract](https://github.com/samkohn/DUNE-FMC-Extract).

### Download
To download this repository, either clone it or download a packaged
version. To clone, use the git command

```
$ git clone https://github.com/samkohn/DUNE-configs.git
```

You can download the latest (unstable) version
[here](https://github.com/samkohn/DUNE-configs/archive/master.zip) or
using the "Download ZIP" button at the top of this page. To download a
stable release version, go to the
[releases](https://github.com/samkohn/DUNE-configs/releases) page and
click on either the zip or tar.gz link.

### Scripts
To run the scripts, first clone or download this repository (see above
instructions). Then, ensure that your system is configured correctly and
set up the correct python path information. Source the `setup` file with
`source setup`. This needs to be done every time a new shell session is
started or the code in the dunesim module will not work.

The easiest script to start with is the plotter.py script, located in
the scripts folder. `cd` there and then you can run
`python plotter.py --help` to get a summary of the available commands.
Here are some examples:

```
# Plot variations in nue spectrum due to uncertainty in mixing angles
# Include integrated number of events (-N) and chi-square (--x2)
$ python plotter.py oscparam -N --x2

# Antineutrino mode (automatically plots nuebar spectrum)
# Also include a ratio plot comparing to the nominal spectrum
$ python plotter.py oscparam --bar --ratio

# Plot numu CC-like spectrum
$ python plotter.py oscparam --flavor muCC

# Suppress signal and only look at background
$ python plotter.py oscparam --suppress numu2nue numubar2nuebar

# Suppress nues from all sources
$ python plotter.py oscparam --suppress nue2nue numu2nue nutau2nue

# Plot variations due to normalization uncertainty
$ python plotter.py norm -N
```

To adjust the beam run parameters, use the command-line arguments
`--pot-per-year` and `--years` (or `-p` and `-y`, respectively).

The plotting script also allows for comparisons between calculated data
sourced from this repository and externally derived data. To make such a
comparison, first create a plain-text file with the spectrum data binned
the same way it is here (in 120 bins from 0 to 10 GeV). The file should
have the value for each bin on its own line. Then you can refer to it
using the following command:

```
# Compare a calculated spectrum to a manually specified spectrum
$ python plotter.py manual --manual-spectrum path/to/manual/spectrum
```

All of the usual command line arguments (`--ratio`, `--bar`, `--suppress`,
`--flavor`, etc.) can be used here as well.

Lastly, to use the plotter with different inputs, simply set up your own
directory with similar contents to this repository's
[Fast-Monte-Carlo](https://github.com/samkohn/DUNE-configs/tree/master/Fast-Monte-Carlo)
directory. If you won't be using the `oscparam` analysis, then the
nested `Oscillation-Parameters/Parameter-Sets` folders are not
necessary. In any event, check out the DUNE-FMC-Extract module
mentioned above if you want to generate inputs based on the DUNE Fast
Monte Carlo and you have access to the FNAL cluster's `/dune` file
system.

The plotter module can be extended to include new analyses. Pull
requests are welcome!

### Provenance and maintenance information
Each plot is labeled with the git commit description representing the
version used to make the plot. Also included is the specific python
command that was called to generate the plot. This allows us to recover
the inputs and assumptions that went into each plot. The only tricky
part is that plots are made with clean git working trees, so that the
files represented by the commit descriptor are exactly the same as the
files used to create the plot.

If you use external files not in the git repository, then make sure to
preserve them. In some cases I may accept a pull request to add them to
this repository if they are important to the experiment.


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
