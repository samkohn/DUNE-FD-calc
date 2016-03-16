/*
 * This macro file will process a flux vector of a single neutrino flavor
 * in CSV format through the approximate matrix-based procedure to get
 * reconstructed spectra for the given neutrino flavor to oscillate to
 * all 3 flavors.
 */
#include "NuIndex2str.C"
#include "csv2array.C"
#include "csv2matrix.C"
#include "Configuration.C"
const size_t NUM_FLAVORS = 3;

/*
 * This macro runs all of the other macros on this page in order
 */
int ProcessFlux(const int STARTNU, const size_t NBINS,
        const double EMIN, const double EMAX,
        const double NTARGETS, std::string infile,
        std::string outfile="tmp.csv")
{
    int ret = 0;
    std::cout << "INFO: Beginning Flux2OscFlux\n";
    ret = Flux2OscFlux(STARTNU, NBINS, EMIN, EMAX, infile);
    std::cout << "INFO: Ending Flux2OscFlux\n";
    if(ret != 0) return ret;
    std::cout << "INFO: Beginning OscFlux2TrueSpectrum\n";
    ret = OscFlux2TrueSpectrum(STARTNU, NBINS, EMIN, EMAX, NTARGETS);
    std::cout << "INFO: Ending OscFlux2TrueSpectrum\n";
    if(ret != 0) return ret;
    std::cout << "INFO: Beginning TrueSpec2RecoSpec\n";
    ret = TrueSpec2RecoSpec(STARTNU, NBINS, EMIN, EMAX);
    std::cout << "INFO: Ending TrueSpec2RecoSpec\n";
    if(ret != 0) return ret;
    std::cout << "INFO: Beginning RecoSpec2SignalSpec\n";
    ret = RecoSpec2SignalSpec(STARTNU, NBINS, EMIN, EMAX, outfile);
    std::cout << "INFO: Ending RecoSpec2SignalSpec\n";
    return ret;
}



/*
 * This macro converts an initial flux into 3 oscillated fluxes, one for
 * each flavor.
 *
 * If no outfile is given, the default of tmp.csv will be used.
 *
 * The input format is a CSV file (separators are , and newline) with
 * NBINS entries denoting the beam flux of neutrino flavor STARTNU (1=e,
 * 2=mu, 3=tau, negative = bar), in energy bins running from EMIN to
 * EMAX.
 *
 * The output format is a CSV file with NUM_FLAVORS*NBINS (NUM_FLAVORS =
 * 3, currently) entries denoting the oscillated flux into e, nu and tau
 * for the first, second and third NBINS entries. (e.g. if NBINS = 40,
 * the first 40 entries give the flux for oscillations to nue(bar), the
 * second 40 (i.e. 40-79) give the flux for oscillations to numu(bar),
 * and so on.
 */
int Flux2OscFlux(const int STARTNU, const size_t NBINS,
        const double EMIN, const double EMAX,
        std::string infile, std::string outfile="tmp.csv")
{
    char outputdir[100];
    CFG_OutputDirectory(outputdir);
    double influx[NBINS];
    std::string influxstr[NBINS];
    double outflux[NUM_FLAVORS][NBINS];
    double probs[NUM_FLAVORS][NBINS];
    std::string probsstr[NUM_FLAVORS][NBINS];
    int result = csv2array(infile, influxstr, NBINS);
    if(result != 0)
    {
        std::cout << "ERROR: Result != 0: " << result << "\n";
        return result;
    }
    else
    {
        std::cout << "INFO: Correctly read in csv file.\n";
    }
    // Convert the string array into a double array
    double inbin = 0;
    for(size_t i = 0; i < NBINS; ++i)
    {
        inbin = atof(influxstr[i].c_str());
        influx[i] = inbin;
    }

    // Now we have the input flux.
    // We need to process the flux according to the oscillation
    // probabilities, so the next step is to read in the oscillation
    // probabilities.
    int nusign = STARTNU > 0 ? 1 : -1;
    char filenameend[10];
    sprintf(filenameend, "%d.csv", NBINS);
    std::string startnustr;
    std::string endnustr[NUM_FLAVORS];
    std::string finstr[NUM_FLAVORS];

    NuIndex2str(STARTNU, startnustr);
    for(size_t i = 0; i < NUM_FLAVORS; ++i)
    {
        NuIndex2str((i + 1) * nusign, endnustr[i]);
        finstr[i] = std::string(outputdir) + "Oscillation-Parameters/" +
            startnustr + "_" + endnustr[i] + filenameend;
        // Read in the probabilities
        result = csv2array(finstr[i], probsstr[i], NBINS);
        if(result != 0)
        {
            return result;
        }
        for(size_t bin = 0; bin < NBINS; ++bin)
        {
            double probbin = atof(probsstr[i][bin].c_str());
            probs[i][bin] = probbin;
        }
    }

    // Now perform the multiplication to get the oscillated spectra
    std::ofstream fout;
    fout.open(outfile.c_str());
    if(!fout.is_open())
    {
        std::cout << "Error: Could not open file\n";
        return 2;
    }
    else
    {
        std::cout << "INFO: Opened output file\n";
    }
    for(size_t flavor = 0; flavor < NUM_FLAVORS; ++flavor)
    {
        for(size_t bin = 0; bin < NBINS; ++bin)
        {
            outflux[flavor][bin] = influx[bin] * probs[flavor][bin];
            fout << outflux[flavor][bin] << "\n";
        }
    }
    fout.close();
    return 0;
}

/*
 * This macro converts an oscillated neutrino flux (e, mu and tau) into
 * true interaction spectra by using the computed charged-current cross
 * sections.
 *
 * If no infile or outfile is given, the default of tmp.csv will be
 * used and the input will be overwritten. If an infile is given but no
 * outfile, the default of tmp.csv will be used for the outfile and the
 * input will be preserved.
 *
 * The input format is the same as the output format of Flux2OscFlux.
 * The output format is analogous to the input format, except the
 * entries represent numbers of interactions rather than fluxes.
 */
int OscFlux2TrueSpectrum(const int STARTNU, const size_t NBINS,
        const double EMIN, const double EMAX, const double NTARGETS,
        std::string infile="tmp.csv", std::string outfile="tmp.csv")
{
    char outputdir[100];
    CFG_OutputDirectory(outputdir);
    const int NUSIGN = STARTNU > 0 ? +1 : -1;
    const double XSEC_UNITS = 1e-38; // cm^2
    double influx[NUM_FLAVORS][NBINS];
    std::string influxstr[NUM_FLAVORS * NBINS];
    double outspec[NUM_FLAVORS][NBINS];
    double xsec[NUM_FLAVORS][NBINS];
    std::string xsecstr[NUM_FLAVORS][NBINS];
    int result = csv2array(infile, influxstr, NUM_FLAVORS * NBINS);
    if(result != 0)
    {
        std::cout << "ERROR: Result != 0: " << result << "\n";
        return result;
    }
    else
    {
        std::cout << "INFO: Correctly read in csv file.\n";
    }

    // Convert the 3N*1 string array into an N*3 double array
    double value = 0;
    size_t flavor = 0;
    size_t ebin = 0;
    for(size_t bin = 0; bin < NUM_FLAVORS * NBINS; ++bin)
    {
        flavor = bin/NBINS; // Integer division
        ebin = bin % NBINS;
        value = atof(influxstr[bin].c_str());
        influx[flavor][ebin] = value;
    }

    // Read in cross sections
    char filenameend[10];
    sprintf(filenameend, "%d.csv", NBINS);
    std::string fileprefix = std::string(outputdir) + "Cross-Sections/";
    for(size_t nuflavor = 1; nuflavor <= NUM_FLAVORS; ++nuflavor)
    {
        int nutype = nuflavor * NUSIGN;
        bool useUnderscores = true;
        std::string nutypestr;
        NuIndex2str(nutype, nutypestr, useUnderscores);
        std::string infilestr = fileprefix + nutypestr +
            "_Ar40__tot_cc" + filenameend;
        result = csv2array(infilestr, xsecstr[nuflavor-1], NBINS);
        if(result != 0)
        {
            std::cout << "ERROR: Result != 0: " << result << "\n";
            return result;
        }
        else
        {
            std::cout << "INFO: Correctly read in csv file.\n";
        }
        // Convert cross sections to doubles
        double value = 0;
        for(size_t ebin = 0; ebin < NBINS; ++ebin)
        {
            value = atof(xsecstr[nuflavor-1][ebin].c_str());
            xsec[nuflavor-1][ebin] = value;
        }
    }

    // Create the spectrum by multiplying the flux by the cross section
    for(size_t flavor = 0; flavor < NUM_FLAVORS; ++flavor)
    {
        for(size_t ebin = 0; ebin < NBINS; ++ebin)
        {
            outspec[flavor][ebin] = influx[flavor][ebin] *
                xsec[flavor][ebin] * NTARGETS * XSEC_UNITS;
        }
    }
    std::ofstream fout;
    // The trunc option overwrites the existing file.
    fout.open(outfile.c_str(), std::ofstream::trunc);
    if(!fout.is_open())
    {
        std::cout << "Error: Could not open file\n";
        return 2;
    }
    else
    {
        std::cout << "INFO: Opened output file\n";
    }
    for(size_t flavor = 0; flavor < NUM_FLAVORS; ++flavor)
    {
        for(size_t bin = 0; bin < NBINS; ++bin)
        {
            fout << outspec[flavor][bin] << "\n";
        }
    }
    fout.close();
    return 0;
}

TrueSpec2RecoSpec(const int STARTNU, const size_t NBINS,
        const double EMIN, const double EMAX,
        std::string infile="tmp.csv", std::string outfile="tmp.csv")
{
    char outputdir[100];
    CFG_OutputDirectory(outputdir);
    const int NUSIGN = STARTNU > 0 ? +1 : -1;
    TMatrixD* detectorresponse[NUM_FLAVORS];
    double inspec[NUM_FLAVORS][NBINS];
    std::string inspecstr[NUM_FLAVORS * NBINS];
    TMatrixD* inspecmat[NUM_FLAVORS];
    double outspec[NUM_FLAVORS][NBINS];
    TMatrixD* outspecmat[NUM_FLAVORS];
    int result = csv2array(infile, inspecstr, NUM_FLAVORS * NBINS);
    if(result != 0)
    {
        std::cout << "ERROR: Result != 0: " << result << "\n";
        return result;
    }
    else
    {
        std::cout << "INFO: Correctly read in csv file.\n";
    }
    // Convert the 3N*1 string array into an N*3 double array
    double value = 0;
    size_t flavor = 0;
    size_t ebin = 0;
    for(size_t bin = 0; bin < NUM_FLAVORS * NBINS; ++bin)
    {
        flavor = bin/NBINS; // Integer division
        ebin = bin % NBINS;
        value = atof(inspecstr[bin].c_str());
        inspec[flavor][ebin] = value;
    }
    // Convert the double array into an array of TMatrixD column
    // vectors.
    for(size_t flavor = 0; flavor < NUM_FLAVORS; ++flavor)
    {
        inspecmat[flavor] = new TMatrixD(NBINS, 1);
        inspecmat[flavor]->SetMatrixArray(inspec[flavor]);
    }

    // Read in the detector response matrices
    std::string filename[NUM_FLAVORS];
    std::string prefix = std::string(outputdir) + "Detector-Response/DetRespMat-";
    if(NUSIGN > 0)
    {
        prefix += "nuflux_";
    }
    else
    {
        prefix += "anuflux_";
    }
    char filenameend[10] = ".csv";
    //sprintf(filenameend, "%d.csv", NBINS);
    for(size_t flavor = 0; flavor < NUM_FLAVORS; ++flavor)
    {
        std::string temp;
        NuIndex2str(2 * NUSIGN, temp);
        filename[flavor] = prefix + temp + "flux_";
        NuIndex2str((flavor + 1) * NUSIGN, temp);
        filename[flavor] += temp + filenameend;
        detectorresponse[flavor] = new TMatrixD(NBINS, NBINS);
        int result = csv2matrix(filename[flavor], NBINS, NBINS,
                detectorresponse[flavor]);
        if(result != 0)
        {
            std::cout << "ERROR: Result != 0: " << result << "\n";
            std::cout << "filename = " << filename[flavor] << "\n";
            return result;
        }
        else
        {
            std::cout << "INFO: Correctly read in csv file.\n";
        }
    }

    // Multiply the matrices
    for(size_t flavor = 0; flavor < NUM_FLAVORS; ++flavor)
    {
        outspecmat[flavor] = new TMatrixD(NBINS, 1);
        outspecmat[flavor]->Mult(*(detectorresponse[flavor]), *(inspecmat[flavor]));
        // Extract the product into an array
        double* elements = outspecmat[flavor]->GetMatrixArray();
        for(size_t i = 0; i < NBINS; ++i)
        {
            outspec[flavor][i] = elements[i];
        }
    }

    // Dump the results to a CSV file
    std::ofstream fout;
    // The trunc option overwrites the existing file.
    fout.open(outfile.c_str(), std::ofstream::trunc);
    if(!fout.is_open())
    {
        std::cout << "Error: Could not open file\n";
        return 2;
    }
    else
    {
        std::cout << "INFO: Opened output file\n";
    }
    for(size_t flavor = 0; flavor < NUM_FLAVORS; ++flavor)
    {
        for(size_t bin = 0; bin < NBINS; ++bin)
        {
            fout << outspec[flavor][bin] << "\n";
        }
    }
    fout.close();

    return 0;
}
/*
 * This macro converts a reconstructed spectrum into a signal spectrum
 * by applying an energy-dependent efficiency factor.
 *
 * If no infile or outfile is given, the default of tmp.csv will be
 * used and the input will be overwritten. If an infile is given but no
 * outfile, the default of tmp.csv will be used for the outfile and the
 * input will be preserved.
 *
 * The input format is the same as the output format of Flux2OscFlux.
 * The output format is analogous to the input format, except the
 * entries represent numbers of interactions rather than fluxes.
 */
int RecoSpec2SignalSpec(const int STARTNU, const size_t NBINS,
        const double EMIN, const double EMAX,
        std::string outfile="tmp.csv", std::string infile="tmp.csv")
{
    char outputdir[100];
    CFG_OutputDirectory(outputdir);
    const int NUSIGN = STARTNU > 0 ? +1 : -1;
    double inspec[NUM_FLAVORS][NBINS];
    std::string inspecstr[NUM_FLAVORS * NBINS];
    double outspec[NUM_FLAVORS][NBINS];
    double efficiencies[NUM_FLAVORS][NBINS];
    std::string efficienciesstr[NUM_FLAVORS][NBINS];
    int result = csv2array(infile, inspecstr, NUM_FLAVORS * NBINS);
    if(result != 0)
    {
        std::cout << "ERROR: Result != 0: " << result << "\n";
        return result;
    }
    else
    {
        std::cout << "INFO: Correctly read in csv file.\n";
    }

    // Convert the 3N*1 string array into an N*3 double array
    double value = 0;
    size_t flavor = 0;
    size_t ebin = 0;
    for(size_t bin = 0; bin < NUM_FLAVORS * NBINS; ++bin)
    {
        flavor = bin/NBINS; // Integer division
        ebin = bin % NBINS;
        value = atof(inspecstr[bin].c_str());
        inspec[flavor][ebin] = value;
    }

    // Read in efficiencies
    char filenameend[10];
    sprintf(filenameend, "%d.csv", NBINS);
    std::string fileprefix = std::string(outputdir) + "Efficiencies/";
    for(size_t nuflavor = 1; nuflavor <= 1/*NUM_FLAVORS TODO*/; ++nuflavor)
    {
        std::string infilestr = fileprefix + "nueCCsig_efficiency.csv";
        result = csv2array(infilestr, efficienciesstr[nuflavor-1], NBINS);
        if(result != 0)
        {
            std::cout << "ERROR: Result != 0: " << result << "\n";
            return result;
        }
        else
        {
            std::cout << "INFO: Correctly read in csv file.\n";
        }
        // Convert efficiencies to doubles
        double value = 0;
        for(size_t ebin = 0; ebin < NBINS; ++ebin)
        {
            value = atof(efficienciesstr[nuflavor-1][ebin].c_str());
            efficiencies[nuflavor-1][ebin] = value;
        }
    }

    // Create the spectrum by multiplying the reco spectrum by the
    // efficiencies
    for(size_t flavor = 0; flavor < 1/*NUM_FLAVORS TODO*/; ++flavor)
    {
        for(size_t ebin = 0; ebin < NBINS; ++ebin)
        {
            outspec[flavor][ebin] = inspec[flavor][ebin] *
                efficiencies[flavor][ebin];
        }
    }
    std::ofstream fout;
    // The trunc option overwrites the existing file.
    fout.open(outfile.c_str(), std::ofstream::trunc);
    if(!fout.is_open())
    {
        std::cout << "Error: Could not open file\n";
        return 2;
    }
    else
    {
        std::cout << "INFO: Opened output file\n";
    }
    for(size_t flavor = 0; flavor < 1/*NUM_FLAVORS TODO*/; ++flavor)
    {
        for(size_t bin = 0; bin < NBINS; ++bin)
        {
            fout << outspec[flavor][bin] << "\n";
        }
    }
    fout.close();
    return 0;
}
