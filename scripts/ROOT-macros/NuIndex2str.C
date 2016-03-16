#include <TMath.h>
#include <string>

int NuIndex2str(int nuIndex, std::string& str, bool useUnderscore=false)
{
    int nuflavor = (int) TMath::Abs(nuIndex);
    if(useUnderscore)
    {
        switch (nuflavor)
        {
            case 1:
                str = "nu_e";
                break;
            case 2:
                str = "nu_mu";
                break;
            case 3:
                str = "nu_tau";
                break;
            default:
                std::cout << "ERROR: invalid neutrino index: " << nuIndex << "\n";
                return 1;
                break;
        }
        if(nuIndex < 0)
        {
            str += "_bar";
        }
        else
        {
            str += "";
        }
    }
    else
    {
        switch (nuflavor)
        {
            case 1:
                str = "nue";
                break;
            case 2:
                str = "numu";
                break;
            case 3:
                str = "nutau";
                break;
            default:
                std::cout << "ERROR: invalid neutrino index: " << nuIndex << "\n";
                return 1;
                break;
        }
        if(nuIndex < 0)
        {
            str += "bar";
        }
        else
        {
            str += "";
        }
    }
    return 0;
}
