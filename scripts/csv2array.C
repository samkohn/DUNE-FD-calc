/**
 * Read data from a CSV file into a C++ array.
 * No line in the CSV file can have more than 10000 characters.
 */
#include <fstream>
int csv2array(std::string filename, std::string* outarray, const size_t length)
{
    int returnval = 0;
    // Open the file
    std::ifstream fin(filename.c_str(), std::ifstream::in);
    if(!fin.good())
    {
        returnval = 4;
        std::cout << "ERROR: Could not open file: " << filename << "\n";
        return returnval;
    }
    else
    {
        std::cout << "INFO: Opened file." << std::endl;
    }
    std::string* entry = 0;
    size_t nentry = 0;
    const size_t bufsize = 10000;
    char buffer[bufsize];
    while(fin.good() && nentry < length)
    {
        fin.getline(buffer, bufsize-1);
        char* token = 0;
        char delimiters[10] = ", \n";
        token = strtok(buffer, delimiters);
        while(token != 0 && nentry < length)
        {
            outarray[nentry] = token;
            ++nentry;
            token = strtok(NULL, delimiters);
        }
    }
    fin.get();
    fin.get();
    if(fin.eof())
    {
        std::cout << "INFO: Read entire file." << std::endl;
    }
    else
    {
        returnval = 5;
        std::cout << "ERROR: Could not read entire file." << std::endl;
        return returnval;
    }
    if(nentry == length)
    {
        std::cout << "INFO: Number of entries is correct." << std::endl;
    }
    else
    {
        returnval = 6;
        std::cout << "ERROR: Too few entries: " << nentry << std::endl;
        return returnval;
    }
    return returnval;
}
