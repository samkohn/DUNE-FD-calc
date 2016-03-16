/**
 * Read data from a CSV file into a TMatrixD. The format of the CSV file
 * is extremely specific:
 * The only allowable characters are: 0123456789., \n
 * Where the last two characters are "space" and "newline".
 * The numbers must be formatted correctly as ints or doubles to be used
 * as input to atof. The number of entries in the CSV file must match
 * exactly the number of entries expected in the matrix. The function
 * will fail if there are too many or too few entries. There must be at
 * most one newline at the end of the file after the last entry. The
 * function will fail if there is more than one newline at the end of
 * the file. The actual layout of the entries, i.e. how many in each
 * row, is not important, as the ROOT matrix library reads in entries
 * linearly, row by row. So, a 10x10 matrix could be a single line of
 * 100 entries, although I don't recommend it.
 */
#include <fstream>
int csv2matrix(std::string filename, const int nrows, const int ncols, TMatrixD* matrix)
{
    // Check to see if the supplied matrix is the right size
    bool goodrows = nrows == matrix->GetNrows();
    bool goodcols = ncols == matrix->GetNcols();
    int returnval;
    if(goodrows && goodcols)
    {
        returnval = 0;
    }
    else if(goodrows)
    {
        returnval = 1;
        std::cout << "ERROR: Matrix has wrong number of columns: parameter = ";
        std::cout << ncols << ", matrix->GetNcols() = " << matrix->GetNcols();
        std::cout << std::endl;
        return returnval;
    }
    else if(goodcols)
    {
        returnval = 2;
        std::cout << "ERROR: Matrix has wrong number of rows: parameter = ";
        std::cout << nrows << ", matrix->GetNrows() = " << matrix->GetNrows();
        std::cout << std::endl;
        return returnval;
    }
    else
    {
        returnval = 3;
        std::cout << "ERROR: Matrix has wrong size: rows = ";
        std::cout << nrows << ", matrix->GetNrows() = " << matrix->GetNrows();
        std::cout << ", cols = " << ncols << ", matrix->GetNcols() = ";
        std::cout << matrix->GetNcols() << std::endl;
        return returnval;
    }

    // Open the file
    std::ifstream fin(filename.c_str(), std::ifstream::in);
    if(!fin.good())
    {
        returnval = 4;
        std::cout << "ERROR: Could not open file." << std::endl;
        return returnval;
    }
    else
    {
        std::cout << "INFO: Opened file." << std::endl;
    }

    // Read in the data into an array.
    // Note: TMatrixD takes a linear array by rows, e.g. [1, 2, 3, 4]
    // will lead to the matrix:
    // 1    2
    // 3    4
    // Consequently, I can just read in the data from the CSV file
    // linearly.
    const int nentries = nrows * ncols;
    double entries[nentries];
    double entry = 0;
    size_t nentry = 0;
    const size_t bufsize = 10000;
    char buffer[bufsize];
    while(fin.good() && nentry < nentries)
    {
        fin.getline(buffer, bufsize-1);
        char* token = 0;
        char delimiters[10] = ", \n";
        token = strtok(buffer, delimiters);
        while(token != 0 && nentry < nentries)
        {
            entry = atof(token);
            entries[nentry] = entry;
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
    if(nentry == nentries)
    {
        std::cout << "INFO: Number of entries is correct." << std::endl;
    }
    else
    {
        returnval = 6;
        std::cout << "ERROR: Too few entries: " << nentry << std::endl;
        return returnval;
    }

    // Insert the data into the matrix
    matrix->SetMatrixArray(entries);
    return returnval;
}
