/*
Created by Nijat Rustamov on Dec 4, 2021
3D pore search algrithm insipired by Lattice Boltzmann Method
Searches and removes pores that are disconnceted from either inlet 
or outlet or both.
*/


#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <math.h>
#include <unistd.h>
using namespace std;

void getPoreSpaceFromText(vector<int> &pore_, double &entry, int &count)
{
    ifstream inFile;
    inFile.open("pore_test.txt");
    count = 0;
    if (inFile.is_open())
    {
        while (inFile >> entry)
        {
            pore_[count] = entry;
            count++;
        }
    }
    else
    {
        throw std::invalid_argument("Can't open file");
    }
}

void outputPoreSpaceToText(vector<int> &pore_)
{
    std::ofstream pore_out("pore_cleaned.txt");
    for (vector<int>::const_iterator i = pore_.begin(); i != pore_.end(); ++i)
    {
        pore_out << std::setprecision(1) << *i << '\n';
    }
}

int main()
{
    // Declare variables
    const int NZ = 64, NX = 801, NY = 801, nc = 19;
    const vector<int> C_X{0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
    const vector<int> C_Y{0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1};
    const vector<int> C_Z{0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1};
    int64_t vec_size = NX * NY * NZ;
    double entry;
    int count = 0;
    int prevStepGroupSize = 0, currentStepGroupSize = 1, idtemp;

    // load pore
    // Note: loaded pore space has already beeen cleanead from NAN normals and padded with zeros
    vector<int> pore(vec_size);
    getPoreSpaceFromText(pore, entry, count);
    cout << "domain loaded " << endl;

    //----------------------------------------------------------------------------------------- Pore search
    // count number of pores
    count = 0;
#pragma omp parallel for default(shared) reduction(+ \
                                                   : count)
    for (int index = 0; index < vec_size; index++)
    {
        if (pore[index] == 1)
            count++;
    }
    cout << "number of pores are " << count << endl;

    // save indices of pores
    vector<int> poreIndices(count);
    count = 0;
    for (int index = 0; index < vec_size; index++)
    {
        if (pore[index] == 1)
        {
            poreIndices[count] = index;
            count++;
        }
    }

    cout << "size of poreIndices " << poreIndices.size() << endl;
    cout << "first element of poreIndices " << poreIndices[0] << endl;
    cout << "last element o poreIndices " << poreIndices[poreIndices.size() - 1] << endl;

    // pore search
    vector<int> poreGroups{-1};
    int i, j, k;
    bool isPartOf;

    for (auto &id : poreIndices)
    {
        isPartOf = 0;
        //id = (i - 1) + (k - 1) * 64 * 2048 + (j - 1) * 64; // 3D index to linear index

        // looking for id in poreGroups
        for (int ic = 0; ic < poreGroups.size(); ic++)
        {
            if (poreGroups[ic] == id)
            {
                isPartOf = 1;
                break;
            }
        }

        //(find(poreGroups.begin(), poreGroups.end(), id) != poreGroups.end()) == 0
        if (isPartOf == 0)
        {
            vector<int> idNeigborsGroup{id};
            prevStepGroupSize = idNeigborsGroup.size();        // initial value before loop
            currentStepGroupSize = idNeigborsGroup.size() + 1; // initial value before loop
            while (prevStepGroupSize != currentStepGroupSize)
            {
                prevStepGroupSize = idNeigborsGroup.size();
                //cout << prevStepGroupSize << endl;
                // Finding neighbors of current id
                vector<int> idNeigbors;                                // no preallocation
                idNeigbors.reserve(idNeigborsGroup.size() * (nc - 1)); // reserve the memory to avoid reallocation

                for (int ic = 0; ic < idNeigborsGroup.size(); ic++)
                    for (int icc = 1; icc < nc; icc++)
                    {
                        idtemp = idNeigborsGroup[ic] + C_X[icc] * (NZ * NY) + C_Y[icc] * NZ + C_Z[icc]; // neighbor cell
                        if (idtemp >= 0 && idtemp < vec_size && pore[idtemp] == 1)                      // if it is in the domain and notin the wall
                        {
                            idNeigbors.push_back(idtemp); // since vector memory reserved no reallocation is performed
                        }
                    }

                // Clean cells that already exist in the group
                idNeigborsGroup.insert(idNeigborsGroup.end(), idNeigbors.begin(), idNeigbors.end());

                vector<int>::iterator ip;
                sort(idNeigborsGroup.begin(), idNeigborsGroup.end());
                ip = unique(idNeigborsGroup.begin(), idNeigborsGroup.end());
                idNeigborsGroup.resize(distance(idNeigborsGroup.begin(), ip));

                // new group size
                currentStepGroupSize = idNeigborsGroup.size();
            }

            // store individual groups
            poreGroups.insert(poreGroups.end(), idNeigborsGroup.begin(), idNeigborsGroup.end());

            // remove group of pores if not connected
            vector<int> idNeigborsGroup_i(idNeigborsGroup.size());
            //#pragma omp parallel for default(shared) private(k, j, i)
            for (int ic = 0; ic < idNeigborsGroup.size(); ic++)
            {
                k = ceil((double)(idNeigborsGroup[ic] + 1) / (double)(NZ * NX));
                j = ceil((double)(idNeigborsGroup[ic] + 1 - (k - 1) * NZ * NX) / (double)NZ);
                i = (idNeigborsGroup[ic] + 1) - (k - 1) * NZ * NX - (j - 1) * NZ;
                idNeigborsGroup_i[ic] = i;
            }

            if (*min_element(idNeigborsGroup_i.begin(), idNeigborsGroup_i.end()) != 1 || *max_element(idNeigborsGroup_i.begin(), idNeigborsGroup_i.end()) != NZ)
            {
#pragma omp parallel for default(shared)
                for (int ic = 0; ic < idNeigborsGroup.size(); ic++)
                    pore[idNeigborsGroup[ic]] = 0;

                cout << idNeigborsGroup.size() << " cells removed " << endl;
            }
            else
            {
                cout << idNeigborsGroup.size() << " cells kept " << endl;
                cout << id << endl;
                if (idNeigborsGroup.size() < 64)
                {
                    int min = *min_element(idNeigborsGroup_i.begin(), idNeigborsGroup_i.end());
                    int max = *max_element(idNeigborsGroup_i.begin(), idNeigborsGroup_i.end());
                    cout << "min " << min << " max " << max << endl;
                    for (int ic = 0; ic < idNeigborsGroup.size(); ic++)
                        cout << idNeigborsGroup[ic] << endl;
                }
            }
        }

        if (id % 100000 == 0)
            cout << "cells checked " << id << endl;
    }

    outputPoreSpaceToText(pore);
    cout << "pore saved" << endl;
    return 0;
}