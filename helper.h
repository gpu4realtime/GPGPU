#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <numbers>

static bool is_number(std::string& token)
{
    return token.find_first_not_of("-+0123456789.") == std::string::npos;
}

template<typename dataT>
static std::pair<std::vector<dataT>, std::vector<std::string>> read_file(std::string filename, bool filter_char)
{
    std::ifstream f(filename);

    if (!f.is_open())
    {
        std::cerr << "ERROR: bad input file." << std::endl;

        return {};
    }

    std::vector<dataT> h_input_data;
    std::vector<std::string> h_input_data_names;

    std::string line;
    while (getline(f, line))
    {
        std::string val;

        std::stringstream s(line);
        while (getline(s, val, ','))
        {
            if (filter_char && !is_number(val))
            {
                h_input_data_names.push_back(val);
                continue;
            }

            if constexpr (std::is_same_v<decltype(h_input_data), std::vector<double>>)
            {
                h_input_data.push_back(stod(val));
            }

            if constexpr (std::is_same_v<decltype(h_input_data), std::vector<float>>)
            {
                h_input_data.push_back(stof(val));
            }

            if constexpr (std::is_same_v<decltype(h_input_data), std::vector<unsigned char>>)
            {
                h_input_data.push_back(stoi(val));
            }
        }
    }

    f.close();

    return { h_input_data,h_input_data_names };
}

template<typename dataT>
static void save_to_file_in_csv(const std::string& file_name, dataT data_to_save, int rows, int cols)
{
    std::ofstream output_file(file_name);

    output_file.precision(16);

    for (auto i = 0; i < rows; ++i)
    {
    	for (auto j = 0; j < cols; ++j)
    		output_file << data_to_save[i * cols + j] << ",";

    	output_file << "\n";
    }

    output_file.close();
}

template<typename dataT>
static bool cmp(const std::vector<dataT>& arr1, const std::vector<dataT>& arr2, dataT tol)
{
    if (arr1.size() != arr2.size())
        return false;

    for (auto i = 0; i < arr1.size(); ++i)
    {
        if (std::abs(arr1[i] - arr2[i]) > tol)
            return false;
    }

    return true;
}