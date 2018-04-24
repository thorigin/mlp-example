#ifndef MLP_UTIL_HPP
#define MLP_UTIL_HPP

#include <random>
#include <exception>

namespace mlp {

using float_t = float;

using vec_t = std::vector<float_t>;
using samples_vec_t = std::vector<vec_t>;
using labels_vec_t = std::vector<size_t>;


/**
 * Network error exception
 */
struct mlp_error : std::exception {

    mlp_error(std::string msg) : why(std::move(msg)) {}

    const char* what () const throw () {
        return why.c_str();
    }

    std::string why;
};


/**
 * Loads a CSV file with a specific number of data points, and the last
 * value is the label
 */
void load_csv(     const std::string& file_path,
                    const size_t& data_points,
                    samples_vec_t& data,
                    labels_vec_t& labels) {

    std::ifstream file(file_path); // File to load data

    /*
     * Part 0: Code to load in from a CSV, no need to edit this
     */
    std::string line;
    while (std::getline(file, line)) {
        vec_t row;
        std::stringstream iss(line);

        std::string val;
        for(size_t i = 0; i < data_points; ++i) {
            if(std::getline(iss, val, ',')) {
                double temp = std::stof(val);
                row.push_back(temp);
            } else {
                throw mlp_error("invalid data");
            }
        }
        if(std::getline(iss, val, ',')) {
            double temp = std::stof(val);
            labels.push_back(temp);
        } else {
            throw mlp_error("invalid data");
        }
        data.push_back(row);
    }
}


/**
 * Normalize a vector of a vector using minmax to range [a, b]
 *
 * First it normalizes to [0, 1] and then translates the function to [a, b]
 */
void normalize(samples_vec_t& values, const float_t& a = 0,  const float_t& b = 1) {

    if(values.empty()) {
        return;
    }

    const size_t row_len = values[0].size();
    const size_t sample_count = values.size();

    vec_t min_of_col(row_len, std::numeric_limits<float_t>::max());
    vec_t max_of_col(row_len, std::numeric_limits<float_t>::min());

    for(size_t row = 0; row < sample_count; ++row) {
        for(size_t col = 0; col < row_len; ++col) {
            min_of_col[col] = std::min(min_of_col[col], values[row][col]);
            max_of_col[col] = std::max(max_of_col[col], values[row][col]);
        }
    }

    for(size_t row = 0; row < sample_count; ++row) {
        for(size_t col = 0; col < row_len; ++col) {
            auto& min = min_of_col[col];
            auto& max = max_of_col[col];
            values[row][col] = (b - a) * (values[row][col] - min) / (max - min) - a;
        }
    }
}


/**
 * Random generator helper class
 */
struct random_generator {

    using random_engine_type = std::default_random_engine;
    static random_engine_type& get() {
        static random_engine_type re = random_engine_type(std::random_device{}());
        return re;
    }
};

} /* end namespace mlp */

#endif /* MLP_UTIL_HPP */

