#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>

#include "mlp/network.hpp"

int main(int argc, char *argv[]) {

    using namespace mlp;

    if(argc != 2) {
        std::cerr << "Invalid input, try: " << argv[0] << " iris.csv\n";
        return 1;
    }

    //Sample data points
    samples_vec_t data;
    //Sample labels
    labels_vec_t labels;

    /**
     * load 4 values as data points, last as label
     */
    load_csv(argv[1], 4, data, labels);

    /**
     * Normalize the data to [0, 1]
     */
    normalize(data, 0, 1.0);

    try {
        /**
         * Configure the MLP as 4 layers:
         */
        network<> nn({4,6,6,6,3});
        nn.alpha = 0.02;
        nn.on_epoch = [&]() {
            auto res = nn.test(data, labels);
            auto fl = std::cout.flags();
            std::cout << std::setprecision(4) << std::fixed;
            std::cout << "Accuracy: " << res.accuracy*100.0 << "%, loss: " << nn.loss_mean(data, labels) << "\n";
            std::cout.flags(fl);
            return res.accuracy >= 1.0f;
        };

        std::cout << "Untrained loss: " << nn.loss_mean(data, labels) << "\n";

        nn.train(data, labels, 25000);

        std::cout << "Trained loss: " << nn.loss_mean(data, labels) << "\n";

        std::cout << "Final Accuracy: " << nn.test(data, labels).accuracy * 100.0f << "%\n";

    }catch(mlp_error& err) {
        std::cout << "Error occured: " << err.why << "\n\n";
    }
}
