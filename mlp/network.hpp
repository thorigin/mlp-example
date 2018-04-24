#ifndef MLP_MLP_HPP
#define MLP_MLP_HPP

#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <iterator>
#include <limits>
#include "util.hpp"
#include "loss.hpp"
#include "activation.hpp"
#include "inner_product_layer.hpp"

namespace mlp {

/**
 * Training results
 */
struct results {
    size_t correct;
    size_t total;
    float_t accuracy;
};

/**
 * Multi-layer perceptron network
 *
 * @tparam Activation the activation function, defaults to sigmoid_activation
 * @tparam LossFunction the error function, defaults to the diff_loss
 */
template<typename Activation = sigmoid_activation, typename LossFunction = error_loss>
struct network {

    using activation_type = Activation;
    using layer_type = inner_product_layer<activation_type>;
    using loss_function_type = LossFunction;

    /**
     * Construct a new multiplayer perceptron with the dimensions given
     *
     * @param dimensions
     */
    network(std::vector<size_t> dimensions) {
        if(dimensions.empty()) {
            throw mlp_error{"Dimensions must be greater or equal to 1"};
        }
        auto it = dimensions.begin();
        auto end = dimensions.end();
        auto input_size = *it++;
        for(;it != end; ++it) {
            layers.emplace_back(input_size, *it);
            input_size = *it;
        }
    }

    /**
     * Test the multiplayer perceptron neural network using the data and labels provided
     * @return the results
     */
    results test(const samples_vec_t& data, const labels_vec_t& labels) {

        size_t total = 0;
        size_t correct = 0;

        if(data.size() != labels.size()) {
            throw mlp_error{"data and label size mismatch"};
        }

        auto label_it = labels.begin();
        for (const auto &row : data)  {
            forward(row);
            auto& res = layers.back().output;
            auto max_it = std::max_element(res.begin(), res.end());
            size_t prediction;
            size_t label = *label_it++;
            if(max_it != res.end()) {
                prediction = std::distance(res.begin(), max_it);
            } else {
                throw mlp_error{"Invalid forward result"};
            }
            ++total;
            if(label == prediction) {
                ++correct;
            }
        }

        return {
            correct,
            total,
            static_cast<float_t>(correct) / static_cast<float_t>(total)
        };
    }

    /**
     * Train the networks given the data and lables for a duration of epochs_max
     */
    void train(const samples_vec_t& data, const labels_vec_t& labels, size_t epochs_max = 1) {

        if(data.size() != labels.size()) {
            throw mlp_error{"data and label size mismatch"};
        }

        vec_t error(output_size());
        vec_t expected(output_size(), 0);

        size_t e = 0;
        for(;e < epochs_max; ++e) {
            auto label_it = labels.begin();
            for (const auto &row : data) {
                forward(row);
                label_to_vector(*label_it, expected);
                gradient(output(), expected, error);
                backward(error);
                update_weights();
                ++label_it;
            }
            if(on_epoch) {
                if(on_epoch()) {
                    break;
                }
            }
        }
    }

    /**
     * Perform forward propagation of the MLP network
     * @param input
     */
    void forward(const vec_t& in) {
        /**
         * Copy the provided input into the input layer's input
         */
        input() = in;
        for(auto it = layers.begin(), end = layers.end(); it != end; ++it) {

            it->forward();

            /**
             * If network has a next layer after '''it''', propagate the
             * output of the current layer as the input of the next
             */
            auto next = std::next(it, 1);
            if(next != end) {
                next->input = it->output;
            }
        }
    }

    /**
     * Perform backward propagation of the MLP network
     * @param error
     */
    void backward(const vec_t& error) {
        if(error.size() != output_size()) {
            throw mlp_error{"Error and output size mismatch"};
        }
        /**
         * Copy the error specified above to the output layer's output gradient
         */
        output_layer().output_grad = error;
        for(auto rit = layers.rbegin(), rend = layers.rend(); rit != rend; ++rit) {
            rit->backward();

            /**
             * If network has a previous layer, propagate the input gradient as
             * the output gradient of that layer
             *
             * (next for a backward iterator (rit) gives the previous element).
             */
            auto prev = std::next(rit, 1);
            if(prev != rend) {
                prev->output_grad = rit->input_grad;
            }
        }
    }

    /**
     * Converts a label to a vector (i.e., label '1' for a output size of 3 becomes [0, 1, 0])
     * @param label
     * @param result
     */
    void label_to_vector(const size_t& label, vec_t& result) {
        result.resize(output_size());
        if(label >= output_size()) {
            throw mlp_error("label too high for output dimension");
        }
        std::fill(result.begin(), result.end(), 0);
        result[label] = 1;
    }

    /**
     * Calculates the output gradeint using the loss function specified
     */
    void gradient( const vec_t& predicted,
                    const vec_t& observed,
                    vec_t& result) {
        //apply the derivative of the loss function (gradient)
        loss_function.df(predicted, observed, result);
    }

    /**
     * Update weights of each layer
     */
    void update_weights() {
        for(auto& l : layers) {
            l.update_weights(alpha);
        }
    }

    /**
     * Calculate the loss of the samples given the labels
     * @return the accumulated loss of the dataset provided
     */
    float_t loss(const samples_vec_t& samples, const labels_vec_t& labels) {
        float_t sum = 0;
        vec_t expected;
        for(size_t i = 0, len = samples.size(); i < len; ++i) {
            label_to_vector(labels[i], expected);
            forward(samples[i]);
            sum += loss_function.f(output(), expected);
        }
        return sum;
    }

    /**
     * Calculate the loss of the samples given the labels
     * @return the accumulated loss of the dataset provided
     */
    float_t loss_mean(const samples_vec_t& samples, const labels_vec_t& labels) {
        return loss(samples, labels) / samples.size();
    }

    /**
     * The input size of the network
     */
    size_t input_size() {
        return layers.front().input_size;
    }

    /**
     * Returns the output size of the network
     */
    size_t output_size() {
        return layers.back().output_size;
    }

    /**
     * Returns the output layer
     */
    layer_type& output_layer() {
        return layers.back();
    }

    /**
     * Returns the input layer
     */
    layer_type& input_layer() {
        return layers.front();
    }

    /**
     *
     * Returns the output vector
     * @return
     */
    vec_t& output() {
        return output_layer().output;
    }

    /**
     * Returns the input vector
     * @return
     */
    vec_t& input() {
        return input_layer().input;
    }

    std::vector<layer_type> layers;


    loss_function_type loss_function;
    /**
     * The learning rate of the network
     */
    float_t alpha = 0.01;

    /**
     * Function reference which can be bound to any function which is called after an epoch finishes
     */
    std::function<bool()> on_epoch;
};

} /* end namespace mlp */

#endif /* MLP_MLP_HPP */
