
#ifndef MLP_INNER_PRODUCT_LAYER_HPP
#define MLP_INNER_PRODUCT_LAYER_HPP

#include <vector>
#include <algorithm>
#include <functional>
#include <random>

#include "network.hpp"
#include "util.hpp"

namespace mlp {

/**
 * Inner product layer, also known as fully connected layer.
 *
 * @tparam Activation the attached activation function
 */
template<typename Activation>
class inner_product_layer {
public:

    using activation_type = Activation;

    /**
     * Construct a new inner_product_layer with the given parameters
     *
     * @param in the input size
     * @param out the output size
     */
    inner_product_layer(size_t in, size_t out)
        :   input_size(in),
            output_size(out),
            weights(in * out, 0),
            bias(out, 0),
            input(in, 0),
            output(out, 0),
            input_grad(in, 0),
            output_grad(out, 0),
            grad_weights(in * out, 0),
            grad_bias(out, 0) {

        /**
         * Simple default initialization
         * Gets the global random_generator and randomizes all weights to
         * a value between [-1, 1]
         */
        auto& rand = random_generator::get();
        float_t r = 1;
        std::uniform_real_distribution<double> dist(-r, r);
        for(size_t i = 0; i < in * out; i++) {
            weights[i] = dist(rand);
        }
    }

    /**
     * Perform forward propagation of the layer, essentially
     * (transpose(w) * x), where w is the weights, ^
     */
    void forward() {
        if(input.size() != input_size) {
            throw mlp_error{"input vectordoes not match input size"};
        }
        for(size_t out = 0; out < output_size; ++out) {
            float_t total = 0;
            for(size_t in = 0; in < input_size; ++in) {
                total += weights[out * input_size + in] * input[in];
            }
            total += bias[out];
            //Apply activation function f(x) to the total output
            output[out] = activator.f(total);
        }
    }

    /**
     * Perform backward propagation
     */
    void backward() {
        if(output_grad.size() != output_size) {
            throw mlp_error{"output gradient vector does not match output size"};
        }
        std::fill(input_grad.begin(), input_grad.end(), 0);
        for(size_t out = 0; out < output_size; ++out) {
            /**
             * Calculate the derivative of the activation value and multiply
             * it by the output value for every output.
             *
             * Example, for sigmoid, this is
             * '''(1 - f(x)) * f(x) * e'''
             * where x is the output, and f(x) is the sigmoid function, and e
             * is the error estimate (e = y - t).
             *
             * However, due to having stored the output, we derive f(x) in
             * terms of y, such that f(x) is passed as y for f(x), this then
             * translates to: '''(1-f'(y)) * f(x)''', and for sigmoid, f'(y)
             * is defined as: (1 - y) * y, where y = f(x) as input.
             */
            const auto grad = activator.df(output[out]) * output_grad[out];
            for(size_t in = 0; in < input_size; ++in) {
                /**
                 * Propagate the gradient as the contribution of the weight
                 */
                input_grad[in] += grad * weights[out * input_size + in];
                /**
                 * Store the accumulated error of the weight
                 */
                grad_weights[out * input_size + in] += input[in] * grad;
            }
            /**
             * The accumulated bias error
             */
            grad_bias[out] += grad;
        }
    }

    /**
     * Update the weights of the layer
     * @param alpha the learning rate to apply
     */
    void update_weights(const float_t& alpha) {
        for(size_t i = 0; i < input_size * output_size; ++i) {
            weights[i] -= alpha * grad_weights[i];
        }
        for(unsigned int i = 0; i < bias.size(); i++) {
            bias[i] -= alpha * grad_bias[i];
        }
        clear_deltas();
    }

    /**
     * Clears the accumulated gradient of weights and biases by setting it to
     * zero
     */
    void clear_deltas() {
        std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
        std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);
    }

    activation_type activator;
    const size_t input_size;
    const size_t output_size;

    /**
     * The weight of each input/output pair
     */
    vec_t weights;
    /**
     * The bias term for each output
     */
    vec_t bias;

    /**
     * Stores the input of the layer
     */
    vec_t input;
    /**
     * Stores the output of the layer
     */
    vec_t output;
    /**
     * Stores the input gradient of the layer
     */
    vec_t input_grad;
    /**
     * Stores the output gradient of the layer
     */
    vec_t output_grad;

    /**
     * Accumulated gradient error of weights
     */
    vec_t grad_weights;
    /**
     * Accumulated gradient error of bias
     */
    vec_t grad_bias;
};

} /* end namespace mlp */

#endif /* MLP_INNER_PRODUCT_LAYER_HPP */
