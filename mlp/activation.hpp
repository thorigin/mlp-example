#ifndef MLP_ACTIVATION_HPP
#define MLP_ACTIVATION_HPP

namespace mlp {

/**
 * Sigmoid activation function
 */
struct sigmoid_activation {

    /**
     * Calculates the f(x) of the sigmoid
     */
    inline float_t f(const float_t& x) {
        return float_t(1.0) / (float_t(1.0) + std::exp(-x));
    }

    /**
     * Calculates the f'(y) of the sigmoid, where y = f(x)
     */
    inline float_t df(const float_t& y) {
        return (float_t(1.0) - y) * y;
    }

};

/**
 * Tanh hyperbolic activation function
 */
struct tanh_activation {


    /**
     * Calculates the f(x) of the sigmoid
     */
    inline float_t f(const float_t& x) {
        return std::tanh(x);
    }

    /**
     * Calculates the f'(y) of the sigmoid, where y = f(x)
     */
    inline float_t df(const float_t& y) {
        return (1.0f - y * y);
    }
};

} /* end namespace mlp */

#endif /* MLP_ACTIVATION_HPP */

