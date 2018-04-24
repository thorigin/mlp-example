#ifndef MLP_LOSS_HPP
#define MLP_LOSS_HPP

#include "util.hpp"


namespace mlp {

/**
 * Error of loss function
 *
 */
struct error_loss {

    void df(    const vec_t& predicted,
                const vec_t& observed,
                vec_t& result) {
        for(size_t i = 0, len = predicted.size(); i < len; ++i) {
            result[i] = (predicted[i] - observed[i]);
        }
    }

    float_t f(   const vec_t& predicted,
                const vec_t& observed) {
        float_t sum = 0;
        for(size_t i = 0, len = predicted.size(); i < len; ++i) {
            sum += std::abs(predicted[i] - observed[i]);
        }
        return sum;
    }
};

/**
 * Absolute loss function
 */
struct absolute_loss {

    void df(    const vec_t& predicted,
                const vec_t& observed,
                vec_t& result) {
        float_t factor = 1.0f /   predicted.size();
        for(size_t i = 0, len = predicted.size(); i < len; ++i) {
            float_t diff = (predicted[i] - observed[i]);
            if(diff < 0.0f) {
                result[i] = -factor;
            } else if(diff > 0.0f) {
                result[i] = factor;
            } else {
                result[i] = 0;
            }
        }
    }

    float_t f(    const vec_t& predictions,
                const vec_t& observed) {
        float_t sum = 0;
        for(size_t i = 0, len = predictions.size(); i < len; ++i) {
            sum += std::abs(predictions[i] - observed[i]);
        }
        return sum;
    }
};

/**
 * Mean Square Error loss function
 */
struct mse_loss {

    void df(     const vec_t& predicted,
                const vec_t& observed,
                vec_t& result) {
        float_t factor = 2.0f / predicted.size();
        for(size_t i = 0, len = predicted.size(); i < len; ++i) {
            result[i] = factor * (predicted[i] - observed[i]);
        }
    }

    float_t f(   const vec_t& predicted,
                const vec_t& observed) {
        float_t sum = 0;
        for(size_t i = 0, len = predicted.size(); i < len; ++i) {
            sum += (predicted[i] - observed[i]) * (predicted[i] - observed[i]);
        }
        return sum / predicted.size();
    }
};

} /* end namespace mlp */

#endif /* MLP_LOSS_HPP */

