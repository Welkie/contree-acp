#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "stopwatch.h"
#include "statistics.h"
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <vector>

#define EPSILON 0.0000001f

// ACP: splits whose penalty factor is at or below this value can never beat the leaf, so they are skipped
#define ACP_MIN_FACTOR 1e-12

#define PRINT_INTERMEDIARY_TIME_SOLUTIONS 0

#ifdef NDEBUG
#define RUNTIME_ASSERT(cond, msg) ((void)0)
#else
#define RUNTIME_ASSERT(cond, msg)                                             \
        do {                                                                      \
            if (!(cond)) {                                                        \
                std::cerr << "Assertion failed: " << #cond                        \
                          << "\nMessage: " << msg                                 \
                          << "\nFile: " << __FILE__                               \
                          << "\nLine: " << __LINE__ << std::endl;                 \
                std::terminate();                                                 \
            }                                                                     \
        } while (0)
#endif

struct Configuration {

    Configuration() : stats(std::make_shared<statistics>()) {}

    int max_depth{ 3 };
    int max_gap{ 0 };
    float max_gap_decay{ 0.0 };
    bool print_logs{ false };
    bool use_upper_bound{ true };
    bool is_root{ false };
    bool sort_gini{ false };
    float complexity_cost{ 0.0 };
    Stopwatch stopwatch;
    std::shared_ptr<statistics> stats{ nullptr };
    Configuration GetLeftSubtreeConfig() const;
    Configuration GetRightSubtreeConfig(int left_gap) const;

    // ACP (ancestor-correlation penalty). A split on feature f at a node whose root-to-node path uses the
    // features in `ancestors` keeps only the share P(f, A) = prod_{p in A, p != f} (1 - alpha |r(f, p)|)
    // of the misclassifications it removes. alpha = 0 gives the original ConTree objective.
    double acp_alpha{ 0.0 };
    int acp_num_features{ 0 };
    std::shared_ptr<const std::vector<double>> acp_abs_correlation{ nullptr }; // row-major |r|, num_features x num_features
    std::vector<int> ancestors; // sorted, distinct features used on the path to this node

    bool uses_acp() const { return acp_alpha > 0.0 && acp_abs_correlation != nullptr; }
    double penalty_factor(int feature) const;
    void add_ancestor(int feature);
    // P(g, A + {feature}) for every feature g, used by the depth-two solver for the splits below `feature`.
    // Empty when ACP is off.
    std::vector<double> child_penalty_factors(int feature) const;
    // Cached subtrees are only valid for the same ancestor set when ACP is on
    std::vector<int> cache_key() const { return uses_acp() ? ancestors : std::vector<int>(); }
};

#endif // CONFIGURATION_H
