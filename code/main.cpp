#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "cache.h"
#include "configuration.h"
#include "dataset.h"
#include "dataview.h"
#include "file_reader.h"
#include "general_solver.h"
#include "parameter_handler.h"
#include "statistics.h"
#include "tree.h"

// Absolute Pearson correlation between every pair of features (row-major), computed once on the training data.
// Constant features get correlation zero.
std::shared_ptr<const std::vector<double>> compute_abs_correlation(const Dataset& data) {
    const int num_features = data.get_features_size();
    const int n = data.get_instance_number();
    std::vector<std::vector<double>> centered(num_features, std::vector<double>(n));
    std::vector<double> norm(num_features, 0.0);
    for (int f = 0; f < num_features; f++) {
        double mean = 0.0;
        for (const auto& element : data.get_feature(f)) mean += element.value;
        mean /= n;
        for (const auto& element : data.get_feature(f)) {
            const double centered_value = element.value - mean;
            centered[f][element.data_point_index] = centered_value;
            norm[f] += centered_value * centered_value;
        }
        norm[f] = std::sqrt(norm[f]);
    }
    auto correlation = std::make_shared<std::vector<double>>(size_t(num_features) * num_features, 0.0);
    for (int f = 0; f < num_features; f++) {
        for (int g = 0; g < num_features; g++) {
            if (f == g) { (*correlation)[size_t(f) * num_features + g] = 1.0; continue; }
            if (norm[f] == 0.0 || norm[g] == 0.0) continue;
            double dot = 0.0;
            for (int i = 0; i < n; i++) dot += centered[f][i] * centered[g][i];
            (*correlation)[size_t(f) * num_features + g] = std::min(1.0, std::abs(dot) / (norm[f] * norm[g]));
        }
    }
    return correlation;
}

void create_optimal_decision_tree(std::string file_name, int run_number, Configuration& config, double runtime_limit) {
    double total_time = 0;
    int instance_number = -1;
    std::shared_ptr<Tree> optimal_decision_tree;

    Dataset unsorted_dataset{}; int class_number = -1;
    file_reader::read_file(file_name, unsorted_dataset, class_number);
    config.complexity_cost *= unsorted_dataset.get_instance_number();
    if (config.acp_alpha > 0.0) {
        config.acp_num_features = unsorted_dataset.get_features_size();
        config.acp_abs_correlation = compute_abs_correlation(unsorted_dataset);
    }
    bool within_time_limit = true;

    for (int run = 0; run < run_number; run++) {
        
        Dataset sorted_dataset = unsorted_dataset;        

        config.stopwatch.Initialise(runtime_limit);

        sorted_dataset.sort_feature_values();

        Dataview dataview = Dataview(&sorted_dataset, &unsorted_dataset, class_number, config.sort_gini);

        optimal_decision_tree = std::make_shared<Tree>();
        int max_gap = config.max_gap;
        do {
            Cache::global_cache = Cache(config.max_depth, unsorted_dataset.get_instance_number());
            config.is_root = true;
            GeneralSolver::create_optimal_decision_tree(dataview, config, optimal_decision_tree, INT_MAX);

            max_gap = int(max_gap * config.max_gap_decay);
        } while (max_gap > 0);

        total_time += config.stopwatch.TimeElapsedInSeconds();
        instance_number = dataview.get_dataset_size();
        within_time_limit = within_time_limit && config.stopwatch.IsWithinTimeLimit();
        if (config.print_logs && !config.stopwatch.IsWithinTimeLimit()) {
            std::cout << std::endl << "The search was stopped because of a time-out. The tree is possibly not optimal." << std::endl << std::endl;
        }
    }

    double average_time = total_time / run_number;
    int check_misclassification_score = optimal_decision_tree->misclassification_score(unsorted_dataset);
    // With ACP the objective is the penalized cost, so the misclassifications are only checked against it without ACP
    int misclassification_score = config.uses_acp()
        ? check_misclassification_score
        : int(std::round(optimal_decision_tree->objective - optimal_decision_tree->get_num_branching_nodes() * config.complexity_cost));
    std::cout << "Objective: " << std::setprecision(10) << optimal_decision_tree->objective << std::endl;
    if (check_misclassification_score == misclassification_score) {
        std::cout << "Misclassification score: " << misclassification_score << std::endl;
    } else {
        std::cout << "Reported misclassification score: " << misclassification_score << std::endl;
        std::cout << "Observed misclassification score: " << check_misclassification_score << std::endl;
        std::cout << "Error! reported and observed misclassification score do not match! Please report a bug!" << std::endl;
    }
    RUNTIME_ASSERT(misclassification_score == check_misclassification_score, "The reported misclassification score should be the same as what the tree predicts.");
    std::cout << "ACP alpha: " << config.acp_alpha << std::endl;
    std::cout << "Within time limit: " << (within_time_limit ? 1 : 0) << std::endl;
    std::cout << "Accuracy: " << ((double) instance_number - misclassification_score) / (double) instance_number << std::endl;
    std::cout << "Number of branching nodes: " << optimal_decision_tree->get_num_branching_nodes() << std::endl;
    std::cout << "Average time taken to get the decision tree: " << std::fixed << std::setprecision(4) << average_time << " seconds" << std::endl;
    std::cout << "Optimal tree: " << std::fixed << std::setprecision(8) << *optimal_decision_tree << std::endl;

    if (config.print_logs) {
        config.stats->print_statistics();
    }

}

int main(int argc, char *argv[]) {
    ParameterHandler parameters = ParameterHandler::DefineParameters();

    bool verbose = true;
    if (argc <= 1) {
        std::cout << "No parameters specified." << std::endl << std::endl;
        parameters.PrintHelpSummary();
        exit(1);
    }
    parameters.ParseCommandLineArguments(argc, argv);

    Configuration config;
    const std::string file = parameters.GetStringParameter("file");
    const int NUM_RUNS = int(parameters.GetIntegerParameter("run-number"));
    const double runtime_limit = parameters.GetFloatParameter("time");
    config.print_logs = parameters.GetBooleanParameter("print-logs");
    config.max_depth = int(parameters.GetIntegerParameter("max-depth"));
    config.use_upper_bound = parameters.GetBooleanParameter("use-upper-bound");
    config.max_gap = int(parameters.GetIntegerParameter("max-gap"));
    config.max_gap_decay = float(parameters.GetFloatParameter("max-gap-decay"));
    config.sort_gini = parameters.GetBooleanParameter("sort-features-gini-index");
    config.complexity_cost = float(parameters.GetFloatParameter("complexity-cost"));
    config.acp_alpha = parameters.GetFloatParameter("acp-alpha");


    if (config.print_logs) {
        parameters.PrintParameterValues();
    }

    create_optimal_decision_tree(file, NUM_RUNS, config, runtime_limit);

    return 0;
}