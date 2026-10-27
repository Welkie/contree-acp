#include "configuration.h"
#include <algorithm>

Configuration Configuration::GetLeftSubtreeConfig() const {
	Configuration left_config(*this);
	left_config.max_depth -= 1;
	left_config.max_gap = (max_gap - (max_gap+1)/2) / 2;
	left_config.is_root = false;
	return left_config;
}

Configuration Configuration::GetRightSubtreeConfig(int left_gap) const {
	Configuration right_config(*this);
	right_config.max_depth -= 1;
	right_config.max_gap = (max_gap - (max_gap + 1) / 2) - left_gap;
	right_config.is_root = false;
	return right_config;
}

double Configuration::penalty_factor(int feature) const {
	if (!uses_acp()) return 1.0;
	double factor = 1.0;
	for (int ancestor : ancestors) {
		if (ancestor == feature) continue; // no self-penalty, so a feature can split again further down
		factor *= 1.0 - acp_alpha * (*acp_abs_correlation)[size_t(feature) * acp_num_features + ancestor];
	}
	return factor;
}

void Configuration::add_ancestor(int feature) {
	auto it = std::lower_bound(ancestors.begin(), ancestors.end(), feature);
	if (it == ancestors.end() || *it != feature) ancestors.insert(it, feature);
}

std::vector<double> Configuration::child_penalty_factors(int feature) const {
	if (!uses_acp()) return {};
	Configuration child(*this);
	child.add_ancestor(feature);
	std::vector<double> factors(acp_num_features);
	for (int f = 0; f < acp_num_features; f++) factors[f] = child.penalty_factor(f);
	return factors;
}