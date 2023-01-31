#pragma once

#include <memory.h>
#include <src/DistanceMetrics.h>

namespace lpq {

/**
 * Exact search index stores all the vectors and performs
 * exhaustive search to compute distances
 **/

class ExactSearchIndex : public std::enable_shared_from_this<ExactSearchIndex> {
public:
  explicit ExactSearchIndex(
      DistanceMetric metric = DistanceMetric::InnerProduct);

    template<typename PRECISION_TYPE>
    void addDataset(const std::vector<std::vector<PRECISION_TYPE>>& dataset);


}

} // namespace lpq