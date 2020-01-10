#pragma once

#include <vector>

#include "masked_image.h"
#include "nnf.h"

class Inpainting {
public:
    Inpainting(cv::Mat image, cv::Mat mask, int patch_size);
    cv::Mat run(bool verbose = false);

private:
    MaskedImage _expectation_maximization(MaskedImage source, MaskedImage target, int level, bool verbose);
    void _expectation_step(const NearestNeighborField &nnf, bool source2target, cv::Mat &vote, const MaskedImage &source, bool upscaled);
    void _maximization_step(MaskedImage &target, const cv::Mat &vote);
    void _weighted_copy(const MaskedImage &source, int ys, int xs, cv::Mat &target, int yt, int xt, double weight);

    MaskedImage m_initial;
    std::vector<MaskedImage> m_pyramid;

    NearestNeighborField m_source2target;
    NearestNeighborField m_target2source;
    int m_patch_size;
};

