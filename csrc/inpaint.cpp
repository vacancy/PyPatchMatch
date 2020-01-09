#include "inpaint.h"
#include <algorithm>
#include <vector>

namespace {
    static std::vector<double> kDistance2Similarity;

    void init_kDistance2Similarity() {
        double base[11] = {1.0, 0.99, 0.96, 0.83, 0.38, 0.11, 0.02, 0.005, 0.0006, 0.0001, 0};
        int length = (MaskedImage::kDistanceScale + 1);
        kDistance2Similarity.resize(length);
        for (int i = 0; i < length; ++i) {
            double t = (double) i / length;
            int j = (int) (100 * t);
            int k = j + 1;
            double vj = (j < 11) ? base[j] : 0;
            double vk = (k < 11) ? base[k] : 0;
            kDistance2Similarity[i] = vj + (100 * t - j) * (vk - vj);
        }
    }
}

Inpainting::Inpainting(cv::Mat image, cv::Mat mask, int patch_size)
    : m_initial(image, mask), m_patch_size(patch_size), m_pyramid(), m_source2target(), m_target2source() {

    auto source = m_initial;
    while (source.size().height > m_patch_size && source.size().width > m_patch_size) {
        source = source.downsample();
        m_pyramid.push_back(source);
    }

    if (kDistance2Similarity.size() == 0) {
        init_kDistance2Similarity();
    }
}

cv::Mat Inpainting::run() {
    const int nr_levels = m_pyramid.size();

    MaskedImage source, target;
    for (int level = nr_levels - 1; level > 0; --level) {
        source = m_pyramid[level];
        if (level == nr_levels - 1) {
            target = source.clone();
            target.clear_mask();

            m_source2target = NearestNeighborField(source, target, m_patch_size);
            m_target2source = NearestNeighborField(target, source, m_patch_size);
        } else {
            m_source2target = NearestNeighborField(source, target, m_patch_size, m_source2target);
            m_target2source = NearestNeighborField(target, source, m_patch_size, m_target2source);
        }

        target = _expectation_maximization(source, target, level);
    }

    return target.image();
}

MaskedImage Inpainting::_expectation_maximization(MaskedImage source, MaskedImage target, int level) {
    const int nr_iters_em = 1 + 2 * level;
    const int nr_iters_nnf = static_cast<int>(std::min(7, 1 + level));

    MaskedImage new_source, new_target;

    for (int iter_em = 0; iter_em < nr_iters_em; ++iter_em) {
        if (iter_em != 0) {
            m_source2target.set_target(new_target);
            m_target2source.set_source(new_source);
            target = new_target;  // TODO:: move?
        }

        auto size = source.size();
        for (int i = 0; i < size.height; ++i) {
            for (int j = 0; j < size.width; ++j) {
                if (!source.contains_mask(i, j, m_patch_size)) {
                    m_source2target.set_identity(i, j);
                    m_target2source.set_identity(i, j);
                }
            }
        }
        m_source2target.minimize(nr_iters_nnf);
        m_target2source.minimize(nr_iters_nnf);

        bool upscaled = false;
        if (level >= 1 && iter_em == nr_iters_em - 1) {
            new_source = m_pyramid[level - 1];
            new_target = target.upsample(new_source.size().width, new_source.size().height);
            upscaled = true;
        } else {
            new_source = m_pyramid[level];
            new_target = target.clone();  // TODO:: why we need this clone?
        }

        auto vote = cv::Mat(new_target.size(), CV_64FC4);
        vote.setTo(cv::Scalar(0));
        _expectation_step(m_source2target, 1, vote, new_source, upscaled);
        _expectation_step(m_target2source, 0, vote, new_source, upscaled);
        _maximization_step(new_target, vote);
    }

    return new_target;
}

void Inpainting::_expectation_step(
    const NearestNeighborField &nnf, bool source2target,
    cv::Mat &vote, const MaskedImage &source, bool upscaled
) {
    auto source_size = nnf.source_size();
    auto target_size = nnf.target_size();

    for (int i = 0; i < source.size().height; ++i) {
        for (int j = 0; j< source.size().width; ++j) {
            int yp = nnf.at(i, j, 0), xp = nnf.at(i, j, 1), dp = nnf.at(i, j, 2);
            double w = kDistance2Similarity[dp];

            for (int di = -m_patch_size; di <= m_patch_size; ++di) {
                for (int dj = -m_patch_size; dj <= m_patch_size; ++dj) {
                    int ys = i + di, xs = j + dj, yt = yp + di, xt = xp + dj;
                    cv::Size sizes, sizet;

                    if (!(ys >= 0 && ys < source_size.height && xs >= 0 && xs < source_size.width)) continue;
                    if (!(yt >= 0 && yt < target_size.height && xt >= 0 && xt < source_size.width)) continue;

                    if (!source2target) {
                        std::swap(ys, yt);
                        std::swap(xs, xt);
                    }

                    if (upscaled) {
                        for (int uy = 0; uy < 2; ++uy) {
                            for (int ux = 0; ux < 2; ++ux) {
                                _weighted_copy(source, 2 * ys + uy, 2 * xs + ux, vote, 2 * yt + uy, 2 * xt + ux, w);
                            }
                        }
                    } else {
                        _weighted_copy(source, ys, xs, vote, yt, xt, w);
                    }
                }
            }
        }
    }
}

void Inpainting::_maximization_step(MaskedImage &target, const cv::Mat &vote) {
    auto target_size = target.size();
    for (int i = 0; i < target_size.height; ++i) {
        for (int j = 0; j < target_size.width; ++j) {
            if (vote.at<double>(i, j, 3) > 0) {
                unsigned char r = cv::saturate_cast<unsigned char>(vote.at<double>(i, j, 0) / vote.at<double>(i, j, 3));
                unsigned char g = cv::saturate_cast<unsigned char>(vote.at<double>(i, j, 1) / vote.at<double>(i, j, 3));
                unsigned char b = cv::saturate_cast<unsigned char>(vote.at<double>(i, j, 2) / vote.at<double>(i, j, 3));

                target.set_image(i, j, 0, r);
                target.set_image(i, j, 1, g);
                target.set_image(i, j, 2, b);
                target.set_mask(i, j, 0);
            }
        }
    }
}

void Inpainting::_weighted_copy(const MaskedImage &source, int ys, int xs, cv::Mat &target, int yt, int xt, double weight) {
    if (source.is_masked(ys, xs)) return;

    for (int c = 0; c < 3; ++c) {
        target.at<double>(yt, xt, c) += static_cast<double>(source.get_image(ys, xs, c)) * weight;
    }
    target.at<double>(yt, xt, 3) += weight;
}
