#include "nnf.h"
#include "masked_image.h"
#include <algorithm>

/**
* Nearest-Neighbor Field (see PatchMatch algorithm)
*  This algorithme uses a version proposed by Xavier Philippeau
*
*/

template <typename T>
T clamp(T value, T min_value, T max_value) {
    return std::min(std::max(value, min_value), max_value);
}

void NearestNeighborField::_randomize_field(int max_retry, bool reset) {
    auto this_size = source_size();
    for (int i = 0; i < this_size.height; ++i) {
        for (int j = 0; j < this_size.width; ++j) {
            int i_target = 0, j_target = 0;
            int distance = reset ? MaskedImage::kDistanceScale : m_field.at<int>(i, j, 2);

            if (distance < MaskedImage::kDistanceScale) {
                continue;
            }

            for (int t = 0; t < max_retry; ++t) {
                i_target = rand() % this_size.height + 1;
                j_target = rand() % this_size.width + 1;
                distance = _distance(i, j, i_target, j_target);
                if (distance < MaskedImage::kDistanceScale)
                    break;
            }

            m_field.at<int>(i, j, 0) = i_target;
            m_field.at<int>(i, j, 1) = j_target;
            m_field.at<int>(i, j, 2) = distance;
        }
    }
}

void NearestNeighborField::_initialize_field_from(const NearestNeighborField &other, int max_retry) {
    const auto &this_size = source_size();
    const auto &other_size = other.source_size();
    double fi = static_cast<double>(this_size.height) / other_size.height;
    double fj = static_cast<double>(this_size.width) / other_size.width;

    for (int i = 0; i < this_size.height; ++i) {
        for (int j = 0; j < this_size.width; ++j) {
            int ilow = static_cast<int>(std::min(i / fi, static_cast<double>(other_size.height - 1)));
            int jlow = static_cast<int>(std::min(j / fj, static_cast<double>(other_size.width - 1)));
            int this_field_i, this_field_j;
            at(i, j, 0) = this_field_i = static_cast<int>(other.at(ilow, jlow, 0) * fi);
            at(i, j, 1) = this_field_j = static_cast<int>(other.at(ilow, jlow, 1) * fj);
            at(i, j, 2) = _distance(i, j, this_field_i, this_field_j);
        }
    }

    _randomize_field(max_retry, false);
}

void NearestNeighborField::minimize(int nr_pass) {
    const auto &this_size = source_size();
    while (nr_pass--) {
        for (int i = 0; i < this_size.height; ++i)
            for (int j = 0; j < this_size.width; ++j)
                if (at(i, j, 2) > 0) _minimize_link(i, j, +1);
        for (int i = this_size.height - 1; i >= 0; --i)
            for (int j = this_size.width - 1; j >= 0; --j)
                if (at(i, j, 2) > 0) _minimize_link(i, j, -1);
    }
}

void NearestNeighborField::_minimize_link(int y, int x, int direction) {
    const auto &this_size = source_size();
    const auto &this_target_size = target_size();

    // propagation along the y direction.
    if (y - direction >= 0 && y - direction < this_size.height) {
        int yp = at(y - direction, x, 0) + direction;
        int xp = at(y - direction, x, 1);
        int dp = _distance(y, x, yp, xp);
        if (dp < at(y, x, 2)) {
            at(y, x, 0) = yp, at(y, x, 1) = xp, at(y, x, 2) = dp;
        }
    }

    // propagation along the x direction.
    if (x - direction >= 0 && x - direction < this_size.width) {
        int yp = at(y, x - direction, 0);
        int xp = at(y, x - direction, 1) + direction;
        int dp = _distance(y, x, yp, xp);
        if (dp < at(y, x, 2)) {
            at(y, x, 0) = yp, at(y, x, 1) = xp, at(y, x, 2) = dp;
        }
    }

    int yp_current = at(y, x, 0), xp_current = at(y, x, 1);
    int random_scale = std::min(this_target_size.height, this_target_size.width);
    // TODO:: optimize this.
    while (random_scale > 0) {
        int yp = yp_current + (rand() % (2 * random_scale) - random_scale);
        int xp = xp_current + (rand() % (2 * random_scale) - random_scale);
        yp = clamp(yp, 0, target_size().height - 1);
        xp = clamp(xp, 0, target_size().width - 1);
        int dp = _distance(y, x, yp, xp);
        if (dp < at(y, x, 2)) {
            at(y, x, 0) = yp, at(y, x, 1) = xp, at(y, x, 2) = dp;
        }
        random_scale /= 2;
    }
}

