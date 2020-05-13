#include <algorithm>
#include <iostream>

#include "masked_image.h"
#include "nnf.h"

/**
* Nearest-Neighbor Field (see PatchMatch algorithm).
* This algorithme uses a version proposed by Xavier Philippeau.
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
            auto this_ptr = mutable_ptr(i, j);
            int distance = reset ? MaskedImage::kDistanceScale : this_ptr[2];
            if (distance < MaskedImage::kDistanceScale) {
                continue;
            }

            int i_target = 0, j_target = 0;
            for (int t = 0; t < max_retry; ++t) {
                i_target = rand() % this_size.height;
                j_target = rand() % this_size.width;

                distance = _distance(i, j, i_target, j_target);
                if (distance < MaskedImage::kDistanceScale)
                    break;
            }

            this_ptr[0] = i_target, this_ptr[1] = j_target, this_ptr[2] = distance;
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
            auto this_value = mutable_ptr(i, j);
            auto other_value = other.ptr(ilow, jlow);

            this_value[0] = static_cast<int>(other_value[0] * fi);
            this_value[1] = static_cast<int>(other_value[1] * fj);
            this_value[2] = _distance(i, j, this_value[0], this_value[1]);
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
    auto this_ptr = mutable_ptr(y, x);

    // propagation along the y direction.
    if (y - direction >= 0 && y - direction < this_size.height) {
        int yp = at(y - direction, x, 0) + direction;
        int xp = at(y - direction, x, 1);
        int dp = _distance(y, x, yp, xp);
        if (dp < at(y, x, 2)) {
            this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
        }
    }

    // propagation along the x direction.
    if (x - direction >= 0 && x - direction < this_size.width) {
        int yp = at(y, x - direction, 0);
        int xp = at(y, x - direction, 1) + direction;
        int dp = _distance(y, x, yp, xp);
        if (dp < at(y, x, 2)) {
            this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
        }
    }

    // random search with a progressive step size.
    int random_scale = (std::min(this_target_size.height, this_target_size.width) - 1) / 2;
    while (random_scale > 0) {
        int yp = this_ptr[0] + (rand() % (2 * random_scale + 1) - random_scale);
        int xp = this_ptr[1] + (rand() % (2 * random_scale + 1) - random_scale);
        yp = clamp(yp, 0, target_size().height - 1);
        xp = clamp(xp, 0, target_size().width - 1);
        int dp = _distance(y, x, yp, xp);
        if (dp < at(y, x, 2)) {
            this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
        }
        random_scale /= 2;
    }
}

