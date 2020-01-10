#pragma once

#include <opencv2/core.hpp>
#include "masked_image.h"

class NearestNeighborField {
public:
    NearestNeighborField() : m_source(), m_target(), m_field(), m_patch_size() {
        // pass
    }
    NearestNeighborField(const MaskedImage &source, const MaskedImage &target, int patch_size, int max_retry = 20)
        : m_source(source), m_target(target), m_patch_size(patch_size) {
        m_field = cv::Mat(m_source.size(), CV_32SC3);
        _randomize_field(max_retry);
    }
    NearestNeighborField(const MaskedImage &source, const MaskedImage &target, int patch_size, const NearestNeighborField &other, int max_retry = 20)
            : m_source(source), m_target(target), m_patch_size(patch_size) {
        m_field = cv::Mat(m_source.size(), CV_32SC3);
        _initialize_field_from(other, max_retry);
    }

    inline cv::Size source_size() const {
        return m_source.size();
    }
    inline cv::Size target_size() const {
        return m_target.size();
    }
    inline void set_source(const MaskedImage &source) {
        m_source = source;
    }
    inline void set_target(const MaskedImage &target) {
        m_target = target;
    }

    inline int *mutable_ptr(int y, int x) {
        return m_field.ptr<int>(y, x);
    }
    inline const int *ptr(int y, int x) const {
        return m_field.ptr<int>(y, x);
    }

    inline int at(int y, int x, int c) const {
        return m_field.ptr<int>(y, x)[c];
    }
    inline int &at(int y, int x, int c) {
        return m_field.ptr<int>(y, x)[c];
    }
    inline void set_identity(int y, int x) {
        auto ptr = mutable_ptr(y, x);
        ptr[0] = y, ptr[1] = x, ptr[2] = 0;
    }

    void minimize(int nr_pass);

private:
    inline int _distance(int source_y, int source_x, int target_y, int target_x) {
        return distance_masked_images(m_source, source_y, source_x, m_target, target_y, target_x, m_patch_size);
    }

    void _randomize_field(int max_retry = 20, bool reset = true);
    void _initialize_field_from(const NearestNeighborField &other, int max_retry);
    void _minimize_link(int y, int x, int direction);

    MaskedImage m_source;
    MaskedImage m_target;
    cv::Mat m_field;  // { y_target, x_target, distance_scaled }
    int m_patch_size;
};

