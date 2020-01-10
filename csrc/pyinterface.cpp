#include "pyinterface.h"
#include "inpaint.h"

int _dtype_py_to_cv(int dtype_py);
int _dtype_cv_to_py(int dtype_cv);
cv::Mat _py_to_cv2(PM_mat_t pymat);
PM_mat_t _cv2_to_py(cv::Mat cvmat);

void PM_free_pymat(PM_mat_t pymat) {
    free(pymat.data_ptr);
}

PM_mat_t PM_inpaint(PM_mat_t source_py, PM_mat_t mask_py, int patch_size) {
    cv::Mat source = _py_to_cv2(source_py);
    cv::Mat mask = _py_to_cv2(mask_py);
    return _cv2_to_py(Inpainting(source, mask, patch_size).run(false));
}


int _dtype_py_to_cv(int dtype_py) {
    switch (dtype_py) {
        case PM_UINT8: return CV_8U;
        case PM_INT8: return CV_8S;
        case PM_UINT16: return CV_16U;
        case PM_INT16: return CV_16S;
        case PM_INT32: return CV_32S;
        case PM_FLOAT32: return CV_32F;
        case PM_FLOAT64: return CV_64F;
    }

    return CV_8U;
}

int _dtype_cv_to_py(int dtype_cv) {
    switch (dtype_cv) {
        case CV_8U: return PM_UINT8;
        case CV_8S: return PM_INT8;
        case CV_16U: return PM_UINT16;
        case CV_16S: return PM_INT16;
        case CV_32S: return PM_INT32;
        case CV_32F: return PM_FLOAT32;
        case CV_64F: return PM_FLOAT64;
    }

    return PM_UINT8;
}

cv::Mat _py_to_cv2(PM_mat_t pymat) {
    int dtype = _dtype_py_to_cv(pymat.dtype);
    dtype = CV_MAKETYPE(pymat.dtype, pymat.shape.channels);
    return cv::Mat(cv::Size(pymat.shape.width, pymat.shape.height), dtype, pymat.data_ptr).clone();
}

PM_mat_t _cv2_to_py(cv::Mat cvmat) {
    PM_shape_t shape = {cvmat.size().width, cvmat.size().height, cvmat.channels()};
    int dtype = _dtype_cv_to_py(cvmat.depth());
    size_t dsize = cvmat.total() * cvmat.elemSize();

    void *data_ptr = reinterpret_cast<void *>(malloc(dsize));
    memcpy(data_ptr, reinterpret_cast<void *>(cvmat.data), dsize);

    return PM_mat_t {data_ptr, shape, dtype};
}

