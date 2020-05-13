#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "masked_image.h"
#include "nnf.h"
#include "inpaint.h"

int main() {
    auto source = cv::imread("./images/forest_pruned.bmp", cv::IMREAD_COLOR);

    auto mask = cv::Mat(source.size(), CV_8UC1);
    mask = cv::Scalar::all(0);
    for (int i = 0; i < source.size().height; ++i) {
        for (int j = 0; j < source.size().width; ++j) {
            auto source_ptr = source.ptr<unsigned char>(i, j);
            if (source_ptr[0] == 255 && source_ptr[1] == 255 && source_ptr[2] == 255) {
                mask.at<unsigned char>(i, j) = 1;
            }
        }
    }

    auto result = Inpainting(source, mask, 3).run(true);
    cv::imwrite("./images/forest_recovered.bmp", result);
    // cv::imshow("Result", result);
    // cv::waitKey();

    return 0;
}

