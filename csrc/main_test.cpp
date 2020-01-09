#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "masked_image.h"
#include "nnf.h"
#include "inpaint.h"

int main() {
    cv::Mat source = cv::imread("./examples/forest_pruned.bmp");
    cv::Mat mask = cv::Mat(source.size(), CV_8UC1);
    mask = cv::Scalar::all(0);

    for (int i = 0; i < source.size().height; ++i) {
        for (int j = 0; j < source.size().width; ++i) {
            if (source.at<unsigned char>(i, j, 0) == 255 && source.at<unsigned char>(i, j, 1) == 255 && source.at<unsigned char>(i, j, 2) == 255) {
                mask.at<unsigned char>(i, j, 0) = 1;
            }
        }
    }
    
    cv::Mat result = Inpainting(source, mask, 15).run();
    cv::imshow("Result", result);
    
    return 0;
}