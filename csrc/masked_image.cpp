#include "masked_image.h"

const int MaskedImage::kDistanceScale = 65535;
const int MaskedImage::kSSDScale = 9 * 255 * 255;
const cv::Size MaskedImage::kDownsampleKernelSize = cv::Size(6, 6);
const int MaskedImage::kDownsampleKernel[6] = {1, 5, 10, 10, 5, 1};

bool MaskedImage::contains_mask(int y, int x, int patch_size) const {
    auto mask_size = size();
    for (int dy = -patch_size; dy <= patch_size; ++dy) {
        for (int dx = -patch_size; dx <= patch_size; ++dx) {
            int yy = y + dy, xx = x + dx;
            if (yy >= 0 && yy < mask_size.height && xx >= 0 && xx < mask_size.width) {
                if (is_masked(yy, xx)) return true;
            }
        }
    }
    return false;
}

MaskedImage MaskedImage::downsample() const {
    const auto &kernel_size = MaskedImage::kDownsampleKernelSize;
    const auto &kernel = MaskedImage::kDownsampleKernel;

    const auto size = this->size();
    const auto new_size = cv::Size(size.width / 2, size.height / 2);

    auto ret = MaskedImage(new_size.width, new_size.height);
    for (int y = 0; y < size.height - 1; y += 2) {
        for (int x = 0; x < size.width - 1; x += 2) {
            int r = 0, g = 0, b = 0, ksum = 0;

            for (int dy = -kernel_size.height / 2 + 1; dy <= kernel_size.height / 2; ++dy) {
                for (int dx = -kernel_size.width / 2 + 1; dx <= kernel_size.width / 2; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy >= 0 && yy < size.height && xx >= 0 && xx < size.width && !is_masked(yy, xx)) {
                        auto source_ptr = get_image(yy, xx);
                        int k = kernel[kernel_size.height / 2 - 1 + dy] * kernel[kernel_size.width / 2 - 1 + dx];
                        r += source_ptr[0] * k, g += source_ptr[1] * k, b += source_ptr[2] * k;
                        ksum += k;
                    }
                }
            }

            if (ksum > 0) r /= ksum, g /= ksum, b /= ksum;


            if (ksum > 0) {
                auto target_ptr = ret.get_mutable_image(y / 2, x / 2);
                target_ptr[0] = r, target_ptr[1] = g, target_ptr[2] = b;
                ret.set_mask(y / 2, x / 2, 0);
            } else {
                ret.set_mask(y / 2, x / 2, 1);
            }
        }
    }

    return ret;
}

MaskedImage MaskedImage::upsample(int new_w, int new_h) const {
    const auto size = this->size();
    auto ret = MaskedImage(new_w, new_h);
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int yy = y * size.height / new_h;
            int xx = x * size.width / new_w;

            if (is_masked(yy, xx)) {
                ret.set_mask(y, x, 1);
            } else {
                auto source_ptr = get_image(yy, xx);
                auto target_ptr = ret.get_mutable_image(y, x);
                for (int c = 0; c < 3; ++c)
                    target_ptr[c] = source_ptr[c];
                ret.set_mask(y, x, 0);
            }
        }
    }

    return ret;
}

int distance_masked_images(
        const MaskedImage &source, int ys, int xs,
        const MaskedImage &target, int yt, int xt,
        int patch_size
) {
    long double distance = 0;
    long double wsum = 0;

    auto source_size = source.size();
    auto target_size = target.size();

    for (int dy = -patch_size; dy <= patch_size; ++dy) {
        for (int dx = -patch_size; dx <= patch_size; ++dx) {
            int yys = ys + dy, xxs = xs + dx;
            int yyt = yt + dy, xxt = xt + dx;
            wsum += 1;

            if ((yys <= 0 || yys >= source_size.height - 1 || xxs <= 0 || xxs >= source_size.width - 1) ||
                (yyt <= 0 || yyt >= target_size.height - 1 || xxt <= 0 || xxt >= target_size.width - 1) ||
                (source.is_masked(yys, xxs) || target.is_masked(yyt, xxt))) {
                distance += 1;
                continue;
            }

            long double ssd = 0;
            for (int c = 0; c < 3; ++c) {
                auto s_value = source.get_image_int(yys, xxs, c), t_value = target.get_image_int(yyt, xxt, c);
                auto s_gx = 128 + (source.get_image_int(yys+1, xxs, c) - source.get_image_int(yys-1, xxs, c)) / 2;
                auto s_gy = 128 + (source.get_image_int(yys, xxs+1, c) - source.get_image_int(yys, xxs-1, c)) / 2;
                auto t_gx = 128 + (source.get_image_int(yyt+1, xxt, c) - source.get_image_int(yyt-1, xxt, c)) / 2;
                auto t_gy = 128 + (source.get_image_int(yyt, xxt+1, c) - source.get_image_int(yyt, xxt-1, c)) / 2;

                ssd += pow((long double) s_value - t_value, 2);
                ssd += pow((long double) s_gx - t_gx, 2);
                ssd += pow((long double) s_gy - t_gy, 2);
            }

            distance += ssd / (long double)(MaskedImage::kSSDScale);
        }
    }

    int res = int(MaskedImage::kDistanceScale * distance / wsum);
    if (res < 0 || res > MaskedImage::kDistanceScale) return MaskedImage::kDistanceScale;
    return res;
}

