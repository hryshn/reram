/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <string>
#include <vector>

#include "puma.h"
#include "conv-layer.h"
#include "fully-connected-layer.h"

void isolated_fully_connected_layer(Model model, std::string layerName, unsigned int in_size, unsigned int out_size) {

    // Input vector
    auto in = InputVector::create(model, "in", in_size);

    // Output vector
    auto out = OutputVector::create(model, "out", out_size);

    // Layer
    out = fully_connected_layer(model, layerName, in_size, out_size, in);

}

int main() {

    Model model = Model::create("vgg16");

    // Input
    unsigned int in_size_x = 227;
    unsigned int in_size_y = 227;
    unsigned int in_channels = 3;
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    // Layer 1 (convolution with max pool) configurations
    unsigned int k_size_x1 = 11;
    unsigned int k_size_y1 = 11;
    unsigned int in_size_x1 = 227;
    unsigned int in_size_y1 = 227;
    unsigned int in_channels1 = 3;
    unsigned int out_channels1 = 96;
    unsigned int max_pool_size_x1 = 3;
    unsigned int max_pool_size_y1 = 3;

    // Layer 2 (convolution with max pool) configurations
    unsigned int k_size_x2 = 5;
    unsigned int k_size_y2 = 5;
    unsigned int in_size_x2 = 27;
    unsigned int in_size_y2 = 27;
    unsigned int in_channels2 = 96;
    unsigned int out_channels2 = 256;
    unsigned int max_pool_size_x2 = 3;
    unsigned int max_pool_size_y2 = 3;

    // Layer 3 (convolution) configurations
    unsigned int k_size_x3 = 3;
    unsigned int k_size_y3 = 3;
    unsigned int in_size_x3 = 13;
    unsigned int in_size_y3 = 13;
    unsigned int in_channels3 = 256;
    unsigned int out_channels3 = 384;

    // Layer 4 (convolution) configurations
    unsigned int k_size_x4 = 3;
    unsigned int k_size_y4 = 3;
    unsigned int in_size_x4 = 13;
    unsigned int in_size_y4 = 13;
    unsigned int in_channels4 = 384;
    unsigned int out_channels4 = 384;

    // Layer 5 (convolution) configurations
    unsigned int k_size_x5 = 3;
    unsigned int k_size_y5 = 3;
    unsigned int in_size_x5 = 13;
    unsigned int in_size_y5 = 13;
    unsigned int in_channels5 = 384;
    unsigned int out_channels5 = 256;
    unsigned int max_pool_size_x5 = 3;
    unsigned int max_pool_size_y5 = 3;

    // Output
    unsigned int out_size_x = 6;
    unsigned int out_size_y = 6;
    unsigned int out_channels = 256;
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, out_channels);

    // Layer 6 (fully-connected) configurations
    unsigned int in_size6 = 9216;
    unsigned int out_size6 = 4096;

    // Layer 7 (fully-connected) configurations
    unsigned int in_size7 = 4096;
    unsigned int out_size7 = 4096;

    // Layer 8 (fully-connected) configurations
    unsigned int in_size8 = 4096;
    unsigned int out_size8 = 1000;

    // Define network
    auto out1 = convmax_layer(model, "layer" + std::to_string(1), k_size_x1, k_size_y1, in_size_x1, in_size_y1, in_channels1, out_channels1, max_pool_size_x1, max_pool_size_y1, in_stream);
    auto out2 = convmax_layer(model, "layer" + std::to_string(2), k_size_x2, k_size_y2, in_size_x2, in_size_y2, in_channels2, out_channels2, max_pool_size_x2, max_pool_size_y2, out1);
    auto out3 = conv_layer(model, "layer" + std::to_string(3), k_size_x3, k_size_y3, in_size_x3, in_size_y3, in_channels3, out_channels3, out2);
    auto out4 = conv_layer(model, "layer" + std::to_string(4), k_size_x4, k_size_y4, in_size_x4, in_size_y4, in_channels4, out_channels4, out3);
    auto out5 = convmax_layer(model, "layer" + std::to_string(5), k_size_x5, k_size_y5, in_size_x5, in_size_y5, in_channels5, out_channels5, max_pool_size_x5, max_pool_size_y5, out4);
    out_stream = out5;
    isolated_fully_connected_layer(model, "layer" + std::to_string(6), in_size6, out_size6);
    isolated_fully_connected_layer(model, "layer" + std::to_string(7), in_size7, out_size7);
    isolated_fully_connected_layer(model, "layer" + std::to_string(8), in_size8, out_size8);

    // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;

}

// stride and pad?