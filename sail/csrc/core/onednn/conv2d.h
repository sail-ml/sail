#pragma once

#include <dnnl.hpp>
#include "Tensor.h"
using namespace dnnl;
using dnnl::inner_product_forward;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace sail {

namespace onednn {
// inline unsigned char* DummyData = nullptr;
struct OneDNNConv2DParams {
    memory::dims src_dims;
    memory::dims weight_dims;
    memory::dims bias_dims;
    memory::dims dest_dims;
    memory::dims dilation = {0, 0};
    tag src_tag;
    tag weight_tag;
    tag bias_tag = tag::a;
    tag dest_tag;

    memory::dims strides;
    memory::dims padding;

    OneDNNConv2DParams(memory::dims src_dims, memory::dims weight_dims,
                       memory::dims bias_dims, memory::dims dest_dims,
                       memory::dims strides, memory::dims padding,
                       tag src_tag = tag::nchw, tag weight_tag = tag::oihw,
                       tag dest_tag = tag::nchw)
        : src_dims(src_dims),
          weight_dims(weight_dims),
          bias_dims(bias_dims),
          dest_dims(dest_dims),
          strides(strides),
          padding(padding),
          src_tag(src_tag),
          weight_tag(weight_tag),
          dest_tag(dest_tag) {}

    OneDNNConv2DParams(Tensor& src, Tensor& kernel, TensorShape output_shape,
                       std::vector<long> strides_, std::vector<long> padding_,
                       bool back = false) {
        const memory::dim N = src.get_shape().shape[0];
        const memory::dim I_H = src.get_shape().shape[2];
        const memory::dim I_W = src.get_shape().shape[3];
        const memory::dim K_H = kernel.get_shape().shape[2];
        const memory::dim K_W = kernel.get_shape().shape[3];
        const memory::dim IC = kernel.get_shape().shape[1];
        const memory::dim OC = kernel.get_shape().shape[0];
        const memory::dim pad_1 = padding_[0];
        const memory::dim pad_2 = padding_[1];
        const memory::dim stride_1 = strides_[0];
        const memory::dim stride_2 = strides_[1];

        const memory::dim O_H = output_shape[2];
        const memory::dim O_W = output_shape[3];

        SAIL_CHECK(kernel.get_shape()[1] == src.get_shape()[1],
                   "Output sizes must match, ", kernel.get_shape()[1], " and ",
                   src.get_shape()[1]);

        src_dims = {N, IC, I_H, I_W};
        weight_dims = {OC, IC, K_H, K_W};
        bias_dims = {OC};
        dest_dims = {N, OC, O_H, O_W};
        if (back) {
            dest_dims = {N, IC, O_H, O_W};
        }
        strides = {stride_1, stride_2};
        padding = {pad_1, pad_2};

        src_tag = tag::nchw;
        weight_tag = tag::oihw;
        dest_tag = tag::nchw;
    }
};

class OneDNNConv2D {
   public:
    dnnl::engine engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream = dnnl::stream(engine);
    std::shared_ptr<OneDNNConv2DParams> params;

    std::shared_ptr<memory::desc> src_md = nullptr;
    std::shared_ptr<memory::desc> weight_md = nullptr;
    std::shared_ptr<memory::desc> bias_md = nullptr;
    std::shared_ptr<memory::desc> dest_md = nullptr;

    std::shared_ptr<memory> src_mem = nullptr;
    std::shared_ptr<memory> weight_mem = nullptr;
    std::shared_ptr<memory> bias_mem = nullptr;
    std::shared_ptr<memory> dest_mem = nullptr;

    std::shared_ptr<convolution_forward::desc> convolution_forward_desc =
        nullptr;
    std::shared_ptr<convolution_forward> convolution_forward_prim = nullptr;
    std::unordered_map<int, memory> convolution_forward_args;

    OneDNNConv2D(std::shared_ptr<OneDNNConv2DParams> _params) {
        params = _params;
    }
    convolution_forward::primitive_desc initialize(bool bias = true) {
        src_md.reset(
            new memory::desc(params->src_dims, dt::f32, params->src_tag));

        weight_md.reset(
            new memory::desc(params->weight_dims, dt::f32, params->weight_tag));
        bias_md.reset(
            new memory::desc(params->bias_dims, dt::f32, params->bias_tag));
        dest_md.reset(
            new memory::desc(params->dest_dims, dt::f32, params->dest_tag));

        src_mem.reset(new memory(*src_md, engine, nullptr));
        weight_mem.reset(new memory(*weight_md, engine, nullptr));
        bias_mem.reset(new memory(*bias_md, engine, nullptr));
        dest_mem.reset(new memory(*dest_md, engine, nullptr));

        if (bias) {
            convolution_forward_desc.reset(new convolution_forward::desc(
                prop_kind::forward, algorithm::convolution_direct, *src_md,
                *weight_md, *bias_md, *dest_md, params->strides,
                params->dilation, params->padding, params->padding));
        } else {
            convolution_forward_desc.reset(new convolution_forward::desc(
                prop_kind::forward, algorithm::convolution_direct, *src_md,
                *weight_md, *dest_md, params->strides, params->dilation,
                params->padding, params->padding));
        }

        auto convolution_pd = convolution_forward::primitive_desc(
            *convolution_forward_desc, engine);
        convolution_forward_prim.reset(new convolution_forward(convolution_pd));
        convolution_forward_args.insert({DNNL_ARG_SRC, *src_mem});
        convolution_forward_args.insert({DNNL_ARG_WEIGHTS, *weight_mem});
        convolution_forward_args.insert({DNNL_ARG_BIAS, *bias_mem});
        convolution_forward_args.insert({DNNL_ARG_DST, *dest_mem});

        return convolution_pd;
    }

    void add_base_data(void* weight_data, void* bias_data) {
        weight_mem->set_data_handle(weight_data);
        bias_mem->set_data_handle(bias_data);
    }

    void add_src_dest_data(void* src_data, void* dest_data) {
        src_mem->set_data_handle(src_data);
        dest_mem->set_data_handle(dest_data);
    }

    void forward() {
        convolution_forward_prim->execute(engine_stream,
                                          convolution_forward_args);
        engine_stream.wait();
        src_mem->set_data_handle(nullptr);
        weight_mem->set_data_handle(nullptr);
        bias_mem->set_data_handle(nullptr);
        dest_mem->set_data_handle(nullptr);
    }
};

struct OneDNNConv2DBackwardParams {
    memory::dims src_dims;
    memory::dims weight_dims;
    memory::dims bias_dims;
    memory::dims grad_dims;

    memory::dims weight_grad_dims;
    memory::dims bias_grad_dims;
    memory::dims src_grad_dims;
    memory::dims dilation = {0, 0};
    tag src_tag;
    tag weight_tag;
    tag bias_tag = tag::a;
    tag grad_tag;
    tag src_grad_tag;
    tag weight_grad_tag;
    tag bias_grad_tag = tag::a;

    memory::dims strides;
    memory::dims padding;

    OneDNNConv2DBackwardParams(Tensor& src, Tensor& kernel, Tensor& grad,
                               std::vector<long> strides_,
                               std::vector<long> padding_, bool back = false) {
        const memory::dim N = src.get_shape().shape[0];
        const memory::dim I_H = src.get_shape().shape[2];
        const memory::dim I_W = src.get_shape().shape[3];

        const memory::dim K_H = kernel.get_shape().shape[2];
        const memory::dim K_W = kernel.get_shape().shape[3];
        const memory::dim IC = kernel.get_shape().shape[1];
        const memory::dim OC = kernel.get_shape().shape[0];
        const memory::dim pad_1 = padding_[0];
        const memory::dim pad_2 = padding_[1];
        const memory::dim stride_1 = strides_[0];
        const memory::dim stride_2 = strides_[1];

        const memory::dim O_H = grad.get_shape()[2];
        const memory::dim O_W = grad.get_shape()[3];

        grad_dims = grad.get_shape().shape;  //{N, OC, I_H, I_W};
        weight_dims = {IC, OC, K_H, K_W};    // kernel.get_shape().shape;  //;
        bias_dims = {OC};
        src_grad_dims = src.get_shape().shape;

        src_dims = src.get_shape().shape;
        weight_grad_dims = kernel.get_shape().shape;
        bias_grad_dims = {OC};

        strides = {stride_1, stride_2};
        padding = {pad_1, pad_2};

        src_tag = tag::nchw;
        weight_tag = tag::iohw;
        grad_tag = tag::nchw;

        src_grad_tag = tag::nchw;
        weight_grad_tag = tag::iohw;
    }
};

class OneDNNConv2DBackward {
   public:
    dnnl::engine engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream = dnnl::stream(engine);
    std::shared_ptr<OneDNNConv2DBackwardParams> params;

    std::shared_ptr<memory::desc> src_md = nullptr;
    std::shared_ptr<memory::desc> weight_md = nullptr;
    std::shared_ptr<memory::desc> bias_md = nullptr;
    std::shared_ptr<memory::desc> grad_md = nullptr;

    std::shared_ptr<memory::desc> kernel_grad_dest_md = nullptr;
    std::shared_ptr<memory::desc> bias_grad_dest_md = nullptr;
    std::shared_ptr<memory::desc> src_grad_dest_md = nullptr;

    std::shared_ptr<memory> src_mem = nullptr;
    std::shared_ptr<memory> weight_mem = nullptr;
    std::shared_ptr<memory> bias_mem = nullptr;
    std::shared_ptr<memory> grad_mem = nullptr;

    std::shared_ptr<memory> kernel_grad_dest_mem = nullptr;
    std::shared_ptr<memory> bias_grad_dest_mem = nullptr;
    std::shared_ptr<memory> src_grad_dest_mem = nullptr;

    std::shared_ptr<convolution_backward_weights::desc>
        convolution_backward_weights_desc = nullptr;
    std::shared_ptr<deconvolution_forward::desc> deconvolution_forward_desc =
        nullptr;

    std::shared_ptr<convolution_backward_weights>
        convolution_backward_weights_prim = nullptr;
    std::shared_ptr<deconvolution_forward> deconvolution_forward_prim = nullptr;
    std::unordered_map<int, memory> convolution_backward_weights_args;
    std::unordered_map<int, memory> deconvolution_forward_args;

    OneDNNConv2DBackward(std::shared_ptr<OneDNNConv2DBackwardParams> _params) {
        params = _params;
    }
    void initialize(convolution_forward::primitive_desc desc) {
        src_grad_dest_md.reset(new memory::desc(params->src_grad_dims, dt::f32,
                                                params->src_grad_tag));
        weight_md.reset(
            new memory::desc(params->weight_dims, dt::f32, params->weight_tag));
        bias_md.reset(
            new memory::desc(params->bias_dims, dt::f32, params->bias_tag));
        grad_md.reset(
            new memory::desc(params->grad_dims, dt::f32, params->grad_tag));

        src_grad_dest_mem.reset(new memory(*src_grad_dest_md, engine, nullptr));
        weight_mem.reset(new memory(*weight_md, engine, nullptr));
        bias_mem.reset(new memory(*bias_md, engine, nullptr));
        grad_mem.reset(new memory(*grad_md, engine, nullptr));

        deconvolution_forward_desc.reset(new deconvolution_forward::desc(
            prop_kind::forward_inference, algorithm::deconvolution_direct,
            *grad_md, *weight_md, *src_grad_dest_md, params->strides,
            params->dilation, params->padding, params->padding));

        auto deconvolution_pd = deconvolution_forward::primitive_desc(
            *deconvolution_forward_desc, engine);
        deconvolution_forward_prim.reset(
            new deconvolution_forward(deconvolution_pd));

        // clang-format off
        deconvolution_forward_args.insert({DNNL_ARG_SRC, *grad_mem});  // input
        deconvolution_forward_args.insert({DNNL_ARG_WEIGHTS, *weight_mem});  // weights
        deconvolution_forward_args.insert({DNNL_ARG_BIAS, *bias_mem});  // bias
        deconvolution_forward_args.insert({DNNL_ARG_DST, *src_grad_dest_mem});  // bias
        // clang-format on

        src_md.reset(
            new memory::desc(params->src_dims, dt::f32, params->src_tag));
        kernel_grad_dest_md.reset(new memory::desc(
            params->weight_grad_dims, dt::f32, params->weight_grad_tag));
        bias_grad_dest_md.reset(new memory::desc(
            params->bias_grad_dims, dt::f32, params->bias_grad_tag));

        kernel_grad_dest_mem.reset(
            new memory(*kernel_grad_dest_md, engine, nullptr));
        bias_grad_dest_mem.reset(
            new memory(*bias_grad_dest_md, engine, nullptr));
        src_mem.reset(new memory(*src_md, engine, nullptr));

        convolution_backward_weights_desc.reset(
            new convolution_backward_weights::desc(
                algorithm::convolution_direct, *src_md, *kernel_grad_dest_md,
                *bias_grad_dest_md, *grad_md, params->strides, params->dilation,
                params->padding, params->padding));

        auto convolution_pd = convolution_backward_weights::primitive_desc(
            *convolution_backward_weights_desc, engine, desc);
        convolution_backward_weights_prim.reset(
            new convolution_backward_weights(convolution_pd));

        // clang-format off

        convolution_backward_weights_args.insert({DNNL_ARG_DIFF_DST, *grad_mem});  // grad
        convolution_backward_weights_args.insert({DNNL_ARG_DIFF_WEIGHTS, *kernel_grad_dest_mem});  // where weights grad get assigned to
        convolution_backward_weights_args.insert({DNNL_ARG_DIFF_BIAS, *bias_grad_dest_mem});  // where weights grad get assigned to

        convolution_backward_weights_args.insert({DNNL_ARG_SRC, *src_mem});  // input
        convolution_backward_weights_args.insert({DNNL_ARG_WEIGHTS, *weight_mem}); // weights
        convolution_backward_weights_args.insert({DNNL_ARG_BIAS, *bias_mem});  // bias

        // clang-format on
    }

    void add_weights_data(void* data) { weight_mem->set_data_handle(data); }
    void add_bias_data(void* data) { bias_mem->set_data_handle(data); }

    void add_grad_data(void* grad_data) {
        grad_mem->set_data_handle(grad_data);
    }
    void add_kernel_grad_loc(void* grad_data) {
        kernel_grad_dest_mem->set_data_handle(grad_data);
    }
    void add_bias_grad_loc(void* grad_data) {
        bias_grad_dest_mem->set_data_handle(grad_data);
    }
    void add_src_grad_loc(void* grad_data) {
        src_grad_dest_mem->set_data_handle(grad_data);
    }

    void add_base_data(void* weight_data, void* bias_data) {
        weight_mem->set_data_handle(weight_data);
        bias_mem->set_data_handle(bias_data);
    }

    void add_src_dest_data(void* src_data, void* dest_data) {
        src_mem->set_data_handle(src_data);
        // kernel_grad_dest_mem->set_data_handle(dest_data);
    }

    void forward() {
        try {
            convolution_backward_weights_prim->execute(
                engine_stream, convolution_backward_weights_args);
            engine_stream.wait();

            deconvolution_forward_prim->execute(engine_stream,
                                                deconvolution_forward_args);
            engine_stream.wait();
        } catch (dnnl::error& e) {
            std::cout << e.status << std::endl;
            std::cout << e.what() << std::endl;
        }

        src_mem->set_data_handle(nullptr);
        weight_mem->set_data_handle(nullptr);
        bias_mem->set_data_handle(nullptr);

        grad_mem->set_data_handle(nullptr);
        kernel_grad_dest_mem->set_data_handle(nullptr);
        bias_grad_dest_mem->set_data_handle(nullptr);
        src_grad_dest_mem->set_data_handle(nullptr);
    }
};

}  // namespace onednn

}  // namespace sail
