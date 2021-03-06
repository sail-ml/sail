// allow-no-source

#pragma once

#include <dnnl.hpp>
#include "Tensor.h"
#include "onednn_utils.h"

using namespace dnnl;
using dnnl::inner_product_forward;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace sail {

namespace onednn {

struct OneDNNMaxPoolingParams {
    memory::dims src_dims;
    memory::dims dest_dims;
    memory::dims dilation = {0, 0};
    tag src_tag;
    tag dest_tag;

    memory::dims strides;
    memory::dims padding_l, padding_r;
    memory::dims kernel_shape;

    std::shared_ptr<memory> workspace_mem = nullptr;

    OneDNNMaxPoolingParams(Tensor src, TensorShape kernel,
                           TensorShape output_shape, std::vector<long> strides_,
                           std::vector<long> padding_l,
                           std::vector<long> padding_r, bool back = false)
        : src_dims(std::move(src).get_shape().shape),
          dest_dims(std::move(output_shape).shape),
          strides(std::move(strides_)),
          padding_l(std::move(padding_l)),
          padding_r(std::move(padding_r)),
          kernel_shape(std::move(kernel).shape),
          src_tag(tag::nchw),
          dest_tag(tag::nchw){};
};

class OneDNNMaxPooling {
   public:
    dnnl::engine engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream = dnnl::stream(engine);
    std::shared_ptr<OneDNNMaxPoolingParams> params;

    std::shared_ptr<memory::desc> src_md = nullptr;
    std::shared_ptr<memory::desc> dest_md = nullptr;

    std::shared_ptr<memory> src_mem = nullptr;
    std::shared_ptr<memory> dest_mem = nullptr;
    std::shared_ptr<memory> workspace_mem = nullptr;

    std::shared_ptr<pooling_v2_forward::desc> pooling_v2_forward_desc = nullptr;
    std::shared_ptr<pooling_v2_forward> pooling_v2_forward_prim = nullptr;
    std::unordered_map<int, memory> forward_args =
        std::unordered_map<int, memory>();

    OneDNNMaxPooling(std::shared_ptr<OneDNNMaxPoolingParams> _params) {
        params = _params;
    }

    pooling_v2_forward::primitive_desc initialize() {
        src_md.reset(
            new memory::desc(params->src_dims, dt::f32, params->src_tag));
        dest_md.reset(
            new memory::desc(params->dest_dims, dt::f32, params->dest_tag));

        src_mem.reset(new memory(*src_md, engine, nullptr));
        dest_mem.reset(new memory(*dest_md, engine, nullptr));

        pooling_v2_forward_desc.reset(new pooling_v2_forward::desc(
            prop_kind::forward_training, algorithm::pooling_max, *src_md,
            *dest_md, params->strides, params->kernel_shape, params->dilation,
            params->padding_l, params->padding_r));

        auto pooling_pd = pooling_v2_forward::primitive_desc(
            *pooling_v2_forward_desc, engine);

        params->workspace_mem.reset(
            new memory(pooling_pd.workspace_desc(), engine, nullptr));

        pooling_v2_forward_prim.reset(new pooling_v2_forward(pooling_pd));
        forward_args.insert({DNNL_ARG_SRC, *src_mem});
        forward_args.insert({DNNL_ARG_DST, *dest_mem});
        forward_args.insert({DNNL_ARG_WORKSPACE, *(params->workspace_mem)});

        return pooling_pd;
    }

    void add_src_dest_data(void* src_data, void* dest_data) {
        src_mem->set_data_handle(src_data);
        dest_mem->set_data_handle(dest_data);
    }

    void forward() {
        pooling_v2_forward_prim->execute(engine_stream, forward_args);
        engine_stream.wait();
        src_mem->set_data_handle(nullptr);
        dest_mem->set_data_handle(nullptr);
    }
};

class OneDNNMaxPoolingBackward {
   public:
    dnnl::engine engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream = dnnl::stream(engine);
    std::shared_ptr<OneDNNMaxPoolingParams> params;

    std::shared_ptr<memory::desc> grad_md = nullptr;

    std::shared_ptr<memory::desc> src_grad_dest_md = nullptr;

    std::shared_ptr<memory> grad_mem = nullptr;

    std::shared_ptr<memory> src_grad_dest_mem = nullptr;
    std::shared_ptr<pooling_v2_forward::desc> pooling_v2_forward_desc = nullptr;

    std::shared_ptr<pooling_v2_backward ::desc> pooling_v2_backward_desc =
        nullptr;

    std::shared_ptr<pooling_v2_backward> pooling_v2_backward_prim = nullptr;
    std::unordered_map<int, memory> backward_args =
        std::unordered_map<int, memory>();

    pooling_v2_forward::primitive_desc forward_prim =
        pooling_v2_forward::primitive_desc();

    OneDNNMaxPoolingBackward(std::shared_ptr<OneDNNMaxPoolingParams> _params)
        : params(std::move(_params)) {}
    void store_forward_desc(pooling_v2_forward::primitive_desc desc) {
        forward_prim = desc;
    }
    void initialize() {
        src_grad_dest_md.reset(
            new memory::desc(params->src_dims, dt::f32, params->src_tag));
        grad_md.reset(
            new memory::desc(params->dest_dims, dt::f32, params->dest_tag));

        src_grad_dest_mem.reset(new memory(*src_grad_dest_md, engine, nullptr));
        grad_mem.reset(new memory(*grad_md, engine, nullptr));

        pooling_v2_backward_desc.reset(new pooling_v2_backward::desc(
            algorithm::pooling_max, *src_grad_dest_md, *grad_md,
            params->strides, params->kernel_shape, params->dilation,
            params->padding_l, params->padding_r));

        pooling_v2_forward_desc.reset(new pooling_v2_forward::desc(
            prop_kind::forward_training, algorithm::pooling_max,
            *src_grad_dest_md, *grad_md, params->strides, params->kernel_shape,
            params->dilation, params->padding_l, params->padding_r));

        auto pooling_pd = pooling_v2_forward::primitive_desc(
            *pooling_v2_forward_desc, engine);

        auto bwd_pooling_pd = pooling_v2_backward::primitive_desc(
            *pooling_v2_backward_desc, engine, pooling_pd);

        pooling_v2_backward_prim.reset(new pooling_v2_backward(bwd_pooling_pd));

        params->workspace_mem.reset(
            new memory(pooling_pd.workspace_desc(), engine));
        // clang-format off
        backward_args.insert({DNNL_ARG_DIFF_DST, *grad_mem}); 
        backward_args.insert({DNNL_ARG_DIFF_SRC, *src_grad_dest_mem});
        backward_args.insert({DNNL_ARG_WORKSPACE, *(params->workspace_mem)});
        // clang-format on
    }

    void add_grad_data(void* grad_data) {
        grad_mem->set_data_handle(grad_data);
    }
    void add_src_grad_loc(void* grad_data) {
        src_grad_dest_mem->set_data_handle(grad_data);
    }

    void forward() {
        pooling_v2_backward_prim->execute(engine_stream, backward_args);
        engine_stream.wait();

        grad_mem->set_data_handle(nullptr);
        src_grad_dest_mem->set_data_handle(nullptr);
    }
};

}  // namespace onednn

}  // namespace sail
