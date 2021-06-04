#pragma once

#include <dnnl.hpp>
#include "../Tensor.h"
using namespace dnnl;
using dnnl::inner_product_forward;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace sail {

namespace onednn {
inline unsigned char* DummyData = nullptr;
struct OneDNNLinearParams {
    memory::dims src_dims;
    memory::dims weight_dims;
    memory::dims bias_dims;
    memory::dims dest_dims;
    tag src_tag;
    tag weight_tag;
    tag bias_tag = tag::a;
    tag dest_tag;

    OneDNNLinearParams(memory::dims src_dims, memory::dims weight_dims,
                       memory::dims bias_dims, memory::dims dest_dims,
                       tag src_tag = tag::any, tag weight_tag = tag::any,
                       tag dest_tag = tag::any)
        : src_dims(src_dims),
          weight_dims(weight_dims),
          bias_dims(bias_dims),
          dest_dims(dest_dims),
          src_tag(src_tag),
          weight_tag(weight_tag),
          dest_tag(dest_tag) {}

    OneDNNLinearParams(Tensor& src, int input_features, int output_features) {
        const memory::dim N = src.get_shape().shape[0], IF = input_features,
                          OF = output_features;

        src_dims = {N, IF};
        weight_dims = {OF, IF};
        bias_dims = {OF};
        dest_dims = {N, OF};

        src_tag = tag::nc;
        weight_tag = tag::io;
        dest_tag = tag::nc;
    }
};

class OneDNNLinear {
   public:
    dnnl::engine engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream = dnnl::stream(engine);
    std::shared_ptr<OneDNNLinearParams> params;

    std::shared_ptr<memory::desc> src_md = nullptr;
    std::shared_ptr<memory::desc> weight_md = nullptr;
    std::shared_ptr<memory::desc> bias_md = nullptr;
    std::shared_ptr<memory::desc> dest_md = nullptr;

    std::shared_ptr<memory> src_mem = nullptr;
    std::shared_ptr<memory> weight_mem = nullptr;
    std::shared_ptr<memory> bias_mem = nullptr;
    std::shared_ptr<memory> dest_mem = nullptr;

    std::shared_ptr<inner_product_forward::desc> inner_product_desc = nullptr;
    std::shared_ptr<inner_product_forward> inner_product_prim = nullptr;
    std::unordered_map<int, memory> inner_product_args;

    OneDNNLinear(std::shared_ptr<OneDNNLinearParams> _params) {
        params = _params;
    }
    void initialize() {
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

        inner_product_desc.reset(new inner_product_forward::desc(
            prop_kind::forward_inference, *src_md, *weight_md, *bias_md,
            *dest_md));

        auto inner_product_pd =
            inner_product_forward::primitive_desc(*inner_product_desc, engine);
        inner_product_prim.reset(new inner_product_forward(inner_product_pd));
        inner_product_args.insert({DNNL_ARG_SRC, *src_mem});
        inner_product_args.insert({DNNL_ARG_WEIGHTS, *weight_mem});
        inner_product_args.insert({DNNL_ARG_BIAS, *bias_mem});
        inner_product_args.insert({DNNL_ARG_DST, *dest_mem});
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
        inner_product_prim->execute(engine_stream, inner_product_args);
        engine_stream.wait();
    }
};

}  // namespace onednn

}  // namespace sail
