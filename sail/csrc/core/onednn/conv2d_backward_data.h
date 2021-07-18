#pragma once

#include <dnnl.hpp>
#include "Tensor.h"
#include "conv2d_forward.h"
#include "onednn_utils.h"

#include <chrono>
using namespace std::chrono;

using namespace dnnl;
using dnnl::inner_product_forward;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace sail {

namespace onednn {

class OneDNNConv2DBackwardData : public Primitive {
   public:
    dnnl::engine engine;
    dnnl::stream engine_stream;
    std::shared_ptr<OneDNNConv2DForwardParams> params;

    std::shared_ptr<memory::desc> src_md = nullptr;
    std::shared_ptr<memory::desc> weight_md = nullptr;
    std::shared_ptr<memory::desc> bias_md = nullptr;
    std::shared_ptr<memory::desc> dest_md = nullptr;

    std::shared_ptr<memory> src_mem = nullptr;
    std::shared_ptr<memory> weight_mem = nullptr;
    std::shared_ptr<memory> bias_mem = nullptr;
    std::shared_ptr<memory> dest_mem = nullptr;

    std::shared_ptr<memory> reorder_src = nullptr;
    std::shared_ptr<memory> reorder_dest = nullptr;
    std::shared_ptr<memory> reorder_weight = nullptr;

    std::shared_ptr<convolution_backward_data::desc> desc = nullptr;
    std::shared_ptr<convolution_forward::desc> forward_desc = nullptr;

    std::shared_ptr<convolution_backward_data> primitive_desc = nullptr;
    std::unordered_map<int, memory> args;

    OneDNNConv2DBackwardData(
        std::shared_ptr<OneDNNConv2DForwardParams> _params) {
        params = _params;
        engine = get_engine();
        engine_stream = dnnl::stream(engine);
    }
    OneDNNConv2DBackwardData(OneDNNConv2DForwardParams _params) {
        params.reset(new OneDNNConv2DForwardParams(_params));
        engine = get_engine();
        engine_stream = dnnl::stream(engine);
    }
    void initialize() {
        src_md.reset(new memory::desc(params->src_dims, dt::f32, tag::any));
        weight_md.reset(
            new memory::desc(params->weight_dims, dt::f32, tag::any));
        dest_md.reset(new memory::desc(params->dest_dims, dt::f32, tag::any));

        desc.reset(new convolution_backward_data::desc(
            algorithm::convolution_direct, *src_md, *weight_md, *dest_md,
            params->strides, params->dilation, params->padding_l,
            params->padding_r));
        forward_desc.reset(new convolution_forward::desc(
            prop_kind::forward, algorithm::convolution_direct, *src_md,
            *weight_md, *dest_md, params->strides, params->dilation,
            params->padding_l, params->padding_r));

        auto forward_pd =
            convolution_forward::primitive_desc(*forward_desc, engine);
        auto backward_pd = convolution_backward_data::primitive_desc(
            *desc, engine, forward_pd);

        src_mem.reset(new memory(backward_pd.diff_src_desc(), engine, nullptr));
        weight_mem.reset(
            new memory(backward_pd.weights_desc(), engine, nullptr));
        dest_mem.reset(
            new memory(backward_pd.diff_dst_desc(), engine, nullptr));

        primitive_desc.reset(new convolution_backward_data(backward_pd));
        args.insert({DNNL_ARG_DIFF_SRC, *src_mem});
        args.insert({DNNL_ARG_WEIGHTS, *weight_mem});
        args.insert({DNNL_ARG_DIFF_DST, *dest_mem});
    }

    inline void add_weight_data(void* data) {
        auto weight_desc =
            memory::desc(params->weight_dims, dt::f32, params->weight_tag);

        if (weight_mem->get_desc() != weight_desc) {
            auto memory_ = memory(weight_desc, engine, data);
            reorder_weight.reset(new memory(weight_mem->get_desc(), engine));
            auto converter = ReorderFactory(&memory_, reorder_weight.get());
            converter.forward();
            weight_mem->set_data_handle(reorder_weight->get_data_handle());
        } else {
            weight_mem->set_data_handle(data);
        }
    }
    inline void add_src_data(void* data) {
        auto input_desc =
            memory::desc(params->src_dims, dt::f32, params->src_tag);
        if (src_mem->get_desc() != input_desc) {
            auto memory_ = memory(input_desc, engine, data);
            reorder_src.reset(new memory(src_mem->get_desc(), engine));
            auto converter = ReorderFactory(&memory_, reorder_src.get());
            converter.forward();
            src_mem->set_data_handle(reorder_src->get_data_handle());
        } else {
            src_mem->set_data_handle(data);
        }
    }
    inline void add_dest_data(void* data) {
        auto dest_desc =
            memory::desc(params->dest_dims, dt::f32, params->dest_tag);

        if (dest_mem->get_desc() != dest_desc) {
            auto memory_ = memory(dest_desc, engine, data);
            reorder_dest.reset(new memory(dest_mem->get_desc(), engine));
            auto converter = ReorderFactory(&memory_, reorder_dest.get());
            converter.forward();
            dest_mem->set_data_handle(reorder_dest->get_data_handle());
        } else {
            dest_mem->set_data_handle(data);
        }
    }

    void copy_src_back(void* to) {
        auto input_desc =
            memory::desc(params->src_dims, dt::f32, params->src_tag);
        if (src_mem->get_desc() != input_desc) {
            auto input_desc =
                memory::desc(params->src_dims, dt::f32, params->src_tag);
            auto out = memory(input_desc, engine, to);
            auto converter = ReorderFactory(src_mem.get(), &out);
            converter.forward();
        }
    }

    void forward(Tensor& src) {
        primitive_desc->execute(engine_stream, args);
        engine_stream.wait();

        copy_src_back(src.get_data());

        src_mem->set_data_handle(nullptr);
        weight_mem->set_data_handle(nullptr);
        dest_mem->set_data_handle(nullptr);
    }
};

class Conv2DBackwardDataFactory
    : public PrimitiveFactory<OneDNNConv2DBackwardData> {
   public:
    OneDNNConv2DBackwardData* prim;

    Tensor src;

    Conv2DBackwardDataFactory(Tensor& input_tensor, Tensor& weights,
                              Tensor& output, std::vector<long> strides_,
                              std::vector<long> padding_l,
                              std::vector<long> padding_r) {
        auto p = OneDNNConv2DForwardParams(
            input_tensor.get_shape(), weights.get_shape(), output.get_shape(),
            strides_, padding_l, padding_r);

        src = input_tensor;

        std::string key = p.get_key_backward_data();
        prim = static_cast<OneDNNConv2DBackwardData*>(get(key));
        if (prim == nullptr) {
            prim = new OneDNNConv2DBackwardData(p);
            prim->initialize();

            add(key, prim);
        }
        fill(input_tensor.get_data(), weights.get_data(), output.get_data());
    }

    void fill(void* d1, void* d2, void* d3) {
        prim->add_src_data(d1);
        prim->add_weight_data(d2);
        prim->add_dest_data(d3);
    }

    void forward() { prim->forward(src); }
};

}  // namespace onednn

}  // namespace sail
