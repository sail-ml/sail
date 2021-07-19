// allow-no-source

#pragma once

#include <dnnl.hpp>
#include "Tensor.h"
#include "onednn_utils.h"
#include "reorder.h"

#include <chrono>
using namespace std::chrono;

using namespace dnnl;
using dnnl::inner_product_forward;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace sail {

namespace onednn {

inline std::tuple<tag, tag> identify_tags(TensorShape src_shape,
                                          TensorShape kernel_shape) {
    if (src_shape[1] == kernel_shape[1]) {
        return std::make_tuple(tag::nchw, tag::oihw);
    } else if (src_shape[3] == kernel_shape[1]) {
        return std::make_tuple(tag::nhwc, tag::oihw);
    } else {
        THROW_ERROR(SailCError,
                    "Format cannot be determined from input shapes");
    }
}

struct OneDNNConv2DForwardParams {
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
    memory::dims padding_l;
    memory::dims padding_r;

    OneDNNConv2DForwardParams(TensorShape src_shape, TensorShape kernel_shape,
                              TensorShape output_shape,
                              std::vector<long> strides_,
                              std::vector<long> padding_l_,
                              std::vector<long> padding_r_, bool back = false) {
        auto tags = identify_tags(src_shape, kernel_shape);
        src_dims = src_shape.shape;
        weight_dims = kernel_shape.shape;
        dest_dims = output_shape.shape;
        auto OC = kernel_shape[0];
        bias_dims = {OC};

        src_tag = std::get<0>(tags);
        weight_tag = std::get<1>(tags);
        dest_tag = std::get<0>(tags);

        strides = strides_;
        padding_l = padding_l_;
        padding_r = padding_r_;
    }

    std::string get_key() {
        auto key = KeyGenerator()
                       .add("Conv2DFwd")
                       .add(src_dims)
                       .add(weight_dims)
                       .add(dest_dims)
                       .add(strides)
                       .add(padding_l)
                       .add(padding_r)
                       .get_key();
        return key;
    }
    std::string get_key_backward_weights() {
        auto key = KeyGenerator()
                       .add("Conv2DWeightsBwd")
                       .add(src_dims)
                       .add(weight_dims)
                       .add(dest_dims)
                       .add(strides)
                       .add(padding_l)
                       .add(padding_r)
                       .get_key();
        return key;
    }
    std::string get_key_backward_data() {
        auto key = KeyGenerator()
                       .add("Conv2DDataBwd")
                       .add(src_dims)
                       .add(weight_dims)
                       .add(dest_dims)
                       .add(strides)
                       .add(padding_l)
                       .add(padding_r)
                       .get_key();
        return key;
    }
};

class OneDNNConv2DForward : public Primitive {
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

    bool use_bias;

    std::shared_ptr<convolution_forward::desc> convolution_forward_desc =
        nullptr;
    std::shared_ptr<convolution_forward> fwd = nullptr;
    std::shared_ptr<convolution_forward::primitive_desc> conv_pd = nullptr;
    std::unordered_map<int, memory> convolution_forward_args;

    OneDNNConv2DForward(std::shared_ptr<OneDNNConv2DForwardParams> _params,
                        bool use_bias_) {
        use_bias = use_bias_;
        params = _params;
        engine = get_engine();
        engine_stream = dnnl::stream(engine);
    }
    OneDNNConv2DForward(OneDNNConv2DForwardParams _params, bool use_bias_) {
        use_bias = use_bias_;
        params.reset(new OneDNNConv2DForwardParams(_params));
        engine = get_engine();
        engine_stream = dnnl::stream(engine);
    }
    void initialize() {
        src_md.reset(new memory::desc({params->src_dims}, dt::f32, tag::any));
        weight_md.reset(
            new memory::desc({params->weight_dims}, dt::f32, tag::any));
        bias_md.reset(
            new memory::desc({params->bias_dims}, dt::f32, params->bias_tag));
        dest_md.reset(new memory::desc({params->dest_dims}, dt::f32, tag::any));

        if (use_bias) {
            convolution_forward_desc.reset(new convolution_forward::desc(
                prop_kind::forward, algorithm::convolution_direct, *src_md,
                *weight_md, *bias_md, *dest_md, params->strides,
                params->dilation, params->padding_l, params->padding_r));
            bias_mem.reset(new memory(*bias_md, engine, nullptr));
            convolution_forward_args.insert({DNNL_ARG_BIAS, *bias_mem});

        } else {
            convolution_forward_desc.reset(new convolution_forward::desc(
                prop_kind::forward_inference, algorithm::convolution_direct,
                *src_md, *weight_md, *dest_md, params->strides,
                params->dilation, params->padding_l, params->padding_r));
        }

        conv_pd.reset(new convolution_forward::primitive_desc(
            *convolution_forward_desc, engine));

        src_mem.reset(new memory(conv_pd->src_desc(), engine, nullptr));
        weight_mem.reset(new memory(conv_pd->weights_desc(), engine, nullptr));
        dest_mem.reset(new memory(conv_pd->dst_desc(), engine, nullptr));

        fwd.reset(new convolution_forward(*conv_pd));
        convolution_forward_args.insert({DNNL_ARG_SRC, *src_mem});
        convolution_forward_args.insert({DNNL_ARG_WEIGHTS, *weight_mem});
        convolution_forward_args.insert({DNNL_ARG_DST, *dest_mem});
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
    inline void add_bias_data(void* data) { bias_mem->set_data_handle(data); }
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

    inline void copy_back(void* to) {
        auto dest_desc =
            memory::desc(params->dest_dims, dt::f32, params->dest_tag);
        if (dest_mem->get_desc() != dest_desc) {
            auto dest_desc =
                memory::desc(params->dest_dims, dt::f32, params->dest_tag);
            auto out = memory(dest_desc, engine, to);
            auto converter = ReorderFactory(dest_mem.get(), &out);
            converter.forward();
        }
    }

    void reset_handles() {
        src_mem->set_data_handle(nullptr);
        weight_mem->set_data_handle(nullptr);
        if (use_bias) {
            bias_mem->set_data_handle(nullptr);
        }
        dest_mem->set_data_handle(nullptr);
    }

    void forward() {
        fwd->execute(engine_stream, convolution_forward_args);
        engine_stream.wait();
        reset_handles();
    }
    inline void forward(Tensor& dst) {
        fwd->execute(engine_stream, convolution_forward_args);
        engine_stream.wait();

        copy_back(dst.get_data());
        reset_handles();
    }
};

class Conv2DForwardFactory : public PrimitiveFactory<OneDNNConv2DForward> {
   public:
    OneDNNConv2DForward* prim;
    Tensor output_tensor;
    Conv2DForwardFactory(Tensor& input_tensor, Tensor& weights, Tensor& biases,
                         Tensor& output, std::vector<long> strides_,
                         std::vector<long> padding_l,
                         std::vector<long> padding_r) {
        auto p = OneDNNConv2DForwardParams(
            input_tensor.get_shape(), weights.get_shape(), output.get_shape(),
            strides_, padding_l, padding_r);

        output_tensor = output;
        std::string key = p.get_key();
        prim = static_cast<OneDNNConv2DForward*>(get(key));
        if (prim == nullptr) {
            prim = new OneDNNConv2DForward(p, true);
            prim->initialize();

            add(key, prim);
        }
        fill(input_tensor.get_data(), weights.get_data(), biases.get_data(),
             output.get_data());
    }
    Conv2DForwardFactory(Tensor& input_tensor, Tensor& weights, Tensor& output,
                         std::vector<long> strides_,
                         std::vector<long> padding_l,
                         std::vector<long> padding_r) {
        auto p = OneDNNConv2DForwardParams(
            input_tensor.get_shape(), weights.get_shape(), output.get_shape(),
            strides_, padding_l, padding_r);

        output_tensor = output;
        std::string key = p.get_key();
        key.append("no_bias");
        prim = static_cast<OneDNNConv2DForward*>(get(key));
        if (prim == nullptr) {
            prim = new OneDNNConv2DForward(p, false);
            prim->initialize();

            add(key, prim);
        }
        fill(input_tensor.get_data(), weights.get_data(), nullptr,
             output.get_data());
    }

    inline void fill(void* d1, void* d2, void* d3, void* d4) {
        prim->add_src_data(d1);
        prim->add_weight_data(d2);
        if (d3 != nullptr) {
            prim->add_bias_data(d3);
        }
        prim->add_dest_data(d4);
    }

    inline void forward() { prim->forward(output_tensor); }
};

}  // namespace onednn

}  // namespace sail
