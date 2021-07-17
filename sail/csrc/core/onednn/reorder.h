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
// inline unsigned char* DummyData = nullptr;

class OneDNNReorder : public Primitive {
   public:
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> dst_mem;
    std::shared_ptr<primitive> reorder_prim;
    dnnl::stream engine_stream;

    OneDNNReorder(const memory* from, const memory* to) {
        engine_stream = dnnl::stream(get_engine());
        src_mem.reset(new memory(from->get_desc(), get_engine(), nullptr));
        dst_mem.reset(new memory(to->get_desc(), get_engine(), nullptr));
        reorder_prim = std::make_shared<reorder>(reorder(*src_mem, *dst_mem));
    }

    void forward() {
        reorder_prim->execute(engine_stream, {{DNNL_ARG_FROM, *src_mem},
                                              {DNNL_ARG_TO, *dst_mem}});
    }

    std::shared_ptr<primitive> GetPrimitive() { return reorder_prim; }

    void SetMemory(const memory* from, const memory* to) {
        src_mem->set_data_handle(from->get_data_handle());
        dst_mem->set_data_handle(to->get_data_handle());
    }

    static std::string get_key(const memory* from, const memory* to) {
        auto keygen = KeyGenerator();
        auto const& from_desc = from->get_desc().data;
        auto const& to_desc = to->get_desc().data;
        memory::dims from_dims(from_desc.dims,
                               &from_desc.dims[from_desc.ndims]);
        memory::dims to_dims(to_desc.dims, &to_desc.dims[to_desc.ndims]);
        auto from_strides = from_desc.format_desc.blocking.strides;

        // As DNNL memory desc has C style array and only init the used
        // part, so need use the valid part as key.
        auto from_inner_nblks = from_desc.format_desc.blocking.inner_nblks;
        auto from_inner_blks = from_desc.format_desc.blocking.inner_blks;
        auto from_inner_idxs = from_desc.format_desc.blocking.inner_idxs;
        memory::dims from_inner_blks_1(from_inner_blks,
                                       &from_inner_blks[from_inner_nblks]);
        memory::dims from_inner_idxs_1(from_inner_idxs,
                                       &from_inner_idxs[from_inner_nblks]);
        auto to_inner_nblks = to_desc.format_desc.blocking.inner_nblks;
        auto to_inner_blks = to_desc.format_desc.blocking.inner_blks;
        auto to_inner_idxs = to_desc.format_desc.blocking.inner_idxs;
        memory::dims to_inner_blks_1(to_inner_blks,
                                     &to_inner_blks[to_inner_nblks]);
        memory::dims to_inner_idxs_1(to_inner_idxs,
                                     &to_inner_idxs[to_inner_nblks]);

        auto to_strides = to_desc.format_desc.blocking.strides;
        memory::dims from_strides_outer_blocks(from_strides,
                                               &from_strides[from_desc.ndims]);
        memory::dims to_strides_outer_blocks(to_strides,
                                             &to_strides[to_desc.ndims]);

        keygen.add("reorder");
        keygen.add(static_cast<int>(from_desc.extra.flags));
        keygen.add(static_cast<int>(from_inner_nblks));
        keygen.add(from_inner_blks_1);
        keygen.add(from_inner_idxs_1);
        keygen.add(static_cast<int>(from_desc.data_type));
        keygen.add(from_dims);
        keygen.add(from_strides_outer_blocks);
        keygen.add("to");
        keygen.add(static_cast<int>(to_desc.extra.flags));
        keygen.add(static_cast<int>(to_inner_nblks));
        keygen.add(to_inner_blks_1);
        keygen.add(to_inner_idxs_1);
        keygen.add(static_cast<int>(to_desc.data_type));
        keygen.add(to_dims);
        keygen.add(to_strides_outer_blocks);

        return keygen.get_key();
    }

    // std::shared_ptr<dnnl::stream> GetStream() { return stream_; }
};

class ReorderFactory : public PrimitiveFactory<OneDNNReorder> {
   public:
    OneDNNReorder* prim;
    ReorderFactory(const memory* from, const memory* to) {
        // auto p = OneDNNReorder(from, to);
        std::string key = OneDNNReorder::get_key(from, to);
        prim = get(key);
        if (prim == nullptr) {
            prim = new OneDNNReorder(from, to);

            add(key, prim);
        }
        prim->SetMemory(from, to);
        // std::cout << ((float*)(to->get_data_handle()))[0] << std::endl;
    }

    std::shared_ptr<primitive> GetPrimitive() { return prim->GetPrimitive(); }

    void SetMemory(const memory* from, const memory* to) {
        prim->SetMemory(from, to);
    }

    void forward() { prim->forward(); }
};

}  // namespace onednn

}  // namespace sail
