#include "gtest/gtest.h"
#include "core/exception.h"
#include "core/tensor_shape.h"
#include "core/Tensor.h"
#include "core/dtypes.h"
#include "core/numeric.h"
#include "core/ops/ops.h"

#include <iostream>

inline std::vector<Dtype> all_dtypes = {Dtype::sBool, 
                                        Dtype::sInt8, Dtype::sUInt8, 
                                        Dtype::sInt16, Dtype::sUInt16,
                                        Dtype::sInt32, Dtype::sUInt32,
                                        Dtype::sInt64, Dtype::sUInt64, 
                                        Dtype::sFloat32, Dtype::sFloat64};
inline std::vector<std::string> all_dtype_string = {"bool", 
                                                    "int8", "uint8", 
                                                    "int16", "uint16",
                                                    "int32", "uint32",
                                                    "int64", "uint64", 
                                                    "float32", "float64"};
inline std::vector<int> all_np_dtypes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12};


using namespace sail;
TEST(DtypeTest, sbool) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sBool);
    ASSERT_THROW(dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          }), DtypeError);
}
TEST(DtypeTest, uint8) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sUInt8);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          });
}

TEST(DtypeTest, int8) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sInt8);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          });
}

TEST(DtypeTest, uint16) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sUInt16);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          });
}

TEST(DtypeTest, int16) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sInt16);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          });
}

TEST(DtypeTest, uint32) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sUInt32);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          });
}

TEST(DtypeTest, int32) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sInt32);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          });
}

TEST(DtypeTest, uint64) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sUInt64);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          });
}

TEST(DtypeTest, int64) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sInt64);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(false);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(true);
                          });
}

TEST(DtypeTest, float32) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sFloat32);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(true);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(false);
                          });
}
TEST(DtypeTest, float64) {
    auto x = random::uniform(TensorShape({10}), 0, 100);
    auto y = x.cast(Dtype::sFloat64);
    dispatch_fp_int_types(y.get_dtype(),
                          [&](auto pt) {
                              ASSERT_TRUE(true);
                          },
                          [&](auto pt2) {
                              ASSERT_TRUE(false);
                          });
}
TEST(DtypeTest, np_dtype) {
    for (int i = 0; i < all_dtypes.size(); i++) {
        ASSERT_EQ(GetDtypeFromNumpyInt(all_np_dtypes[i]), all_dtypes[i]);
    }
    ASSERT_THROW(GetDtypeFromNumpyInt(100), SailCError);
}
TEST(DtypeTest, np_dtype2) {
    for (int i = 0; i < all_dtypes.size(); i++) {
        ASSERT_EQ(get_np_type_numFromDtype(all_dtypes[i]), all_np_dtypes[i]);
    }
}
TEST(DtypeTest, ostream) {
    for (int i = 0; i < all_dtypes.size(); i++) {
        std::stringstream buffer;
        buffer << all_dtypes[i];
        ASSERT_EQ(buffer.str(), all_dtype_string[i]);
    }
}
TEST(DtypeTest, dispatch_all) {
    for (int i = 0; i < all_dtypes.size(); i++) {
        dispatch_all_types(all_dtypes[i], [&](auto pt) {
            ASSERT_TRUE(true);
        });
    }
}
TEST(DtypeTest, dispatch_all_numeric) {
    for (int i = 1; i < all_dtypes.size(); i++) {
        dispatch_all_numeric_types(all_dtypes[i], [&](auto pt) {
            ASSERT_TRUE(true);
        });
    }

    ASSERT_THROW(dispatch_all_numeric_types(Dtype::sBool, [&](auto pt) {}), DtypeError);
}
TEST(DtypeTest, GetDtypeSize) {
    for (int i = 0; i < all_dtypes.size(); i++) {
        dispatch_all_types(all_dtypes[i], [&](auto pt) {
            using T = typename decltype(pt)::type;
            ASSERT_EQ(GetDtypeSize(all_dtypes[i]), sizeof(T));
        });
    }
}
TEST(DtypeTest, MinDtype) {
    ASSERT_EQ(min_type((long)10), Dtype::sUInt8);
    ASSERT_EQ(min_type((long)259), Dtype::sUInt16);
    ASSERT_EQ(min_type((long)84345), Dtype::sUInt32);
    ASSERT_EQ(min_type((long)5294967295), Dtype::sUInt64);
    ASSERT_EQ(min_type((long)-10), Dtype::sInt8);
    ASSERT_EQ(min_type((long)-259), Dtype::sInt16);
    ASSERT_EQ(min_type((long)-1147483648), Dtype::sInt32);
    ASSERT_EQ(min_type((long)-5294967295), Dtype::sInt64);
}
TEST(DtypeTest, Promotion) {
    ASSERT_EQ(promote_dtype(Dtype::sBool, Dtype::sBool), Dtype::sUInt8);
    
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sUInt16, true), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sUInt8, true), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sInt8, true), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sInt16, true), Dtype::sFloat32);
    
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sUInt32, true), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sInt32, true), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sUInt64, true), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sInt64, true), Dtype::sFloat64);
    
    ASSERT_EQ(promote_dtype(Dtype::sUInt16, Dtype::sFloat32, true), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sUInt8, Dtype::sFloat32, true), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sInt8, Dtype::sFloat32, true), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sInt16, Dtype::sFloat32, true), Dtype::sFloat32);
    
    ASSERT_EQ(promote_dtype(Dtype::sUInt32, Dtype::sFloat32, true), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sInt32, Dtype::sFloat32, true), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sUInt64, Dtype::sFloat32, true), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sInt64, Dtype::sFloat32, true), Dtype::sFloat64);

    ASSERT_EQ(promote_dtype(Dtype::sUInt16, Dtype::sUInt32), Dtype::sUInt32);
    ASSERT_EQ(promote_dtype(Dtype::sUInt16, Dtype::sInt16), Dtype::sInt32);
    ASSERT_EQ(promote_dtype(Dtype::sInt16, Dtype::sUInt16), Dtype::sInt32);

    ASSERT_EQ(promote_dtype(Dtype::sInt16, Dtype::sFloat32), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sInt8, Dtype::sFloat32), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sInt32, Dtype::sFloat32), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sInt64, Dtype::sFloat32), Dtype::sFloat64);
    
    ASSERT_EQ(promote_dtype(Dtype::sInt16, Dtype::sFloat64), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sInt8, Dtype::sFloat64), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sUInt32, Dtype::sFloat32), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sUInt64, Dtype::sFloat32), Dtype::sFloat64);

    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sInt16), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sInt8), Dtype::sFloat32);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sInt32), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sInt64), Dtype::sFloat64);

    ASSERT_EQ(promote_dtype(Dtype::sFloat64, Dtype::sInt16), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sFloat64, Dtype::sInt8), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sUInt32), Dtype::sFloat64);
    ASSERT_EQ(promote_dtype(Dtype::sFloat32, Dtype::sUInt64), Dtype::sFloat64);

    for (auto const& dt1 : all_dtypes) {
        for (auto const& dt2 : all_dtypes) {
            promote_dtype(dt1, dt2);
        }
    }

    for (auto const& dt1 : all_dtypes) {
        for (auto const& dt2 : all_dtypes) {
            promote_dtype(dt1, dt2, true);
        }
    }

}
