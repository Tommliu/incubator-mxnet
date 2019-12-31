/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * Copyright (c) 2019 by Contributors
 * \file np_percentile_op-inl.h
*/

#ifndef MXNET_OPERATOR_NUMPY_NP_PERCENTILE_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_PERCENTILE_OP_INL_H_

#include "../tensor/ordering_op-inl.h"
#include "../../common/utils.h"
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct NumpyPercentileParam : public dmlc::Parameter<NumpyPercentileParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  std::string interpolation;
  bool keepdims; 
  DMLC_DECLARE_PARAMETER(NumpyPercentileParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(interpolation).set_default("linear")
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
  }
};

template<int NDim>
struct percentile_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out,
                                  const DType* a_sort,
                                  mshadow::Shape<NDim> t_shape,
                                  mshadow::Shape<NDim> r_shape, 
                                  const string interpolation) {
  using namespace mxnet_op;
  using namespace mshadow;
  using namespace std;


}

inline bool NumpyPercentileShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape qshape = in_attrs->at(1);
  CHECK_LE(qshape.ndim(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyPercentileParam& param = nnvm::get<NumpyPercentileParam>(attrs.parsed);
  mxnet::TShape shape = NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims);

  if (qshape.ndim() == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  } else {
    mxnet::TShape oshape(shape.ndim() + 1 , -1);
    oshape[0] = qshape[0];
    for (size_t i = 1 ; i < oshape.ndim(); i ++)
      oshape[i] = shape[i - 1];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  }
  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyPercentileType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  if (in_attrs->at(0) == mshadow::kFloat64){
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat64);
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kFloat32);
  }
  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

template<typename xpu>
void NumpyPercentileForward(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  if (req[0] == kNullOp) return;

  using namespace mxnet;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob &data = inputs[0];
  const TBlob &percentile = inputs[1];
  const TBlob &out = outputs[0];
  const NumpyPercentileParam& param = nnvm::get<NumpyPercentileParam>(attrs.parsed); 
  const string interpolation = param.interpolation;
  //const keepdims = param.keepdims;
  auto small = NumpyReduceAxesShapeImpl(data.shape_, param.axis, false);
  // this impelementation only considers keepdims=false; need to be reviewed.
  TShape r_shape;
  r_shape = TShape(small.ndim()+1, 1);
  r_shape[0] = percentile.Size();
  for (int i = 1; i < r_shape.ndim(); ++i) {
    r_shape[i] = small[i-1];
  }
  //Origin axes
  TShape axes;
  if (!axis.has_value()) {
    axes = TShape(data.shape_.ndim(), 1);
    for (int i = 0; i < data.shape_.ndim(); ++i) {
      axes[i] = i;
    }
  } else {
    axes = TShape(axis.value());
  }
  //Transpose the axes
  TShape t_axes(data.shape_.ndim(), 1);
  int j, k = 0; 
  for(int i = 0; i < 6; i++){
      if (j < 2 && data.shape_[i] == axis[j]){
          j ++; 
          continue;
      }
      t_axes[k] = i; 
      k++;
  }
  for (int jj = k; jj < t_axes.ndim(); ++jj) {
    t_axes[jj] = axes[jj-k];
  }
  //Transpose Shape with reduced dims at dim -1
  Shape t_shape(small.ndim()+1, 1);
  for (int i = 0; i < small.ndim(); ++i) {
    t_shape[i] = small[i];
  }
  size_t red_size = 1;
  for (int i = 0; i < axes.ndim(); ++i) {
    red_size *= data.shape_[axes[i]];
  }
  t_shape[t_shape.ndim()-1] = red_size;
  //Transpose Shape extension
  TShape t_shape_ex(data.shape_.ndim(), 1);
  for (int i = 0; i < data.shape_.ndim(); ++i) {
    t_shape_ex[i] = a.shape_[t_axes[i]];
  }

  TopKParam topk_param = TopKParam();
  topk_param.axis = dmlc::optional<int>(-1);
  topk_param.is_ascend = true;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnValue;

  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    size_t alignment = std::max(sizeof(DType), sizeof(index_t));
    topk_workspace_size = TopK_Workspace_Cal<xpu, DType>(data, topk_param, alignment)

    size_t temp_data_size = data.Size() * sizeof(DType);
    size_t idx_size = data.Size() * sizeof(index_t);
    size_t temp_mem_size = 2 * temp_data_size + idx_size;
    size_t workspace_size = topk_workspace_size * 2 + temp_mem_size; // need 2 times the workspace? 

    Tensor<xpu, 1, char> temp_mem =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);

    DType* trans_ptr, *sort_ptr;
    char* workspace_curr_ptr;
    index_t* idx_ptr;
    if (sizeof(DType) >= sizeof(index_t)) {
      trans_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
      sort_ptr = reinterpret_cast<DType*>(temp_mem.dptr_ + temp_data_size);
      idx_ptr = reinterpret_cast<index_t*>(temp_mem.dptr_ + 2 * temp_data_size);
    } else {
      idx_ptr = reinterpret_cast<index_t*>(temp_mem.dptr_);
      trans_ptr = reinterpret_cast<DType*>(temp_mem.dptr_ + idx_size);
      sort_ptr = reinterpret_cast<DType*>(temp_mem.dptr_ + temp_data_size + idx_size);
    }
    workspace_curr_ptr = temp_mem.dptr_ + 2 * temp_data_size + idx_size;

    TBlob a_trans = TBlob(trans_ptr, t_shape_ex, xpu::kDevMask);
    TransposeImpl<xpu>(ctx.run_ctx, a, a_trans, t_axes);
    TBlob a_sort = TBlob(sort_ptr, t_shape, xpu::kDevMask);
    TBlob a_idx = TBlob(idx_ptr, t_shape, xpu::kDevMask);

    std::vector<OpReqType> req_TopK = {kWriteTo, kNullOp};
    TBlob src = a_trans.reshape(t_shape);
    std::vector<TBlob> ret = {a_sort, a_idx};

    TopK_Workspace_Impl<xpu, DType, index_t>(ctx, req_TopK, src, ret, topk_param,
                                             workspace_curr_ptr);

    MXNET_NDIM_SWITCH(small.ndim()+1, NDim, {
      Kernel<percentile_forward<NDim>, xpu>::Launch(
          s, r_shape.Size(), r.dptr<DType>(), a_sort.dptr<DType>(),
            t_shape.get<NDim>(), r_shape.get<NDim>(), interpolation);
      })
  })
}


}  // namespace op
}  // namespace mxnet



#endif  // MXNET_OPERATOR_NUMPY_NP_PERCENTILE_OP_INL_H_
