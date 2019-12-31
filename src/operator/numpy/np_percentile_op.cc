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
 * \file np_nonzero_op-inl.h
 * \brief CPU Implementation of Numpy-compatible percentile
*/

#include "np_percentile_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyPercentileParam);

NNVM_REGISTER_OP(_npi_percentile)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyPercentileParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyPercentileShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyPercentileType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "q"};
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyPercentileForward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);
.add_argument("a", "NDArray-or-Symbol", "The input")
.add_argument("q", "NDArray-or-Symbol", "The input")
.add_arguments(NumpyPercentileParam::__FIELDS__())


}  // namespace op
}  // namespace mxnet
