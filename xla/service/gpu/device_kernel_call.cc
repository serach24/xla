/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/device_kernel_call.h"

#include <cstdint>
#include <string_view>
#include <utility>

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

namespace xla::gpu {

PtxCall PtxCall::Parse(std::string_view backend_config,
                       mlir::MLIRContext* mlir_context) {
  auto attrs = mlir::cast<mlir::DictionaryAttr>(
      mlir::parseAttribute(backend_config, mlir_context));
  auto name = attrs.getAs<mlir::StringAttr>("name").getValue().str();
  auto source = attrs.getAs<mlir::StringAttr>("source").str();
  auto grid_x = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("grid_x").getValue().getSExtValue());
  auto grid_y = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("grid_y").getValue().getSExtValue());
  auto grid_z = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("grid_z").getValue().getSExtValue());
  auto block_x = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("block_x").getValue().getSExtValue());
  auto block_y = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("block_y").getValue().getSExtValue());
  auto block_z = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("block_z").getValue().getSExtValue());
  auto shared_mem = static_cast<int32_t>(
      attrs.getAs<mlir::IntegerAttr>("shared_mem").getValue().getSExtValue());
  return PtxCall{std::move(name), std::move(source), grid_x,  grid_y,
                 grid_z,          block_x,           block_y, block_z};
}

}  // namespace xla::gpu
