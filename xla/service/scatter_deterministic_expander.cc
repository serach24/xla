/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/scatter_deterministic_expander.h"

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/while_util.h"

namespace xla {

// Transposes the given scatter_indices such that the index_vector_dim becomes
// the most-minor dimension.
static absl::StatusOr<HloInstruction*> TransposeIndexVectorDimToLast(
    HloInstruction* scatter_indices, int64_t index_vector_dim) {
  const Shape& scatter_indices_shape = scatter_indices->shape();

  if (scatter_indices_shape.dimensions_size() == index_vector_dim) {
    return scatter_indices;
  }

  if (index_vector_dim == (scatter_indices_shape.dimensions_size() - 1)) {
    return scatter_indices;
  }

  std::vector<int64_t> permutation;
  permutation.reserve(scatter_indices_shape.dimensions_size());
  for (int64_t i = 0, e = scatter_indices_shape.dimensions_size(); i < e; i++) {
    if (i != index_vector_dim) {
      permutation.push_back(i);
    }
  }
  permutation.push_back(index_vector_dim);
  return MakeTransposeHlo(scatter_indices, permutation);
}

// Canonicalizes the scatter_indices tensor in order to keep them uniform while
// performing the scatter operation.
static absl::StatusOr<HloInstruction*> CanonicalizeScatterIndices(
    HloInstruction* scatter_indices, int64_t index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * transposed_scatter_indices,
      TransposeIndexVectorDimToLast(scatter_indices, index_vector_dim));
  if (scatter_indices->shape().rank() == index_vector_dim + 1 &&
      scatter_indices->shape().dimensions(index_vector_dim) == 1) {
    auto new_shape =
        ShapeUtil::DeleteDimension(index_vector_dim, scatter_indices->shape());
    TF_ASSIGN_OR_RETURN(scatter_indices,
                        MakeReshapeHlo(new_shape, scatter_indices));
  }
  bool indices_are_scalar =
      index_vector_dim == scatter_indices->shape().dimensions_size();

  // The number of dimensions in scatter_indices that are index dimensions.
  const int64_t index_dims_in_scatter_indices = indices_are_scalar ? 0 : 1;

  // If there is only one index (i.e. scatter_indices has rank 1 and this
  // scatter is really just a dynamic update slice) add a leading degenerate
  // dimension for uniformity.  Otherwise create a "collapsed" leading dimension
  // that subsumes all of the non-index-vector dimensions.
  const Shape& shape = transposed_scatter_indices->shape();
  if (shape.dimensions_size() == index_dims_in_scatter_indices) {
    return PrependDegenerateDims(transposed_scatter_indices, 1);
  } else {
    // Collapse all but the dimensions (0 or 1) in scatter_indices containing
    // the index vectors.
    return CollapseFirstNDims(
        transposed_scatter_indices,
        shape.dimensions_size() - index_dims_in_scatter_indices);
  }
}

// Permutes the `updates` tensor such that all the scatter dims appear in the
// major dimensions and all the window dimensions appear in the minor
// dimensions.
static absl::StatusOr<HloInstruction*> PermuteScatterAndWindowDims(
    HloInstruction* updates, absl::Span<const int64_t> update_window_dims) {
  std::vector<int64_t> permutation;
  const int64_t updates_rank = updates->shape().rank();
  permutation.reserve(updates_rank);

  for (int64_t i = 0; i < updates_rank; ++i) {
    bool is_scatter_dim = !absl::c_binary_search(update_window_dims, i);
    if (is_scatter_dim) {
      permutation.push_back(i);
    }
  }
  for (auto window_dim : update_window_dims) {
    permutation.push_back(window_dim);
  }

  return MakeTransposeHlo(updates, permutation);
}

// Expands or contracts the scatter indices in the updates tensor.
static absl::StatusOr<HloInstruction*> AdjustScatterDims(
    const Shape& scatter_indices_shape, HloInstruction* updates,
    int64_t index_vector_dim) {
  int64_t num_scatter_dims = scatter_indices_shape.dimensions_size();
  if (index_vector_dim < scatter_indices_shape.dimensions_size()) {
    --num_scatter_dims;
  }
  if (num_scatter_dims == 0) {
    // If there are no scatter dims, this must be a dynamic-update-slice kind of
    // scatter. In this case, we prepend a degenerate dimension to work
    // uniformly in the while loop.
    return PrependDegenerateDims(updates, 1);
  }
  return CollapseFirstNDims(updates, num_scatter_dims);
}

// Expands an index vector from the scatter_indices tensor into a vector that
// can be used to dynamic-update-slice to perform the scatter update.
static absl::StatusOr<HloInstruction*> ExpandIndexVectorIntoOperandSpace(
    HloInstruction* index_vector, const ScatterDimensionNumbers& dim_numbers,
    int64_t operand_rank) {
  HloComputation* computation = index_vector->parent();
  const Shape& index_shape = index_vector->shape();

  // Scatter of a scalar. Return a zero-sized vector of indices.
  if (operand_rank == 0) {
    return computation->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateFromDimensions(index_shape.element_type(), {0})));
  }

  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateFromDimensions(index_shape.element_type(), {1})));

  // We extract out individual components from the smaller index and concatenate
  // them (interspersing zeros as needed) into the larger index.
  std::vector<HloInstruction*> expanded_index_components;

  for (int i = 0; i < operand_rank; i++) {
    int64_t index_vector_dim_index =
        FindIndex(dim_numbers.scatter_dims_to_operand_dims(), i);
    if (index_vector_dim_index !=
        dim_numbers.scatter_dims_to_operand_dims_size()) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * component_to_concat,
          MakeSliceHlo(index_vector, /*start_indices=*/{index_vector_dim_index},
                       /*limit_indices=*/{index_vector_dim_index + 1},
                       /*strides=*/{1}));
      expanded_index_components.push_back(component_to_concat);
    } else {
      expanded_index_components.push_back(zero);
    }
  }

  return MakeConcatHlo(expanded_index_components, /*dimension=*/0);
}


// TODO(chenhao): should I use the expanded indices that I computed already for check index validity?
// or should I use this?
// check if the performance is worse in the former case first, and then consider reuse
// HloComputation* CheckIndexValidity(
//     HloModule* module, 
//     absl::Span<const int64_t> operand_dims,
//     absl::Span<const int64_t> window_sizes) {
//   DCHECK_NE(nullptr, module);
//   DCHECK_EQ(operand_dims.size(), window_sizes.size());

//   // Valid range for the index: [0, operand_dims - window_sizes]

//   HloComputation::Builder builder("check_out_of_bound_computation");
//   auto index = builder.AddInstruction(HloInstruction::CreateParameter(0, index_shape, "base_index"));
//   // broadcast to be the same shape as the index_offset
//   auto broadcasted_index = builder.AddInstruction(HloInstruction::CreateBroadcast(index_offset->shape(), index, {}));
//   // add the index_offset to the base index
//   auto expanded_index = builder.AddInstruction(HloInstruction::CreateBinary(index_offset->shape(), HloOpcode::kAdd, broadcast_index, index_offset));


//   // Check if the index has any negative values.
//   // HloInstruction* zero_index = BroadcastZeros(
//       // computation, index->shape().element_type(), index->shape().dimensions());
//   // Implement a similar thing using builder
//   auto zero_constant = builder.AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0)));
//   auto zero_index = builder.AddInstruction(HloInstruction::CreateBroadcast(index->shape(), zero_constant, {}));
  
//   TF_ASSIGN_OR_RETURN(
//       HloInstruction * negative_index_check,
//       MakeCompareHlo(ComparisonDirection::kLe, zero_index, index));

//   // Check if the index is OOB w.r.t. the operand dimensions and window sizes.
//   std::vector<int64_t> max_valid_index(operand_dims.size());
//   for (int i = 0; i < operand_dims.size(); ++i) {
//     max_valid_index[i] = operand_dims[i] - window_sizes[i];
//   }
//   TF_ASSIGN_OR_RETURN(
//       HloInstruction * max_valid_index_constant,
//       MakeR1ConstantHlo<int64_t>(computation, index->shape().element_type(),
//                                  max_valid_index));
//   TF_ASSIGN_OR_RETURN(HloInstruction * oob_index_check,
//                       MakeCompareHlo(ComparisonDirection::kGe,
//                                      max_valid_index_constant, index));

//   // Combine the results of the two checks above.
//   TF_ASSIGN_OR_RETURN(
//       HloInstruction * valid_index,
//       MakeBinaryHlo(HloOpcode::kAnd, negative_index_check, oob_index_check));

//   // Reduce the index validity check vector into a scalar predicate.
//   auto reduction_init = computation->AddInstruction(
//       HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
//   TF_ASSIGN_OR_RETURN(
//       HloInstruction * valid_index_reduced,
//       MakeReduceHlo(valid_index, reduction_init, HloOpcode::kAnd, module));

//   // Return a broadcasted value of the scalar predicate to the same size as the
//   // window.
//   return MakeBroadcastHlo(valid_index_reduced, {}, window_sizes);
// }

static absl::StatusOr<HloInstruction*> CheckIndexValidity(
    HloComputation* computation, HloInstruction* index,
    absl::Span<const int64_t> operand_dims,
    absl::Span<const int64_t> window_sizes, HloModule* module) {
  DCHECK_NE(nullptr, module);
  DCHECK_EQ(operand_dims.size(), window_sizes.size());

  // Valid range for the index: [0, operand_dims - window_sizes]

  // Check if the index has any negative values.
  HloInstruction* zero_index = BroadcastZeros(
      computation, index->shape().element_type(), index->shape().dimensions());
  TF_ASSIGN_OR_RETURN(
      HloInstruction * negative_index_check,
      MakeCompareHlo(ComparisonDirection::kLe, zero_index, index));

  // Check if the index is OOB w.r.t. the operand dimensions and window sizes.
  std::vector<int64_t> max_valid_index(operand_dims.size());
  for (int i = 0; i < operand_dims.size(); ++i) {
    max_valid_index[i] = operand_dims[i] - window_sizes[i];
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * max_valid_index_constant,
      MakeR1ConstantHlo<int64_t>(computation, index->shape().element_type(),
                                 max_valid_index));
  TF_ASSIGN_OR_RETURN(HloInstruction * oob_index_check,
                      MakeCompareHlo(ComparisonDirection::kGe,
                                     max_valid_index_constant, index));

  // Combine the results of the two checks above.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * valid_index,
      MakeBinaryHlo(HloOpcode::kAnd, negative_index_check, oob_index_check));

  // Reduce the index validity check vector into a scalar predicate.
  auto reduction_init = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * valid_index_reduced,
      MakeReduceHlo(valid_index, reduction_init, HloOpcode::kAnd, module));

  // Return a broadcasted value of the scalar predicate to the same size as the
  // window.
  return valid_index_reduced;
  // return MakeBroadcastHlo(valid_index_reduced, {}, window_sizes);
}

HloComputation* CheckIndexValidityWrapper(
    HloModule* module, 
    absl::Span<const int64_t> operand_dims,
    absl::Span<const int64_t> window_sizes,
    const Shape& index_shape) {
  HloComputation::Builder builder("check_out_of_bound_computation");
  // TODO(chenhao): change shape here
  auto index = builder.AddInstruction(HloInstruction::CreateParameter(0, index_shape, "base_index"));
  // add a fake convert to set the type to be pred
  auto index_reshaped = builder.AddInstruction(HloInstruction::CreateConvert(ShapeUtil::MakeShape(PRED, index_shape.dimensions()), index));

  // // build the computation
  auto computation = builder.Build();
  // auto computation_raw_ptr = computation.release();

  // // call the utility function to check the index validity
  // // TF_ASSIGN_OR_RETURN(HloInstruction * valid_index_reduced,
  // //     CheckIndexValidity(computation_raw_ptr, index, operand_dims, window_sizes, module));
  // auto output = CheckIndexValidity(computation_raw_ptr, index, operand_dims, window_sizes, module).value();
  // computation_raw_ptr->set_root_instruction(output);


  // add the computation to the module
  // return module->AddEmbeddedComputation(std::unique_ptr<HloComputation>(computation_raw_ptr));
  return module->AddEmbeddedComputation(std::move(computation));
}

// Assume the indices are in operand space already
// indices shape: (num_indices, num_dims)
// updates shape: (num_indices,)
HloInstruction* FlattenIndices(HloComputation* parent, HloInstruction* indices, absl::Span<const int64_t> operand_dims) {
  // Step 1: based on the operand_dims, calculate the strides
  Array2D<int64_t> strides(operand_dims.size(), 1);
  int64_t stride = 1;
  for (int i = operand_dims.size() - 1; i >= 0; --i) {
    // strides.push_back(stride);
    strides(i, 0) = stride;
    stride *= operand_dims[i];
  }
  auto strides_tensor = parent->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D<int64_t>(strides)));

  // Step 2: calculate the flattened indices
  auto dot_shape = ShapeUtil::MakeShape(indices->shape().element_type(), {indices->shape().dimensions(0)});
  DotDimensionNumbers dim_numbers;
  dim_numbers.add_lhs_contracting_dimensions(1);
  dim_numbers.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  std::vector<SparsityDescriptor> sparsity;
  absl::Span<HloInstruction* const> sparse_meta;
  auto flattened_indices = parent->AddInstruction(HloInstruction::CreateDot(
    dot_shape,
    indices,
    strides_tensor, 
    dim_numbers,
    precision_config,
    sparsity,
    sparse_meta));
  // todo(chenhao): whether reshape here?
  return flattened_indices;
}

static absl::StatusOr<HloComputation*> CallAndGetOutput(
    HloComputation* original, int output_index) {
  HloInstruction* original_root = original->root_instruction();
  if (!original_root->shape().IsTuple()) {
    return original;
  }
  HloComputation* new_comp = [&] {
    HloComputation::Builder builder(
        absl::StrCat(original->name(), ".dup.", output_index));
    for (int i = 0, n = original->num_parameters(); i < n; ++i) {
      HloInstruction* original_param = original->parameter_instruction(i);
      builder.AddInstruction(HloInstruction::CreateParameter(
          i, original_param->shape(), original_param->name()));
    }
    return original->parent()->AddEmbeddedComputation(builder.Build());
  }();
  HloInstruction* call_original = new_comp->AddInstruction(
      HloInstruction::CreateCall(original_root->shape(),
                                 new_comp->parameter_instructions(), original));
  new_comp->set_root_instruction(
      new_comp->AddInstruction(
          HloInstruction::CreateGetTupleElement(call_original, output_index)),
      /*accept_different_shape=*/true);
  TF_RETURN_IF_ERROR(CallInliner::Inline(call_original).status());
  return new_comp;
}

// Body of the while loop that performs the scatter operation using other HLOs.
static absl::StatusOr<std::vector<HloInstruction*>> ScatterLoopBody(
    HloScatterInstruction* scatter, HloInstruction* induction_var,
    absl::Span<HloInstruction* const> loop_state) {
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();
  CHECK_EQ(loop_state.size(), scatter->operand_count());
  auto operands = loop_state.first(scatter->scatter_operand_count());
  HloInstruction* scatter_indices = loop_state[operands.size()];
  auto updates = loop_state.last(operands.size());

  bool has_scalar_indices = scatter_indices->shape().dimensions_size() == 1;

  // Build a vector form of the induction variable of the while loop.
  HloInstruction* induction_var_as_vector =
      MakeBroadcastHlo(induction_var, /*broadcast_dimensions=*/{},
                       /*result_shape_bounds=*/{1});

  // Pick the index to scatter from scatter_indices based on the induction_var
  // and transform that to an index into the `operand` space.
  HloInstruction* index_vector;
  if (has_scalar_indices) {
    TF_ASSIGN_OR_RETURN(
        index_vector,
        MakeDynamicSliceHlo(scatter_indices, induction_var_as_vector, {1}));
  } else {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * index_into_scatter_indices,
        PadVectorWithZeros(induction_var_as_vector,
                           /*zeros_to_prepend=*/0, /*zeros_to_append=*/1));
    int index_vector_size = scatter_indices->shape().dimensions(1);
    TF_ASSIGN_OR_RETURN(
        HloInstruction * index_vector_2d,
        MakeDynamicSliceHlo(scatter_indices, index_into_scatter_indices,
                            {1, index_vector_size}));
    TF_ASSIGN_OR_RETURN(index_vector,
                        ElideDegenerateDims(index_vector_2d, {0}));
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * scatter_slice_start,
      ExpandIndexVectorIntoOperandSpace(
          index_vector, dim_numbers, operands[0]->shape().dimensions_size()));

  // Extract the slice to be used to update from `updates` tensor for the
  // induction_var corresponding to this iteration of the while loop.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * index_into_updates,
      PadVectorWithZeros(
          induction_var_as_vector, /*zeros_to_prepend=*/0,
          /*zeros_to_append=*/updates[0]->shape().dimensions_size() - 1));
  std::vector<int64_t> update_slice_bounds(
      updates[0]->shape().dimensions().begin(),
      updates[0]->shape().dimensions().end());
  update_slice_bounds[0] = 1;

  absl::InlinedVector<HloInstruction*, 2> map_operands(
      operands.size() + updates.size(), nullptr);
  auto operand_slices_to_update =
      absl::MakeSpan(map_operands).first(operands.size());
  auto update_slices_with_dims_inserted =
      absl::MakeSpan(map_operands).last(updates.size());
  absl::Span<const int64_t> actual_update_slice_dims;
  for (int i = 0, n = operands.size(); i < n; ++i) {
    HloInstruction* update = updates[i];
    TF_ASSIGN_OR_RETURN(
        HloInstruction * update_slice,
        MakeDynamicSliceHlo(update, index_into_updates, update_slice_bounds));
    TF_ASSIGN_OR_RETURN(HloInstruction * update_slice_for_scatter,
                        ElideDegenerateDims(update_slice, {0}));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * update_slice_with_dims_inserted,
        InsertDegenerateDims(update_slice_for_scatter,
                             dim_numbers.inserted_window_dims()));
    update_slices_with_dims_inserted[i] = update_slice_with_dims_inserted;
    // Note that the following transformation assumes that both DynamicSlice and
    // DynamicUpdateSlice follow the same semantics for OOB indices. For
    // example, if there are negative indices and DynamicSlice uses "clamping"
    // semantics, then the extracted data will be "shifted". Since
    // DynamicUpdateSlice also follows the same "clamping" semantics, writing
    // the update will also be "shifted" by exactly the same amount. So, this
    // transformation is correct as long as the semantics of handling OOB
    // indices remain the same in DynamicSlice and DynamicUpdateSlice.

    // Extract the slice to update from `operand` tensor.
    HloInstruction* operand = operands[i];
    const Shape& update_slice_shape = update_slice_with_dims_inserted->shape();
    TF_ASSIGN_OR_RETURN(HloInstruction * operand_slice_to_update,
                        MakeDynamicSliceHlo(operand, scatter_slice_start,
                                            update_slice_shape.dimensions()));
    operand_slices_to_update[i] = operand_slice_to_update;
    if (i == 0) {
      actual_update_slice_dims = update_slice_shape.dimensions();
    } else {
      TF_RET_CHECK(actual_update_slice_dims == update_slice_shape.dimensions());
    }
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * is_index_valid,
      CheckIndexValidity(operands[0]->parent(), scatter_slice_start,
                         operands[0]->shape().dimensions(),
                         actual_update_slice_dims, scatter->GetModule()));

  // Write the updated value of the slice into `operand` tensor.
  std::vector<HloInstruction*> updated_loop_state;
  updated_loop_state.reserve(loop_state.size());
  for (int i = 0, n = operands.size(); i < n; ++i) {
    // Compute the new value for the slice to be updated in `operand` tensor by
    // combining the existing value and the update value using the update
    // computation.
    // NOTE: For scatters with N outputs, we currently have duplicate the Map
    // computation N times because we don't support multioutput Map yet.
    TF_ASSIGN_OR_RETURN(HloComputation * to_apply,
                        CallAndGetOutput(scatter->to_apply(), i));
    TF_ASSIGN_OR_RETURN(HloInstruction * updated_operand_slice,
                        MakeMapHlo(map_operands, to_apply));
    // Select the updated operand only if the index is valid. If not, select the
    // original value.
    TF_ASSIGN_OR_RETURN(HloInstruction * updates_to_apply,
                        MakeSelectHlo(is_index_valid, updated_operand_slice,
                                      operand_slices_to_update[i]));
    TF_ASSIGN_OR_RETURN(HloInstruction * updated_operand,
                        MakeDynamicUpdateSliceHlo(operands[i], updates_to_apply,
                                                  scatter_slice_start));
    updated_loop_state.push_back(updated_operand);
  }
  updated_loop_state.push_back(scatter_indices);
  absl::c_copy(updates, std::back_inserter(updated_loop_state));

  return updated_loop_state;
}

static int64_t ScatterTripCount(const HloScatterInstruction* scatter) {
  // Compute the trip count for the while loop to be used for scatter. This
  // should be the number of indices we should scatter into the operand.
  const HloInstruction* scatter_indices = scatter->scatter_indices();
  const Shape& scatter_indices_shape = scatter_indices->shape();
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();
  int64_t scatter_loop_trip_count = 1;
  for (int64_t i = 0, e = scatter_indices_shape.dimensions_size(); i < e; i++) {
    if (i != dim_numbers.index_vector_dim()) {
      scatter_loop_trip_count *= scatter_indices_shape.dimensions(i);
    }
  }
  return scatter_loop_trip_count;
}

// This is work-inefficient version
HloInstruction* CreateScan(HloComputation* parent, HloInstruction* updates){
  // HloComputation::Builder builder("scan_computation");

  auto updates_shape = updates->shape();
  // Get the length of the input array
  int64_t n = updates_shape.dimensions(0);

  // Calculate the number of iterations needed (log_2(n))
  int64_t log_n = static_cast<int64_t>(std::ceil(std::log2(n)));

  // Placeholder for offset calculation (2^d)
  int64_t offset;

  // Start to traverse
  auto* prev_array = updates;
  HloInstruction* new_array = nullptr;
  for (int64_t d = 0; d < log_n; ++d){
    offset = 1 << d;
    std::vector<int64_t> start_indices = {0};
    std::vector<int64_t> end_indices = {n-offset};
    std::vector<int64_t> strides = {1};
    
    auto shifted_array_shape = ShapeUtil::MakeShape(updates_shape.element_type(), {n-offset});
    auto padding_array_shape = ShapeUtil::MakeShape(updates_shape.element_type(), {offset});

    auto* shifted_array = parent->AddInstruction(
        HloInstruction::CreateSlice(shifted_array_shape, prev_array, start_indices, end_indices, strides));
    auto* padding_array = parent->AddInstruction(
        HloInstruction::CreateBroadcast(padding_array_shape, parent->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0(updates_shape.element_type(), 0))), {}));

    auto* concatenated_array = parent->AddInstruction(
        HloInstruction::CreateConcatenate(updates_shape, {padding_array, shifted_array}, 0));
    new_array = parent->AddInstruction(
        HloInstruction::CreateBinary(updates_shape, HloOpcode::kAdd, prev_array, concatenated_array));
    prev_array = new_array;
  }
  return new_array;
}

// This is work-inefficient version
HloInstruction* CreateScanWithIndices(HloComputation* parent, HloInstruction* updates, HloInstruction* indices){
  // TODO(chenhao) checkings, like make sure they are of the same length


  auto updates_shape = updates->shape();
  auto indices_shape = indices->shape();
  // Get the length of the input array
  int64_t n = updates_shape.dimensions(0);

  // Calculate the number of iterations needed (log_2(n))
  int64_t log_n = static_cast<int64_t>(std::ceil(std::log2(n)));

  // Placeholder for offset calculation (2^d)
  int64_t offset;

  // Start to traverse
  auto* prev_updates = updates;
  auto* prev_indices = indices;
  HloInstruction* new_updates = nullptr;

  std::vector<int64_t> start_indices = {0};
  std::vector<int64_t> strides = {1};

  for (int64_t d = 0; d < log_n; ++d){
    offset = 1 << d;
    std::vector<int64_t> end_indices = {n-offset};
    
    auto shifted_updates_shape = ShapeUtil::MakeShape(updates_shape.element_type(), {n-offset});
    auto padding_updates_shape = ShapeUtil::MakeShape(updates_shape.element_type(), {offset});
 
    auto shifted_indices_shape = ShapeUtil::MakeShape(indices_shape.element_type(), {n-offset});
    auto padding_indices_shape = ShapeUtil::MakeShape(indices_shape.element_type(), {offset});


    auto* shifted_updates = parent->AddInstruction(
        HloInstruction::CreateSlice(shifted_updates_shape, prev_updates, start_indices, end_indices, strides));
    auto* padding_updates = parent->AddInstruction(
        HloInstruction::CreateBroadcast(padding_updates_shape, parent->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0(updates_shape.element_type(), 0))), {}));

    auto* shifted_indices = parent->AddInstruction(
        HloInstruction::CreateSlice(shifted_indices_shape, prev_indices, start_indices, end_indices, strides));
    auto* padding_indices = parent->AddInstruction(
        HloInstruction::CreateBroadcast(padding_indices_shape, parent->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0(indices_shape.element_type(), 0))), {}));

    auto* concatenated_updates = parent->AddInstruction(
        HloInstruction::CreateConcatenate(updates_shape, {padding_updates, shifted_updates}, 0));
    auto* concatenated_indices = parent->AddInstruction(
        HloInstruction::CreateConcatenate(indices_shape, {padding_indices, shifted_indices}, 0));
    
    auto* indices_mask = parent->AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {n}), prev_indices, concatenated_indices, ComparisonDirection::kEq));
    auto* summed_updates = parent->AddInstruction(
        HloInstruction::CreateBinary(updates_shape, HloOpcode::kAdd, prev_updates, concatenated_updates));
    new_updates = parent->AddInstruction(
        HloInstruction::CreateTernary(updates_shape, HloOpcode::kSelect, indices_mask, summed_updates, prev_updates));
    prev_updates = new_updates;
  }
  return new_updates;
}

HloInstruction* ExpandIndexOffsetsFromUpdateShape(HloComputation* parent, const Shape& update_shape, const ScatterDimensionNumbers& dim_num){
  // Calculate the offset tensor for each element of the update tensor.
  // The offset tensor is represented in (num_elements_in_update, index_dim).

  // Calculate the total number of elements in the update tensor

  int64_t num_elements = ShapeUtil::ElementsIn(update_shape);

  // Iterate over every element in the update tensor
  int index_vector_dim = dim_num.index_vector_dim();
  Array2D<int> offset_tensor(num_elements, index_vector_dim);
  for (int64_t linear_index = 0; linear_index < num_elements; ++linear_index) {
    // Calculate the multi-dimensional index from the linear index
    int64_t current_index = linear_index;

    for (int i = index_vector_dim - 1; i >= 0; --i) {
      if (std::find(dim_num.inserted_window_dims().begin(), 
                   dim_num.inserted_window_dims().end(), i) != dim_num.inserted_window_dims().end()) {
        // This dimension does not correspond to the update tensor, so it should remain unchanged
        offset_tensor(linear_index, i) = 0;
      } else{
        int dim_size = update_shape.dimensions(i);
        offset_tensor(linear_index, i)= current_index % dim_size;
        current_index /= dim_size;
      }
   }
  }

  // Return the offset tensor as an HloInstruction
  return parent->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D<int>(offset_tensor)));
}


// For each (index, update) pair, calculate the new indices 
// TODO(maybe could reshape the indices before the map to avoid reshape for indices and mask)
HloInstruction* ExpandIndices(HloComputation* parent, HloInstruction* indices, HloInstruction* index_offsets){
  // The output should still be a 1D tensor and each element being a index
  // The index_offset has the shape of updates, so we need to broadcast the indices to the same shape as the index_offset and add them
  int64_t num_indices = ShapeUtil::ElementsIn(indices->shape());
  int64_t num_offsets = ShapeUtil::ElementsIn(index_offsets->shape());
  
  auto final_shape = ShapeUtil::MakeShape(indices->shape().element_type(), {num_indices, num_offsets});
  auto reshaped_indices = parent->AddInstruction(HloInstruction::CreateReshape(ShapeUtil::MakeShape(indices->shape().element_type(), {ShapeUtil::ElementsIn(indices->shape())}), indices));
  auto reshaped_offsets = parent->AddInstruction(HloInstruction::CreateReshape(ShapeUtil::MakeShape(index_offsets->shape().element_type(), {ShapeUtil::ElementsIn(index_offsets->shape())}), index_offsets));

  auto broadcasted_indices = parent->AddInstruction(HloInstruction::CreateBroadcast(final_shape, reshaped_indices, {0}));
  auto broadcasted_offsets = parent->AddInstruction(HloInstruction::CreateBroadcast(final_shape, reshaped_offsets, {1}));
  // add the index_offset to the base index
  auto expanded_indices = parent->AddInstruction(HloInstruction::CreateBinary(final_shape, HloOpcode::kAdd, broadcasted_indices, broadcasted_offsets));
  return expanded_indices;
}

// For each (index, update) pair, calculate the new updates
HloInstruction* ExpandUpdates(HloComputation* parent, HloInstruction* updates){
  const int64_t num_elements = ShapeUtil::ElementsIn(updates->shape());
  auto expanded_updates = parent->AddInstruction(HloInstruction::CreateReshape(ShapeUtil::MakeShape(updates->shape().element_type(), {num_elements, 1}), updates));
  return expanded_updates;
}

// TODO(chenhao) check if the time or memory is a concern, for now we just write the code in a straightforward way
HloInstruction* ExpandMask(HloComputation* parent, HloInstruction* mask, const Shape& update_shape){
  // first broadcast the mask to the same shape as the expanded_updates, and then flatten it to a 1D tensor
  // TODO(chenhao) check if the mask is a 1D tensor
  // get the number of elements in the update tensor
  int64_t num_elements_in_update = ShapeUtil::ElementsIn(update_shape);
  int64_t mask_length  = mask->shape().dimensions(0);
  auto reshaped_mask = parent->AddInstruction(HloInstruction::CreateReshape(ShapeUtil::MakeShape(PRED, {mask_length}), mask));

  Shape broadcast_shape = ShapeUtil::MakeShape(PRED, {mask_length, num_elements_in_update});

  auto broadcasted_mask = parent->AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, reshaped_mask, {0}));
  auto expanded_mask = parent->AddInstruction(HloInstruction::CreateReshape(ShapeUtil::MakeShape(PRED, {mask_length * num_elements_in_update}), broadcasted_mask));
  return expanded_mask;
}


// Map function that check if the expanded_index of current (index, update) is out of bound  
// output a boolean mask of the same length as the update
// HloComputation* CheckAnyUpdateOutOfBound(HloModule* parent, const Shape& index_shape, HloInstruction* index_offset){
//   // Take the base index of the update, calculate the expanded_index for each value based on index_offset and see if it is out of bound 
//   // This is an HloComputation to be used in a Map operation, and it takes one index as the input
//   HloComputation::Builder builder("check_out_of_bound_computation");
//   auto index = builder.AddInstruction(HloInstruction::CreateParameter(0, index_shape, "base_index"));
//   // broadcast to be the same shape as the index_offset
//   auto broadcasted_index = builder.AddInstruction(HloInstruction::CreateBroadcast(index_offset->shape(), index, {}));
//   // add the index_offset to the base index
//   auto expanded_index = builder.AddInstruction(HloInstruction::CreateBinary(index_offset->shape(), HloOpcode::kAdd, broadcast_index, index_offset));
// }


// TODO(chenhao) add multi-dim support
HloComputation* SortingComparison(HloModule* module, const PrimitiveType indices_type, const PrimitiveType values_type){
  HloComputation::Builder builder("sorting_computation");
  auto key_shape = ShapeUtil::MakeShape(indices_type, {});
  auto value_shape = ShapeUtil::MakeShape(values_type, {});
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(0, key_shape, "lhs_key"));
  auto param1 = builder.AddInstruction(HloInstruction::CreateParameter(1, key_shape, "rhs_key"));
  builder.AddInstruction(HloInstruction::CreateParameter(2, value_shape, "lhs_value"));
  builder.AddInstruction(HloInstruction::CreateParameter(3, value_shape, "rhs_value"));

  builder.AddInstruction(HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0, param1, ComparisonDirection::kLt));

  return module->AddEmbeddedComputation(builder.Build());
}

HloInstruction* CheckValidIndices(HloComputation *parent,
    HloInstruction* indices,
    absl::Span<const int64_t> operand_dims,
    absl::Span<const int64_t> window_sizes){
    // check if indices and indices with the largest offsets are out of bound
    // Essentially we need to do the following:
    // 1. Check base indices >= [0, 0, 0, ...]
    // 2. Check last indices <= [bounds...]
    // 3. For each check, generate a same size tensor, and then do a reduce across rows to get a mask of size (n, 1)

    // 1. Check base indices >= [0, 0, 0, ...]
    // first generate a zero tensor of the same size as the indices
    auto* zero_constant = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0)));
    auto* zero_broadcasted = parent->AddInstruction(
      HloInstruction::CreateBroadcast(ShapeUtil::MakeShape(indices->shape(), zero_constant, {})));
    auto* zero_check = parent->AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, indices->shape().dimensions()), indices, zero_broadcasted, ComparisonDirection::kGe));
    // reduce each row to get a mask
    auto* zero_check_mask = parent->AddInstruction(
      HloInstruction::CreateReduce(ShapeUtil::MakeShape(PRED, {indices->shape().dimensions(0)}), zero_check, {1}, HloOpcode::kAnd));
    
    // 2. Check last indices <= [bounds...]
    // Check if the index is OOB w.r.t. the operand dimensions and window sizes.
    std::vector<int64_t> max_valid_index(operand_dims.size());
    for (int i = 0; i < operand_dims.size(); ++i) {
      max_valid_index[i] = operand_dims[i] - window_sizes[i];
    }

    Literal max_valid_index_literal = LiteralUtil::CreateR1<NativeT>(max_valid_index);
    if (literal.shape().element_type() != indices->shape().element_type()) {
      TF_ASSIGN_OR_RETURN(max_valid_index_literal, literal.Convert(indices->shape().element_type()));
    }
    auto max_valid_index_constant = parent->AddInstruction(
      HloInstruction::CreateConstant(std::move(max_valid_index_literal)));
    auto oob_check = parent->AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, indices->shape().dimensions()), max_valid_index_constant, indices, ComparisonDirection::kGe)); 
    auto oob_check_mask = parent->AddInstruction(
      HloInstruction::CreateReduce(ShapeUtil::MakeShape(PRED, {indices->shape().dimensions(0)}), oob_index_check, {1}, HloOpcode::kAnd));
    
    // Combine the results of the two checks above.
    auto* valid_index_mask = parent->AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(PRED, indices->shape().dimensions()), HloOpcode::kAnd, zero_check_mask, oob_check_mask));
    return valid_index_mask;
}

absl::StatusOr<HloInstruction*> ScatterDeterministicExpander::ExpandInstruction(
    HloInstruction* inst) {
  auto* scatter = Cast<HloScatterInstruction>(inst);
  auto scatter_operands = scatter->scatter_operands();
  HloInstruction* scatter_indices = scatter->scatter_indices();
  auto scatter_updates = scatter->scatter_updates();
  const ScatterDimensionNumbers& dim_numbers =
      scatter->scatter_dimension_numbers();

  // If the updates tensors are empty, there is no need to update the operands.
  // The operands can be forwarded.
  if (ShapeUtil::IsZeroElementArray(scatter_updates[0]->shape())) {
    if (scatter_operands.size() == 1) {
      return scatter_operands[0];
    }
    return scatter->parent()->AddInstruction(
        HloInstruction::CreateTuple(scatter_operands));
  }

  // Compute the trip count for the while loop to be used for scatter. This
  // should be the number of indices we should scatter into the operand.
  int64_t scatter_loop_trip_count = ScatterTripCount(scatter);
  if (!IsInt32(scatter_loop_trip_count)) {
    return Unimplemented(
        "Scatter operations with more than 2147483647 scatter indices are not "
        "supported. This error occurred for %s.",
        scatter->ToString());
  }

   // Canonicalize the scatter_indices, after which the size of its most-major
  // dimension must be same as the while loop trip count.
  TF_ASSIGN_OR_RETURN(HloInstruction * canonical_scatter_indices,
                      CanonicalizeScatterIndices(
                          scatter_indices, dim_numbers.index_vector_dim()));
  CHECK_EQ(scatter_loop_trip_count,
           canonical_scatter_indices->shape().dimensions(0));

  scatter_indices = canonical_scatter_indices;

  bool has_scalar_indices = scatter_indices->shape().dimensions_size() == 1;
  CHECK_EQ(scatter_indices->shape().dimensions_size(), 1);
  if (has_scalar_indices){
    // reshape the indices to be a 2D tensor
    scatter_indices = scatter->parent()->AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(scatter_indices->shape().element_type(), {scatter_loop_trip_count, 1}), scatter_indices));
  }


  // Check if each update is a scalar based on update shape
  // TODO(chenhao) change to actual checking
  bool non_scalar_update = true;

  // // Compute the trip count for the while loop to be used for scatter. This
  // // should be the number of indices we should scatter into the operand.
  // int64_t scatter_loop_trip_count = ScatterTripCount(scatter);
  // if (!IsInt32(scatter_loop_trip_count)) {
  //   return Unimplemented(
  //       "Scatter operations with more than 2147483647 scatter indices are not "
  //       "supported. This error occurred for %s.",
  //       scatter->ToString());
  // }
  HloInstruction* expanded_mask;
  HloInstruction* scatter_update = scatter_updates[0];

  if (non_scalar_update){
    auto updates_shape = scatter_updates[0]->shape();
    std::vector<int64_t> one_update_dimensions(updates_shape.dimensions().begin() + 1, updates_shape.dimensions().end());
    auto update_shape = ShapeUtil::MakeShape(updates_shape.element_type(), one_update_dimensions);
    auto* index_offsets = ExpandIndexOffsetsFromUpdateShape(scatter->parent(), update_shape, dim_numbers);
    // create a map that applies the MapFunc to each (index, update) pair
    // auto* map_func = CheckAnyUpdateOutOfBound(scatter->parent(), index_offsets);
    
    // TODO(chenhao) (URGENT):
    auto update_dims = scatter_updates[0]->shape().dimensions();
    std::vector<int64_t> actual_update_slice_dims(update_dims.begin(), update_dims.end());

    // Insert dimensions of size 1 for each index in inserted_window_dims
    for (int64_t dim : dim_numbers.inserted_window_dims()) {
        actual_update_slice_dims.insert(actual_update_slice_dims.begin() + dim, 1);
    }
    auto index_shape = ShapeUtil::MakeShape(scatter_indices->shape().element_type(), {1, scatter_indices->shape().dimensions(1)});
    auto mask = CheckValidIndices(scatter_indices, scatter_operands[0]->shape().dimensions(), actual_update_slice_dims);
    
    // Expand the mask to the same shape as the expanded_updates
    expanded_mask = ExpandMask(scatter->parent(), oob_check_mask, update_shape);
    
    scatter_indices = ExpandIndices(scatter->parent(), scatter_indices, index_offsets);
    // TODO(chenhao) need to check when to use index 0 and when to use whole updates
    scatter_update = ExpandUpdates(scatter->parent(), scatter_updates[0]);
  }

  // Sort the scatter indices and updates together based on the scatter indices.
  auto* parent = scatter->parent();
  auto* module = scatter->GetModule();
  // Assume scatter_indices is defined and is an HloInstruction pointer
  const Shape& indices_shape = scatter_indices->shape();
  
  // Get the dimensionality of a single index tuple
  // This assumes that each index is a tuple specifying a position in scatter_operands
  int last_dimension = indices_shape.dimensions_size() - 1;
  // int index_tuple_size = indices_shape.dimensions(last_dimension);

  // Extract operand dimensions
  const Shape& operand_shape = scatter_operands[0]->shape();
  auto operand_dims = operand_shape.dimensions();


  // Create the shape for a single index tuple
  auto num_indices = ShapeUtil::ElementsIn(indices_shape);
  // TODO(chenhao) change check to differenciate scalar and non-scalar
  // const Shape& index_shape = ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices, 1});

  const Shape& index_shape = ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices});
  // For now lets just use one-d as a proof of concept
  ScatterDimensionNumbers new_dim_numbers = scatter->scatter_dimension_numbers();
  // auto update_window_dims = dim_numbers.update_window_dims();
  // do a computation for update_window_dims since we changed them to be scalar updates
  // for (int i=1; i < scatter_updates[0]->shape().dimensions_size(); i++){
  //   // update_window_dims.push_back(i);
  //   new_dim_numbers.set_update_window_dims(i-1, i);
  // }
  new_dim_numbers.set_scatter_dims_to_operand_dims(0,0);
  new_dim_numbers.clear_update_window_dims();
  // new_dim_numbers.set_index_vector_dim(0);
  // TODO(chenhao) do some checking here
  // new_dim_numbers.set_update_window_dims(update_window_dims);

  scatter_update = parent->AddInstruction(
    HloInstruction::CreateReshape(ShapeUtil::MakeShape(scatter_update->shape().element_type(), {num_indices}), scatter_update));
  scatter_indices = parent->AddInstruction(
    HloInstruction::CreateReshape(index_shape, scatter_indices));

  auto* comparison= SortingComparison(module, index_shape.element_type(), scatter_updates[0]->shape().element_type());
  int64_t sort_dimension = 0;

  // operands.insert(operands.end(), scatter_updates.begin(), scatter_updates.end());
  std::vector<HloInstruction*> operands = {scatter_indices, scatter_update};

  auto* sorting = parent->AddInstruction(
    HloInstruction::CreateSort(ShapeUtil::MakeTupleShape({scatter_indices->shape(), scatter_update->shape()}), sort_dimension, operands, comparison, false));
  auto* sorted_indices = parent->AddInstruction(
    HloInstruction::CreateGetTupleElement(scatter_indices->shape(), sorting, 0));
  auto* sorted_updates = parent->AddInstruction(
    HloInstruction::CreateGetTupleElement(scatter_update->shape(), sorting, 1));
  // Compute the scan
  // auto* prefix_scan_updates = CreateScan(parent, sorted_updates);
  auto* prefix_scan_updates = CreateScanWithIndices(parent, sorted_updates, sorted_indices);

  // Compute which unique indices to write
  int64_t indices_len = sorted_indices->shape().dimensions(0);
  auto* sorted_indices_preceding_part = parent->AddInstruction(
    HloInstruction::CreateSlice(ShapeUtil::MakeShape(index_shape.element_type(), {indices_len-1}), sorted_indices, {0}, {indices_len-1}, {1}));
  auto* sorted_indices_following_part = parent->AddInstruction(
    HloInstruction::CreateSlice(ShapeUtil::MakeShape(index_shape.element_type(), {indices_len-1}), sorted_indices, {1}, {indices_len}, {1}));
  auto* indices_mask_without_padding = parent->AddInstruction(
    HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {indices_len-1}), sorted_indices_preceding_part, sorted_indices_following_part, ComparisonDirection::kNe));
  // Pad the comparison with a true value at the end
  auto* true_constant = parent->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto* padding = parent->AddInstruction(HloInstruction::CreateBroadcast(ShapeUtil::MakeShape(PRED, {1}), true_constant, {})); 
  std::vector<HloInstruction *> padding_operands = {indices_mask_without_padding, padding};
  auto* indices_mask = parent->AddInstruction(
    HloInstruction::CreateConcatenate(ShapeUtil::MakeShape(PRED, {indices_len}), padding_operands, 0));

  // Mask the indices
  // TODO(chenhao) : change the outofbound index to support multi-dim
  auto* output_len_constant = parent->AddInstruction(HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(scatter->shape().dimensions(0))));
  auto* out_of_bound_tensor = parent->AddInstruction(HloInstruction::CreateBroadcast(index_shape, output_len_constant, {}));
  auto* masked_indices = parent->AddInstruction(
    HloInstruction::CreateTernary(index_shape, HloOpcode::kSelect, indices_mask, sorted_indices, out_of_bound_tensor));
  
  if (non_scalar_update){
    // If non-scalar, we need to apply the oob_check_mask again onto the masked_indices
    masked_indices = parent->AddInstruction(
      HloInstruction::CreateTernary(index_shape, HloOpcode::kSelect, expanded_mask, masked_indices, out_of_bound_tensor));
  }



  // Finally, recreate the scatter instruction with unique indices
  // TODO(chenhao): need to figure out how to handle multiple operands 
  // auto* new_scatter = parent->AddInstruction(
  //   HloInstruction::CreateScatter(scatter->shape(), scatter_operands[0], masked_indices, prefix_scan_updates, scatter->to_apply(), scatter->scatter_dimension_numbers(), true/*indices_are_sorted*/, true/*unique_indices*/));
  // todo(chenhao) reshape the scatter_operands
  auto operand = parent->AddInstruction(
    HloInstruction::CreateReshape(ShapeUtil::MakeShape(scatter_operands[0]->shape().element_type(), {ShapeUtil::ElementsIn(scatter_operands[0]->shape())}), scatter_operands[0]));
  auto temp_scatter_shape = ShapeUtil::MakeShape(scatter->shape().element_type(), {ShapeUtil::ElementsIn(scatter->shape())});
  auto* new_scatter = parent->AddInstruction(
    HloInstruction::CreateScatter(temp_scatter_shape, operand, masked_indices, prefix_scan_updates, scatter->to_apply(), new_dim_numbers, true/*indices_are_sorted*/, true/*unique_indices*/));
  
  new_scatter = parent->AddInstruction(
    HloInstruction::CreateReshape(scatter->shape(), new_scatter));
 
  
  return new_scatter;
}

namespace {

bool IsCombinerAssociative(const HloComputation* combiner) {
  // Consider simple binary combiner functions only.
  if (combiner->instruction_count() != 3) {
    return false;
  }
  switch (combiner->root_instruction()->opcode()) {
    // Minimum and Maximum are common associative combiners.
    case HloOpcode::kMinimum:
    case HloOpcode::kMaximum:
      return true;
    // Other common combiners are associative at least for integer arithmetic.
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      return combiner->root_instruction()->shape().IsInteger();
    default:
      return false;
  }
}

bool IsDeterministic(const HloScatterInstruction* scatter) {
  if (scatter->unique_indices()) {
    return true;
  }
  if (IsCombinerAssociative(scatter->to_apply())) {
    return true;
  }
  return false;
}

}  // namespace

bool ScatterDeterministicExpander::InstructionMatchesPattern(HloInstruction* inst) {
  auto* scatter = DynCast<HloScatterInstruction>(inst);
  return (scatter != nullptr) && !IsDeterministic(scatter);
}

}  // namespace xla
