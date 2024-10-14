/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/scatter_determinism_expander.h"
#include <cstdint>
#include "absl/container/flat_hash_set.h"

#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "xla/service/scatter_utils.h"

namespace xla {

// Canonicalizes the scatter_updates in order to keep them uniform while
// performing the scatter operation.
static StatusOr<std::vector<HloInstruction*>> CanonicalizeScatterUpdates(
    const std::vector<HloInstruction*>& scatter_updates,
    HloInstruction* scatter_indices, const ScatterDimensionNumbers& dim_numbers,
    int64_t scatter_loop_trip_count) {
  std::vector<HloInstruction*> adjusted_updates;
  adjusted_updates.reserve(scatter_updates.size());
  for (HloInstruction* update : scatter_updates) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * canonical_update,
        PermuteScatterAndWindowDims(update, dim_numbers.update_window_dims()));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * adjusted_update,
        AdjustScatterDims(scatter_indices->shape(), canonical_update,
                          dim_numbers.index_vector_dim()));
    CHECK_EQ(scatter_loop_trip_count, adjusted_update->shape().dimensions(0));
    adjusted_updates.push_back(adjusted_update);
  }
  return adjusted_updates;
}

// Creates a tensor for the scatter operation based on the value of
// is_out_of_bound.
//
// When is_out_of_bound is true, the tensor is filled with values representing
// the maximum bounds of the scatter shape (out-of-bound values). This is used
// to simulate out-of-bound conditions in the scatter operation.
//
// When is_out_of_bound is false, the tensor is filled with the maximum valid
// indices (calculated as operand_dimensions - window_dimensions). This is used
// to check whether indices are within valid bounds for non-scalar updates.
//
// This function is reusable for both out-of-bound tensor generation and valid
// index checks in scatter operations with non-scalar updates.
HloInstruction* CreateBoundTensor(
    HloComputation* parent, HloInstruction* scatter_indices,
    absl::Span<const int64_t> operand_dims, bool is_out_of_bound = false,
    absl::optional<absl::Span<const int64_t>> window_sizes = absl::nullopt) {
  if (scatter_indices->shape().rank() == 1) {
    CHECK(operand_dims.size() == 1);
    int32_t value = is_out_of_bound
                        ? operand_dims[0]
                        : operand_dims[0] - (*window_sizes)[0];
    Array<int32_t> out_of_bound_array({scatter_indices->shape().dimensions(0)},
                                      value);
    return parent->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateFromArray(out_of_bound_array)));
  }
  // More than one dimension in scatter_indices
  Array2D<int32_t> out_of_ound_array(scatter_indices->shape().dimensions(0),
                                     scatter_indices->shape().dimensions(1));
  for (int i = 0; i < scatter_indices->shape().dimensions(0); ++i) {
    for (int j = 0; j < scatter_indices->shape().dimensions(1); ++j) {
      out_of_ound_array(i, j) =
          is_out_of_bound ? operand_dims[j]
                          : operand_dims[j] - (*window_sizes)[j];
    }
  }
  return parent->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D<int>(out_of_ound_array)));
}

// indices shape: (num_indices, num_dims)
// updates shape: (num_indices,)
HloInstruction* FlattenIndices(HloComputation* parent, HloInstruction* indices,
                               absl::Span<const int64_t> operand_dims) {
  if (indices->shape().rank() == 1) {
    return indices;
  }
  if (operand_dims.size() == 1) {
    return parent->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(indices->shape().element_type(),
                             {indices->shape().dimensions(0)}),
        indices));
  }
  // Step 1: based on the operand_dims, calculate the strides
  Array2D<int64_t> strides(operand_dims.size(), 1);
  int64_t stride = 1;
  for (int i = operand_dims.size() - 1; i >= 0; --i) {
    strides(i, 0) = stride;
    stride *= operand_dims[i];
  }
  auto strides_tensor = parent->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D<int64_t>(strides)));

  // Step 2: calculate the flattened indices
  auto dot_shape = ShapeUtil::MakeShape(indices->shape().element_type(),
                                        {indices->shape().dimensions(0), 1});
  DotDimensionNumbers dim_numbers;
  dim_numbers.add_lhs_contracting_dimensions(1);
  dim_numbers.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  auto flattened_indices = parent->AddInstruction(HloInstruction::CreateDot(
      dot_shape, indices, strides_tensor, dim_numbers, precision_config));
  return parent->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(indices->shape().element_type(),
                           {indices->shape().dimensions(0)}),
      flattened_indices));
}

// Computation for sorting the scalar scatter indices and updates together
static HloComputation* SortingComparison(HloModule* module,
                                         const PrimitiveType indices_type,
                                         const PrimitiveType updates_type,
                                         int64_t num_updates,
                                         bool has_scalar_indices) {
  Shape key_shape = ShapeUtil::MakeShape(indices_type, {});
  Shape update_shape = ShapeUtil::MakeShape(updates_type, {});
  HloComputation::Builder builder("sorting_computation");
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, key_shape, "lhs_key"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, key_shape, "rhs_key"));
  int param_count = 2;
  for (int i = 0; i < num_updates; ++i) {
    builder.AddInstruction(HloInstruction::CreateParameter(
        param_count, update_shape, absl::StrFormat("lhs_update_%d", i)));
    builder.AddInstruction(HloInstruction::CreateParameter(
        param_count + 1, update_shape, absl::StrFormat("rhs_update_%d", i)));
    param_count += 2;
  }
  if (!has_scalar_indices) {
    builder.AddInstruction(HloInstruction::CreateParameter(
        param_count, key_shape, "lhs_permutation"));
    builder.AddInstruction(HloInstruction::CreateParameter(
        param_count + 1, key_shape, "rhs_permutation"));
  }
  builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0,
                                    param1, ComparisonDirection::kLt));
  return module->AddEmbeddedComputation(builder.Build());
}

static std::vector<HloInstruction*> SortIndicesAndUpdates(
    HloInstruction* scatter_indices,
    const std::vector<HloInstruction*>& scatter_updates, int64_t num_indices,
    HloScatterInstruction* scatter, HloComputation* parent,
    absl::Span<const int64_t> operand_dims, bool has_scalar_indices) {
  const Shape& indices_shape = scatter_indices->shape();
  const Shape& updates_shape = scatter_updates[0]->shape();
  auto updates_dims = updates_shape.dimensions();
  // Since we canonicalized the scatter updates, the first dim will always be
  // the number of updates and the rest will be the shape of each update
  HloInstruction* scalar_indices =
      FlattenIndices(scatter->parent(), scatter_indices, operand_dims);

  // Create the shape for a single index tuple
  // Create [0...num_indices] tensor for permutation in sorting
  auto indices_permutation = parent->AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices}), 0));

  std::vector<int64_t> single_update_dimensions(updates_dims.begin() + 1,
                                                updates_dims.end());

  const Shape update_shape = ShapeUtil::MakeShape(updates_shape.element_type(),
                                                  single_update_dimensions);

  const Shape& scalar_index_shape =
      ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices});
  auto* comparison = SortingComparison(
      scatter->GetModule(), indices_shape.element_type(),
      updates_shape.element_type(), scatter_updates.size(), has_scalar_indices);

  // The sorting operation contains the scalar indices and the updates, and if
  // the scatter indices were not scalar, the sorting operation will also
  // contain the indices permutation
  std::vector<HloInstruction*> sort_operands = {scalar_indices};
  std::vector<Shape> sort_shapes = {scalar_index_shape};
  for (auto update : scatter_updates) {
    sort_operands.push_back(update);
    sort_shapes.push_back(update->shape());
  }
  if (!has_scalar_indices) {
    sort_operands.push_back(indices_permutation);
    sort_shapes.push_back(indices_permutation->shape());
  }

  auto* sorting = parent->AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeTupleShape(sort_shapes), 0, sort_operands, comparison,
      false /*is_stable*/));
  auto* sorted_scalar_indices =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          scalar_indices->shape(), sorting, 0));

  std::vector<HloInstruction*> sorted_updates(scatter_updates.size());
  for (int i = 0; i < scatter_updates.size(); i++) {
    sorted_updates[i] =
        parent->AddInstruction(HloInstruction::CreateGetTupleElement(
            scatter_updates[i]->shape(), sorting, i + 1));
  }
  std::vector<HloInstruction*> sorted_tensors = {sorted_scalar_indices};
  sorted_tensors.insert(sorted_tensors.end(), sorted_updates.begin(),
                        sorted_updates.end());
  if (has_scalar_indices) {
    return sorted_tensors;
  }
  // When the scatter indices were not scalar, need to return the sorted scatter
  // indices
  auto* sorted_indices_arg =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          indices_permutation->shape(), sorting, sorted_tensors.size()));
  sorted_indices_arg = parent->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(sorted_indices_arg->shape().element_type(),
                           {num_indices, 1}),
      sorted_indices_arg));
  // Use gather of sorted_indices_arg to get the sorted original indices
  GatherDimensionNumbers gather_dim_numbers;
  gather_dim_numbers.add_offset_dims(
      1);  // Preserving the inner dimension (columns)
  gather_dim_numbers.add_start_index_map(
      0);  // Mapping start_indices to the first dimension of the operand
  gather_dim_numbers.add_collapsed_slice_dims(0);
  gather_dim_numbers.set_index_vector_dim(1);
  std::vector<int64_t> slice_sizes = {1,
                                      scatter_indices->shape().dimensions(1)};
  auto* sorted_expanded_indices =
      parent->AddInstruction(HloInstruction::CreateGather(
          scatter_indices->shape(), scatter_indices, sorted_indices_arg,
          gather_dim_numbers, slice_sizes,
          /*indices_are_sorted=*/true));
  sorted_tensors.push_back(sorted_expanded_indices);
  return sorted_tensors;
}

// CreateScanWithIndices performs a prefix scan operation (akin to parallel
// prefix sum) on the updates and indices, to compute the accumulated updates in
// log(n) time.
//
// High-level algorithm:
//
// Iteration through log2(num_updates):
//   - For each iteration, the `updates` tensor will be sliced and padded to
//   perform shifting by `offset`.
//   - Similarly, the `indices` tensor is also sliced and padded.
//   - A mask is created that compares each element of shifted `indices` and
//   original `indices` are equal (used to avoid combining updates from
//   different indices).
//   - The `to_apply` function is used to combine the original and shifted
//   updates to generate a combined update tensor.
//   - Based on the mask, the new update tensor will choose from either the
//   combined update or the original update.
//   - The result becomes the `new_updates`, which is then used as the
//   input for the next iteration.
static StatusOr<HloInstruction*> CreateScanWithIndices(
    HloComputation* parent, HloInstruction* updates, HloInstruction* indices,
    HloComputation* to_apply) {
  const Shape& updates_shape = updates->shape();
  const Shape& indices_shape = indices->shape();
  // Get the length of the input array
  int64_t num_updates = updates_shape.dimensions(0);

  // Calculate the number of iterations needed (log_2(n))
  int64_t log_n = static_cast<int64_t>(std::ceil(std::log2(num_updates)));

  // Placeholder for offset calculation (2^d)
  int64_t offset;

  // Start to traverse
  HloInstruction* prev_updates = updates;
  HloInstruction* prev_indices = indices;
  HloInstruction* new_updates = nullptr;

  std::vector<int64_t> start_indices = {0};
  std::vector<int64_t> strides = {1};

  for (int64_t iteration = 0; iteration < log_n; ++iteration) {
    offset = 1 << iteration;
    std::vector<int64_t> end_indices = {num_updates - offset};

    auto shifted_updates_shape = ShapeUtil::MakeShape(
        updates_shape.element_type(), {num_updates - offset});
    auto padding_updates_shape =
        ShapeUtil::MakeShape(updates_shape.element_type(), {offset});

    auto shifted_indices_shape = ShapeUtil::MakeShape(
        indices_shape.element_type(), {num_updates - offset});
    auto padding_indices_shape =
        ShapeUtil::MakeShape(indices_shape.element_type(), {offset});

    auto* shifted_updates = parent->AddInstruction(
        HloInstruction::CreateSlice(shifted_updates_shape, prev_updates,
                                    start_indices, end_indices, strides));
    auto* padding_updates =
        parent->AddInstruction(HloInstruction::CreateBroadcast(
            padding_updates_shape,
            parent->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0(updates_shape.element_type(), 0))),
            {}));

    auto* shifted_indices = parent->AddInstruction(
        HloInstruction::CreateSlice(shifted_indices_shape, prev_indices,
                                    start_indices, end_indices, strides));
    auto* padding_indices =
        parent->AddInstruction(HloInstruction::CreateBroadcast(
            padding_indices_shape,
            parent->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0(indices_shape.element_type(), 0))),
            {}));

    auto* concatenated_updates =
        parent->AddInstruction(HloInstruction::CreateConcatenate(
            updates_shape, {padding_updates, shifted_updates}, 0));
    auto* concatenated_indices =
        parent->AddInstruction(HloInstruction::CreateConcatenate(
            indices_shape, {padding_indices, shifted_indices}, 0));

    auto* indices_mask = parent->AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {num_updates}), prev_indices,
        concatenated_indices, ComparisonDirection::kEq));
    std::vector<HloInstruction*> map_operands = {prev_updates,
                                                 concatenated_updates};
    TF_ASSIGN_OR_RETURN(HloInstruction * reduced_updates,
                        MakeMapHlo(map_operands, to_apply));
    new_updates = parent->AddInstruction(HloInstruction::CreateTernary(
        updates_shape, HloOpcode::kSelect, indices_mask, reduced_updates,
        prev_updates));
    prev_updates = new_updates;
  }
  return new_updates;
}

StatusOr<std::vector<HloInstruction*>> ComputePrefixScan(
    const std::vector<HloInstruction*>& sorted_updates,
    HloInstruction* sorted_scalar_indices, HloScatterInstruction* scatter,
    HloComputation* parent) {
  std::vector<HloInstruction*> prefix_scans(sorted_updates.size());
  for (int i = 0; i < sorted_updates.size(); i++) {
    // TODO(chenhao) change to use the extracted computation
    TF_ASSIGN_OR_RETURN(
        HloComputation * to_apply,
        CallComputationAndGetIthOutputWithBinaryParams(scatter->to_apply(), i));
    TF_ASSIGN_OR_RETURN(prefix_scans[i],
                        CreateScanWithIndices(parent, sorted_updates[i],
                                              sorted_scalar_indices, to_apply));
  }
  return prefix_scans;
}

static HloInstruction* FindLastOccurrenceIndices(
    HloInstruction* sorted_indices, HloInstruction* sorted_scalar_indices,
    HloInstruction* scatter, HloComputation* parent, int64_t num_indices,
    HloInstruction* out_of_bound_tensor) {
  int64_t indices_len = sorted_indices->shape().dimensions(0);
  const PrimitiveType& indices_type = sorted_indices->shape().element_type();
  auto* sorted_indices_preceding_part =
      parent->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(indices_type, {indices_len - 1}),
          sorted_scalar_indices, {0}, {indices_len - 1}, {1}));
  auto* sorted_indices_following_part =
      parent->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(indices_type, {indices_len - 1}),
          sorted_scalar_indices, {1}, {indices_len}, {1}));
  auto* indices_mask_without_padding =
      parent->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PRED, {indices_len - 1}),
          sorted_indices_preceding_part, sorted_indices_following_part,
          ComparisonDirection::kNe));
  // Pad the comparison with a true value at the end
  auto* true_constant = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto* padding = parent->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(PRED, {1}), true_constant, {}));
  std::vector<HloInstruction*> padding_operands = {indices_mask_without_padding,
                                                   padding};
  auto* indices_mask = parent->AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(PRED, {indices_len}), padding_operands, 0));

  // Mask the indices
  indices_mask = parent->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(PRED, sorted_indices->shape().dimensions()),
      indices_mask, {0}));

  auto* masked_indices = parent->AddInstruction(HloInstruction::CreateTernary(
      sorted_indices->shape(), HloOpcode::kSelect, indices_mask, sorted_indices,
      out_of_bound_tensor));
  return masked_indices;
}

HloInstruction* ExpandIndexOffsetsFromUpdateShape(
    HloComputation* parent, const Shape& update_shape,
    const ScatterDimensionNumbers& dim_num, const Shape& operand_shape) {
  // Calculate the offset tensor for each element of the update tensor.
  // The offset tensor is represented in (num_elements_in_update, index_dim).

  int64_t num_elements = ShapeUtil::ElementsIn(update_shape);
  int64_t operand_rank = operand_shape.dimensions_size();
  Array2D<int> offset_tensor(num_elements, operand_rank);

  std::vector<bool> is_inserted_window_dims(operand_rank, false);
  for (int dim : dim_num.inserted_window_dims()) {
    is_inserted_window_dims[dim] = true;
  }

  for (int64_t linear_index = 0; linear_index < num_elements; ++linear_index) {
    // Calculate the multi-dimensional index from the linear index
    int64_t current_index = linear_index;
    int inserted_window_dim_size = 0;
    // Handle 0th to (operand_rank-2)th dimensions
    for (int i = 0; i < operand_rank - 1; ++i) {
      if (is_inserted_window_dims[i]) {
        inserted_window_dim_size++;
        offset_tensor(linear_index, i) = 0;
      } else {
        // When computing the multi-dimensional index, we want to divide by the
        // next dimension size, so we need to add 1. We also wants to skip the
        // inserted window dims
        int dim_size =
            update_shape.dimensions(i + 1 - inserted_window_dim_size);
        offset_tensor(linear_index, i) = current_index / dim_size;
        current_index %= dim_size;
      }
    }
    // Handle (operand_rank-1)th dimension
    if (is_inserted_window_dims[operand_rank - 1]) {
      offset_tensor(linear_index, operand_rank - 1) = 0;
    } else {
      offset_tensor(linear_index, operand_rank - 1) = current_index;
    }
  }

  // Return the offset tensor as an HloInstruction
  return parent->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2FromArray2D<int>(offset_tensor)));
}

// For each (index, update) pair, calculate the new indices
HloInstruction* ExpandIndices(HloComputation* parent, HloInstruction* indices,
                              HloInstruction* index_offsets) {
  // For each index we need to add the index_offset to the base index
  // To do that, we first broadcast the indices and index_offsets to the same
  // shape, then add the index_offset to the base index and flatten the
  // result Broadcast to be (num_indices, length_of_index_offsets,
  // length_of_indices).

  // If the indices is scalar, return indices directly
  if (indices->shape().dimensions_size() == 1) {
    return indices;
  }

  int64_t num_indices = indices->shape().dimensions(0);
  int64_t num_offsets = index_offsets->shape().dimensions(0);
  int64_t index_length = indices->shape().dimensions(1);

  Shape final_shape =
      ShapeUtil::MakeShape(indices->shape().element_type(),
                           {num_indices, num_offsets, index_length});
  auto broadcasted_indices = parent->AddInstruction(
      HloInstruction::CreateBroadcast(final_shape, indices, {0, 2}));
  auto broadcasted_offsets = parent->AddInstruction(
      HloInstruction::CreateBroadcast(final_shape, index_offsets, {1, 2}));
  auto expanded_indices = parent->AddInstruction(HloInstruction::CreateBinary(
      final_shape, HloOpcode::kAdd, broadcasted_indices, broadcasted_offsets));
  // Flatten the result to be (num_indices * num_offsets, index_length)
  return parent->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(indices->shape().element_type(),
                           {num_indices * num_offsets, index_length}),
      expanded_indices));
}

// Function to create a reduction computation for logical AND
HloComputation* ReduceAndComputation(HloModule* module) {
  // Create a computation builder
  HloComputation::Builder builder("reduce_logical_and");

  // Define the scalar shape for boolean operations
  const Shape bool_shape = ShapeUtil::MakeShape(PRED, {});

  // Add parameters for the reduction computation.
  // These represent the elements to be combined (lhs and rhs).
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bool_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bool_shape, "rhs"));

  // Create the logical AND operation between the two parameters
  builder.AddInstruction(
      HloInstruction::CreateBinary(bool_shape, HloOpcode::kAnd, lhs, rhs));

  // Build and return the computation object
  return module->AddEmbeddedComputation(builder.Build());
}

absl::StatusOr<HloInstruction*> CheckValidIndices(
    HloComputation* parent, HloInstruction* indices,
    absl::Span<const int64_t> operand_dims,
    absl::Span<const int64_t> window_sizes) {
  // check if indices and indices with the largest offsets are out of bound
  // Essentially we need to do the following:
  // 1. Check base indices >= [0, 0, 0, ...]
  // 2. Check last indices <= [bounds...]
  // 3. For each check, generate a same size tensor, and then do a reduce across
  // rows to get a mask of size (n, 1)
  auto init_reduce_value = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto reduce_computation = ReduceAndComputation(parent->parent());

  // 1. Check base indices >= [0, 0, 0, ...]
  // first generate a zero tensor of the same size as the indices
HloInstruction* zero_check_mask;
if (indices->shape().rank() == 1) {
  // Scalar case: Directly compare the scalar index to zero.
  auto* zero_constant = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  zero_check_mask = parent->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}), indices, zero_constant, 
      ComparisonDirection::kGe));
} else {
  // Non-scalar case: Broadcast a zero tensor and compare element-wise.
  auto* zero_constant = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  auto* zero_broadcasted = parent->AddInstruction(
      HloInstruction::CreateBroadcast(indices->shape(), zero_constant, {}));
  
  // Compare each index to the zero tensor.
  auto* zero_check = parent->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, indices->shape().dimensions()), indices,
      zero_broadcasted, ComparisonDirection::kGe));
  
  // Reduce across rows to get a mask (for multi-dimensional indices).
  zero_check_mask = parent->AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(PRED, {indices->shape().dimensions(0)}), zero_check,
      init_reduce_value, {1}, reduce_computation));
}
  // auto* zero_constant = parent->AddInstruction(
  //     HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  // auto* zero_broadcasted = parent->AddInstruction(
  //     HloInstruction::CreateBroadcast(indices->shape(), zero_constant, {}));
  // auto* zero_check = parent->AddInstruction(HloInstruction::CreateCompare(
  //     ShapeUtil::MakeShape(PRED, indices->shape().dimensions()), indices,
  //     zero_broadcasted, ComparisonDirection::kGe));
  // // reduce each row to get a mask
  // auto* zero_check_mask = parent->AddInstruction(HloInstruction::CreateReduce(
  //     ShapeUtil::MakeShape(PRED, {indices->shape().dimensions(0)}), zero_check,
  //     init_reduce_value, {1}, reduce_computation));

  // 2. Check last indices <= [bounds...]
  // Check if the index is OOB w.r.t. the operand dimensions and window sizes.
  auto max_valid_index_constant = CreateBoundTensor(parent, indices, operand_dims, false, window_sizes);
  auto oob_check = parent->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, indices->shape().dimensions()),
      max_valid_index_constant, indices, ComparisonDirection::kGe));
  auto oob_check_mask = parent->AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(PRED, {indices->shape().dimensions(0)}), oob_check,
      init_reduce_value, {1}, reduce_computation));

  // Combine the results of the two checks above.
  auto* valid_index_mask = parent->AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {indices->shape().dimensions(0)}),
      HloOpcode::kAnd, zero_check_mask, oob_check_mask));
  return parent->AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(PRED, indices->shape().dimensions()), oob_check_mask,
      {0}));
}

StatusOr<HloInstruction*> ScatterDeterminismExpander::ExpandInstruction(
    HloInstruction* inst) {
  auto* scatter = Cast<HloScatterInstruction>(inst);
  auto scatter_operands = scatter->scatter_operands();
  HloInstruction* scatter_indices = scatter->scatter_indices();
  std::vector<HloInstruction*> scatter_updates(
      scatter->scatter_updates().begin(), scatter->scatter_updates().end());
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
  int64_t scatter_indices_count = ScatterIndicesCount(scatter);
  if (!IsInt32(scatter_indices_count)) {
    // 2147483647 is the maximum value for a 32-bit signed integer (INT32_MAX).
    return Unimplemented(
        "Scatter operations with more than 2147483647 scatter indices are not "
        "supported. This error occurred for %s.",
        scatter->ToString());
  }

  bool has_scalar_indices = scatter_indices->shape().dimensions_size() == 1 ||
                            scatter_indices->shape().dimensions(1) == 1;
  // Canonicalize the scatter_indices, after which the size of its most-major
  // dimension must be same as the while loop trip count.
  TF_ASSIGN_OR_RETURN(scatter_indices,
                      CanonicalizeScatterIndices(
                          scatter_indices, dim_numbers.index_vector_dim()));
  CHECK_EQ(scatter_indices_count, scatter_indices->shape().dimensions(0));

  // Canonicalize the updates, after which the size of their most-major
  // dimensions must be same as the while loop trip count.
  TF_ASSIGN_OR_RETURN(scatter_updates, CanonicalizeScatterUpdates(
                                           scatter_updates, scatter_indices,
                                           dim_numbers, scatter_indices_count));

  CHECK(scatter_indices->shape().dimensions_size() >= 1);

  HloComputation* parent = scatter->parent();
  auto updates_shape = scatter_updates[0]->shape();
  auto updates_dims = scatter_updates[0]->shape().dimensions();
  // Since we canonicalized the scatter updates, the first dim will always be
  // the number of updates and the rest will be the shape of each update
  std::vector<int64_t> one_update_dimensions(updates_dims.begin() + 1,
                                             updates_dims.end());
  const Shape& update_shape =
      ShapeUtil::MakeShape(updates_shape.element_type(), one_update_dimensions);

  ScatterDimensionNumbers new_dim_numbers;
  // Check if each update is a scalar based on update shape
  bool non_scalar_update = scatter_updates[0]->shape().dimensions_size() > 1;

  HloInstruction* out_of_bound_tensor =
      CreateBoundTensor(parent, scatter_indices, scatter->shape().dimensions());

  if (non_scalar_update) {
    // Extract operand dimensions
    const Shape& operand_shape = scatter_operands[0]->shape();

    auto* index_offsets = ExpandIndexOffsetsFromUpdateShape(
        scatter->parent(), update_shape, dim_numbers, operand_shape);

    int num_operand_dims = operand_shape.dimensions_size();
    std::vector<int64_t> actual_update_window_dims(num_operand_dims);
    int update_dim_index = 0;
    for (int i = 0; i < num_operand_dims; ++i) {
      if (std::find(dim_numbers.inserted_window_dims().begin(),
                    dim_numbers.inserted_window_dims().end(),
                    i) != dim_numbers.inserted_window_dims().end()) {
        actual_update_window_dims[i] = 1;
      } else {
        actual_update_window_dims[i] =
            update_shape.dimensions(update_dim_index);
        update_dim_index++;
      }
    }

    auto index_shape =
        ShapeUtil::MakeShape(scatter_indices->shape().element_type(),
                             {1, scatter_indices->shape().dimensions(1)});

    // if any updates are out of bound, we change the corresponding indices to
    // be oob_tensor values
    TF_ASSIGN_OR_RETURN(
        HloInstruction * oob_check_mask,
        CheckValidIndices(scatter->parent(), scatter_indices,
                          scatter_operands[0]->shape().dimensions(),
                          actual_update_window_dims));

    scatter_indices = parent->AddInstruction(HloInstruction::CreateTernary(
        scatter_indices->shape(), HloOpcode::kSelect, oob_check_mask,
        scatter_indices, out_of_bound_tensor));
    scatter_indices =
        ExpandIndices(scatter->parent(), scatter_indices, index_offsets);

    // Expand the updates
    const int64_t num_elements =
        ShapeUtil::ElementsIn(scatter_updates[0]->shape());
    for (int i = 0; i < scatter_updates.size(); i++) {
      scatter_updates[i] = parent->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(scatter_updates[i]->shape().element_type(),
                               {num_elements}),
          scatter_updates[i]));
    }

    // Create a new dimension numbers for the new scatter operation
    new_dim_numbers.clear_update_window_dims();
    new_dim_numbers.set_index_vector_dim(1);
    // Mitigate the missed dimensions
    for (int i = 0; i < operand_shape.dimensions_size() -
                            dim_numbers.input_batching_dims_size();
         i++) {
      new_dim_numbers.add_inserted_window_dims(i);
    }
    // Set the scatter_dims_to_operand_dims
    // copy from the original scatter_dims_to_operand_dims
    for (int i = 0; i < dim_numbers.scatter_dims_to_operand_dims_size(); i++) {
      new_dim_numbers.add_scatter_dims_to_operand_dims(
          dim_numbers.scatter_dims_to_operand_dims(i));
    }
  } else {
    new_dim_numbers = dim_numbers;
  }

  // Sort the scatter indices and updates together based on the scatter indices.
  int64_t num_indices = ShapeUtil::ElementsIn(scatter_updates[0]->shape());
  std::vector<HloInstruction*> sorted_tensors = SortIndicesAndUpdates(
      scatter_indices, scatter_updates, num_indices, scatter, parent,
      scatter_operands[0]->shape().dimensions(), has_scalar_indices);
  HloInstruction* sorted_scalar_indices = sorted_tensors[0];
  std::vector<HloInstruction*> sorted_updates(
      sorted_tensors.begin() + 1,
      sorted_tensors.begin() + 1 + scatter_updates.size());
  HloInstruction* sorted_indices = sorted_scalar_indices;
  if (!has_scalar_indices) {
    sorted_indices = sorted_tensors[sorted_tensors.size() - 1];
  }

  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> prefix_scan_updates,
                      ComputePrefixScan(sorted_updates, sorted_scalar_indices,
                                        scatter, parent));

  HloInstruction* last_occurrence_indices =
      FindLastOccurrenceIndices(sorted_indices, sorted_scalar_indices, scatter,
                                parent, num_indices, out_of_bound_tensor);

  // Finally, recreate the scatter instruction with unique indices
  auto* new_scatter = parent->AddInstruction(HloInstruction::CreateScatter(
      scatter->shape(), scatter_operands, last_occurrence_indices,
      prefix_scan_updates, scatter->to_apply(), new_dim_numbers,
      true /*indices_are_sorted*/, true /*unique_indices*/));
  return new_scatter;
}

namespace {
void RecursivelyGetInputParamNumbers(
    const HloInstruction* instruction, std::vector<int64_t>& param_numbers,
    absl::flat_hash_set<const HloInstruction*>& visited) {
  if (!visited.emplace(instruction).second) {
    return;
  }

  if (instruction->opcode() == HloOpcode::kParameter) {
    param_numbers.push_back(instruction->parameter_number());
    return;
  }
  for (HloInstruction* operand : instruction->operands()) {
    RecursivelyGetInputParamNumbers(operand, param_numbers, visited);
  }
}

// Check if every output of the scatter computation only depends on the
// corresponding operand and updates
bool CheckOutputDependency(HloComputation* to_apply, int operand_size) {
  HloInstruction* root = to_apply->root_instruction();
  if (!root->shape().IsTuple()) {
    return true;
  }
  CHECK_EQ(operand_size, root->operand_count());

  // traverse the tuple output of the computation
  for (int i = 0; i < operand_size; ++i) {
    const HloInstruction* output = root->operand(i);
    std::vector<int64_t> param_numbers;
    absl::flat_hash_set<const HloInstruction*> visited;
    RecursivelyGetInputParamNumbers(output, param_numbers, visited);
    // The input dependencies can be at most 2
    if (param_numbers.size() > 2) {
      return false;
    }
    for (int64_t param_number : param_numbers) {
      if (param_number != i && param_number != operand_size + i) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool ScatterDeterminismExpander::InstructionMatchesPattern(
    HloInstruction* inst) {
  auto* scatter = DynCast<HloScatterInstruction>(inst);
  return (scatter != nullptr) && !IsScatterDeterministic(scatter) &&
         CheckOutputDependency(scatter->to_apply(),
                               scatter->scatter_operands().size());
}

}  // namespace xla
