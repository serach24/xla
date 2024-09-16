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

#include "xla/service/scatter_determinism_expander.h"

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
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

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

// Assume the indices are in operand space already
// indices shape: (num_indices, num_dims)
// updates shape: (num_indices,)
HloInstruction* FlattenIndices(HloComputation* parent, HloInstruction* indices,
                               absl::Span<const int64_t> operand_dims) {
  // Step 1: based on the operand_dims, calculate the strides
  Array2D<int64_t> strides(operand_dims.size(), 1);
  int64_t stride = 1;
  for (int i = operand_dims.size() - 1; i >= 0; --i) {
    // strides.push_back(stride);
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
  std::vector<SparsityDescriptor> sparsity;
  absl::Span<HloInstruction* const> sparse_meta;
  auto flattened_indices = parent->AddInstruction(
      HloInstruction::CreateDot(dot_shape, indices, strides_tensor, dim_numbers,
                                precision_config, sparsity, sparse_meta));
  return parent->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(indices->shape().element_type(),
                           {indices->shape().dimensions(0)}),
      flattened_indices));
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
absl::StatusOr<HloInstruction*> CreateScanWithIndices(
    HloComputation* parent, HloInstruction* updates, HloInstruction* indices,
    HloComputation* to_apply) {
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

  for (int64_t d = 0; d < log_n; ++d) {
    offset = 1 << d;
    std::vector<int64_t> end_indices = {n - offset};

    auto shifted_updates_shape =
        ShapeUtil::MakeShape(updates_shape.element_type(), {n - offset});
    auto padding_updates_shape =
        ShapeUtil::MakeShape(updates_shape.element_type(), {offset});

    auto shifted_indices_shape =
        ShapeUtil::MakeShape(indices_shape.element_type(), {n - offset});
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
        ShapeUtil::MakeShape(PRED, {n}), prev_indices, concatenated_indices,
        ComparisonDirection::kEq));
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
// TODO(chenhao) maybe could reshape the indices before the map to avoid reshape
// for indices and mask)
HloInstruction* ExpandIndices(HloComputation* parent, HloInstruction* indices,
                              HloInstruction* index_offsets) {
  // For each index we need to add the index_offset to the base index
  // To do that, we first broadcast the indices and index_offsets to the same
  // shape Then we add the index_offset to the base index and flatten the result
  // Broadcast to be (num_indices, length_of_index_offsets, length_of_indices)
  auto num_indices = indices->shape().dimensions(0);
  auto num_offsets = index_offsets->shape().dimensions(0);
  auto index_length = indices->shape().dimensions(1);
  auto final_shape =
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

// For each (index, update) pair, calculate the new updates
HloInstruction* ExpandUpdates(HloComputation* parent, HloInstruction* updates) {
  const int64_t num_elements = ShapeUtil::ElementsIn(updates->shape());
  auto expanded_updates = parent->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(updates->shape().element_type(), {num_elements}),
      updates));
  return expanded_updates;
}

HloComputation* SortingComparison(HloModule* module, const Shape key_shape,
                                  const Shape update_shape,
                                  int64_t num_updates) {
  HloComputation::Builder builder("sorting_computation");
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, key_shape, "lhs_key"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, key_shape, "rhs_key"));
  builder.AddInstruction(
      HloInstruction::CreateParameter(2, key_shape, "lhs_indices"));
  builder.AddInstruction(
      HloInstruction::CreateParameter(3, key_shape, "rhs_indices"));
  for (int i = 0; i < num_updates; ++i) {
    builder.AddInstruction(HloInstruction::CreateParameter(
        4 + i, update_shape, absl::StrCat("lhs_update_", i)));
    builder.AddInstruction(HloInstruction::CreateParameter(
        4 + 1 + i, update_shape, absl::StrCat("rhs_update_", i)));
  }
  builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0,
                                    param1, ComparisonDirection::kLt));

  return module->AddEmbeddedComputation(builder.Build());
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
  auto* zero_constant = parent->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  auto* zero_broadcasted = parent->AddInstruction(
      HloInstruction::CreateBroadcast(indices->shape(), zero_constant, {}));
  auto* zero_check = parent->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, indices->shape().dimensions()), indices,
      zero_broadcasted, ComparisonDirection::kGe));
  // reduce each row to get a mask
  auto* zero_check_mask = parent->AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(PRED, {indices->shape().dimensions(0)}), zero_check,
      init_reduce_value, {1}, reduce_computation));

  // 2. Check last indices <= [bounds...]
  // Check if the index is OOB w.r.t. the operand dimensions and window sizes.
  // Todo(chenhao): figure out if there is a bug here
  std::vector<int64_t> max_valid_index(operand_dims.size());
  for (int i = 0; i < operand_dims.size(); ++i) {
    max_valid_index[i] = operand_dims[i] - window_sizes[i];
  }

  Literal max_valid_index_literal =
      LiteralUtil::CreateR1<int64_t>(max_valid_index);
  if (max_valid_index_literal.shape().element_type() !=
      indices->shape().element_type()) {
    TF_ASSIGN_OR_RETURN(
        max_valid_index_literal,
        max_valid_index_literal.Convert(indices->shape().element_type()));
  }
  auto max_valid_index_constant = parent->AddInstruction(
      HloInstruction::CreateConstant(std::move(max_valid_index_literal)));
  max_valid_index_constant =
      parent->AddInstruction(HloInstruction::CreateBroadcast(
          indices->shape(), max_valid_index_constant, {1}));
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
  return valid_index_mask;
}

absl::StatusOr<HloInstruction*> ScatterDeterminismExpander::ExpandInstruction(
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

  // Canonicalize the updates, after which the size of its most-major dimension
  // must be same as the while loop trip count.
  std::vector<HloInstruction*> adjusted_canonical_updates;
  adjusted_canonical_updates.reserve(scatter_updates.size());
  for (HloInstruction* update : scatter_updates) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * canonical_update,
        PermuteScatterAndWindowDims(update, dim_numbers.update_window_dims()));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * adjusted_canonical_update,
        AdjustScatterDims(scatter_indices->shape(), canonical_update,
                          dim_numbers.index_vector_dim()));
    CHECK_EQ(scatter_loop_trip_count,
             adjusted_canonical_update->shape().dimensions(0));
    adjusted_canonical_updates.push_back(adjusted_canonical_update);
  }

  scatter_indices = canonical_scatter_indices;
  scatter_updates = adjusted_canonical_updates;
  // expand the indices into operand space
  // TODO(chenhao) handle this?
  // TF_ASSIGN_OR_RETURN(
  //     scatter_indices,
  //     ExpandIndexVectorIntoOperandSpace(scatter_indices, dim_numbers,
  //     scatter_operands[0]->shape().dimensions_size()));
  // scatter_indices = ExpandIndexVectorIntoOperandSpace(scatter_indices,
  // dim_numbers, scatter_operands[0]->shape().dimensions_size());

  bool has_scalar_indices = scatter_indices->shape().dimensions_size() == 1;
  // TODO(chenhao) do some checking
  // CHECK_EQ(scatter_indices->shape().dimensions_size(), 1);
  if (has_scalar_indices) {
    // reshape the indices to be a 2D tensor
    // TODO(chenhao): this makes the implementation uniform, but need to check
    // if performance is different (e.g. sorting indices directly or create a
    // gather)
    scatter_indices =
        scatter->parent()->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(scatter_indices->shape().element_type(),
                                 {scatter_loop_trip_count, 1}),
            scatter_indices));
  }

  auto index_length = scatter_indices->shape().dimensions(1);

  auto* parent = scatter->parent();
  auto num_indices = ShapeUtil::ElementsIn(scatter_updates[0]->shape());
  // HloInstruction* expanded_mask = nullptr;

  // Check if each update is a scalar based on update shape
  bool non_scalar_update = scatter_updates[0]->shape().dimensions_size() > 1;
  // Extract operand dimensions
  const Shape& operand_shape = scatter_operands[0]->shape();
  auto operand_dims = operand_shape.dimensions();
  auto updates_shape = scatter_updates[0]->shape();
  auto updates_dims = scatter_updates[0]->shape().dimensions();
  // Since we canonicalized the scatter updates, the first dim will always be
  // the number of updates and the rest will be the shape of each update
  std::vector<int64_t> one_update_dimensions(updates_dims.begin() + 1,
                                             updates_dims.end());
  const Shape& update_shape =
      ShapeUtil::MakeShape(updates_shape.element_type(), one_update_dimensions);

  ScatterDimensionNumbers new_dim_numbers;

  const Shape& expanded_indices_shape = ShapeUtil::MakeShape(
      scatter_indices->shape().element_type(), {num_indices, index_length});
  auto* output_dimensions_constant =
      parent->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int64_t>(scatter->shape().dimensions())));
  output_dimensions_constant =
      parent->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::MakeShape(
              scatter_indices->shape().element_type(),
              output_dimensions_constant->shape().dimensions()),
          output_dimensions_constant));
  auto* test_oob_tensor =
      parent->AddInstruction(HloInstruction::CreateBroadcast(
          scatter_indices->shape(), output_dimensions_constant, {1}));

  auto* out_of_bound_tensor =
      parent->AddInstruction(HloInstruction::CreateBroadcast(
          expanded_indices_shape, output_dimensions_constant, {1}));

  if (non_scalar_update) {
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

    TF_ASSIGN_OR_RETURN(
        HloInstruction * oob_check_mask,
        CheckValidIndices(scatter->parent(), scatter_indices,
                          scatter_operands[0]->shape().dimensions(),
                          actual_update_window_dims));

    // if any updates are out of bound, we change the corresponding indices to
    // be oob_tensor values
    oob_check_mask = parent->AddInstruction(HloInstruction::CreateBroadcast(
        ShapeUtil::MakeShape(PRED, scatter_indices->shape().dimensions()),
        oob_check_mask, {0}));
    scatter_indices = parent->AddInstruction(HloInstruction::CreateTernary(
        scatter_indices->shape(), HloOpcode::kSelect, oob_check_mask,
        scatter_indices, test_oob_tensor));
    scatter_indices =
        ExpandIndices(scatter->parent(), scatter_indices, index_offsets);

    for (int i = 0; i < scatter_updates.size(); i++) {
      scatter_updates[i] = ExpandUpdates(scatter->parent(), scatter_updates[i]);
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
  // Assume scatter_indices is defined and is an HloInstruction pointer
  const Shape& indices_shape = scatter_indices->shape();

  // Create the shape for a single index tuple
  const Shape& scalar_index_shape =
      ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices});

  auto scalar_indices =
      FlattenIndices(scatter->parent(), scatter_indices, operand_dims);

  // Create [0...num_indices] tensor for permutation in sorting
  auto indices_permutation = parent->AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices}), 0));

  auto* comparison =
      SortingComparison(scatter->GetModule(),
                        ShapeUtil::MakeShape(indices_shape.element_type(), {}),
                        ShapeUtil::MakeShape(updates_shape.element_type(), {}),
                        scatter_updates.size());
  // std::vector<HloInstruction*> sort_operands = {
  //     scalar_indices, indices_permutation, scatter_update};
  std::vector<HloInstruction*> sort_operands = {scalar_indices,
                                                indices_permutation};
  std::vector<Shape> sort_shapes = {scalar_index_shape,
                                    indices_permutation->shape()};
  for (auto update : scatter_updates) {
    sort_operands.push_back(update);
    sort_shapes.push_back(update->shape());
  }

  auto* sorting = parent->AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeTupleShape(sort_shapes), 0, sort_operands, comparison,
      false /*is_stable*/));
  auto* sorted_scalar_indices =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          scalar_indices->shape(), sorting, 0));
  auto* sorted_indices_arg =
      parent->AddInstruction(HloInstruction::CreateGetTupleElement(
          indices_permutation->shape(), sorting, 1));
  sorted_indices_arg = parent->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(sorted_indices_arg->shape().element_type(),
                           {num_indices, 1}),
      sorted_indices_arg));

  std::vector<HloInstruction*> sorted_updates(scatter_updates.size());
  for (int i = 0; i < scatter_updates.size(); i++) {
    sorted_updates[i] =
        parent->AddInstruction(HloInstruction::CreateGetTupleElement(
            scatter_updates[i]->shape(), sorting, i + 2));
  }

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

  // Compute the scan
  std::vector<HloInstruction*> prefix_scan_updates(scatter_updates.size());
  for (int i = 0; i < scatter_updates.size(); i++) {
    TF_ASSIGN_OR_RETURN(HloComputation * to_apply,
                        CallAndGetOutput(scatter->to_apply(), i));
    TF_ASSIGN_OR_RETURN(prefix_scan_updates[i],
                        CreateScanWithIndices(parent, sorted_updates[i],
                                              sorted_scalar_indices, to_apply));
  }

  // Compute which unique indices to write
  int64_t indices_len = sorted_scalar_indices->shape().dimensions(0);
  auto* sorted_indices_preceding_part =
      parent->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(scalar_index_shape.element_type(),
                               {indices_len - 1}),
          sorted_scalar_indices, {0}, {indices_len - 1}, {1}));
  auto* sorted_indices_following_part =
      parent->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(scalar_index_shape.element_type(),
                               {indices_len - 1}),
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
      ShapeUtil::MakeShape(PRED, sorted_expanded_indices->shape().dimensions()),
      indices_mask, {0}));

  auto* masked_indices = parent->AddInstruction(HloInstruction::CreateTernary(
      sorted_expanded_indices->shape(), HloOpcode::kSelect, indices_mask,
      sorted_expanded_indices, out_of_bound_tensor));

  // Finally, recreate the scatter instruction with unique indices
  auto* new_scatter = parent->AddInstruction(HloInstruction::CreateScatter(
      scatter->shape(), scatter_operands, masked_indices, prefix_scan_updates,
      scatter->to_apply(), new_dim_numbers, true /*indices_are_sorted*/,
      true /*unique_indices*/));
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

bool ScatterDeterminismExpander::InstructionMatchesPattern(
    HloInstruction* inst) {
  auto* scatter = DynCast<HloScatterInstruction>(inst);
  // Need to check if updates and indices are scalar, as the current pass does
  // not expand scatter with multi-dimensional updates or indices. This is
  // temporary and will be removed in a future PR soon.
  if (scatter == nullptr) {
    return false;
  }

  const Shape& indices_shape = scatter->scatter_indices()->shape();
  const Shape& updates_shape = scatter->scatter_updates()[0]->shape();

  // Check if indices and updates are effectively 1D.
  bool indices_are_1d =
      (indices_shape.rank() == 1 ||
       (indices_shape.rank() == 2 && indices_shape.dimensions(1) == 1));
  bool updates_are_1d =
      (updates_shape.rank() == 1 ||
       (updates_shape.rank() == 2 && updates_shape.dimensions(1) == 1));

  return indices_are_1d && updates_are_1d && !IsDeterministic(scatter);
}

}  // namespace xla
