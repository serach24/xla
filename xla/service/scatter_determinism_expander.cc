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
#include <unordered_set>

#include "absl/algorithm/container.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
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
static StatusOr<HloInstruction*> TransposeIndexVectorDimToLast(
    HloInstruction* scatter_indices, int64_t index_vector_dim) {
  const Shape& scatter_indices_shape = scatter_indices->shape();

  if (index_vector_dim >= (scatter_indices_shape.dimensions_size() - 1)) {
    return scatter_indices;
  }

  std::vector<int64_t> permutation;
  permutation.reserve(scatter_indices_shape.dimensions_size());
  for (int64_t i = 0; i < scatter_indices_shape.dimensions_size(); i++) {
    if (i != index_vector_dim) {
      permutation.push_back(i);
    }
  }
  permutation.push_back(index_vector_dim);
  return MakeTransposeHlo(scatter_indices, permutation);
}

// Canonicalizes the scatter_indices tensor in order to keep them uniform while
// performing the scatter operation.
static StatusOr<HloInstruction*> CanonicalizeScatterIndices(
    HloInstruction* scatter_indices, int64_t index_vector_dim) {
  // Transpose the non-index-vector dimensions to the front.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * transposed_scatter_indices,
      TransposeIndexVectorDimToLast(scatter_indices, index_vector_dim));
  if (scatter_indices->shape().rank() - 1 == index_vector_dim &&
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
  }
  // Collapse all but the dimensions (0 or 1) in scatter_indices containing
  // the index vectors.
  auto indices = CollapseFirstNDims(
      transposed_scatter_indices,
      shape.dimensions_size() - index_dims_in_scatter_indices);
  bool has_scalar_indices = scatter_indices->shape().dimensions_size() == 1;
  if (has_scalar_indices) {
    scatter_indices =
        scatter_indices->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(scatter_indices->shape().element_type(),
                                 {scatter_indices->shape().dimensions(0), 1}),
            scatter_indices));
  }
  return scatter_indices;
}

// Permutes the `updates` tensor such that all the scatter dims appear in the
// major dimensions and all the window dimensions appear in the minor
// dimensions.
static StatusOr<HloInstruction*> PermuteScatterAndWindowDims(
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
  for (int64_t window_dim : update_window_dims) {
    permutation.push_back(window_dim);
  }

  return MakeTransposeHlo(updates, permutation);
}

// Expands or contracts the scatter indices in the updates tensor.
static StatusOr<HloInstruction*> AdjustScatterDims(
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

// Create the out-of-bound tensor for the scatter operation.
HloInstruction* CreateOutOfBoundTensor(HloComputation* parent,
                                       HloInstruction* scatter_indices,
                                       const Shape& scatter_shape) {
  // Create a constant with the dimensions of the scatter output shape
  auto* output_dimensions_constant =
      parent->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int64_t>(scatter_shape.dimensions())));

  // Convert the constant to match the element type of scatter_indices
  output_dimensions_constant =
      parent->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::MakeShape(
              scatter_indices->shape().element_type(),
              output_dimensions_constant->shape().dimensions()),
          output_dimensions_constant));

  // Broadcast the converted constant to match the shape of scatter_indices
  auto* out_of_bound_tensor =
      parent->AddInstruction(HloInstruction::CreateBroadcast(
          scatter_indices->shape(), output_dimensions_constant, {1}));

  return out_of_bound_tensor;
}

static StatusOr<HloComputation*> CallAndGetOutput(HloComputation* original,
                                                  int output_index) {
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

static int64_t ScatterCount(const HloScatterInstruction* scatter) {
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

// Computation for sorting the scalar scatter indices and updates together
HloComputation* ScalarSortingComparison(HloModule* module,
                                        const Shape key_shape,
                                        const Shape update_shape,
                                        int64_t num_updates) {
  HloComputation::Builder builder("sorting_computation");
  auto param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, key_shape, "lhs_key"));
  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, key_shape, "rhs_key"));
  const int kExistingParams = 2;
  for (int i = 0; i < num_updates; ++i) {
    builder.AddInstruction(
        HloInstruction::CreateParameter(kExistingParams + i, update_shape,
                                        absl::StrFormat("lhs_update_%d", i)));
    builder.AddInstruction(
        HloInstruction::CreateParameter(kExistingParams + 1 + i, update_shape,
                                        absl::StrFormat("rhs_update_%d", i)));
  }
  builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), param0,
                                    param1, ComparisonDirection::kLt));
  return module->AddEmbeddedComputation(builder.Build());
}

static std::vector<HloInstruction*> SortIndicesAndUpdates(
    HloInstruction* scatter_indices,
    const std::vector<HloInstruction*>& scatter_updates, int64_t num_indices,
    HloScatterInstruction* scatter, HloComputation* parent) {
  const Shape& indices_shape = scatter_indices->shape();
  const Shape& updates_shape = scatter_updates[0]->shape();
  auto updates_dims = updates_shape.dimensions();
  // Since we canonicalized the scatter updates, the first dim will always be
  // the number of updates and the rest will be the shape of each update

  auto scalar_indices = parent->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices}),
      scatter_indices));

  std::vector<int64_t> single_update_dimensions(updates_dims.begin() + 1,
                                                updates_dims.end());

  const Shape update_shape = ShapeUtil::MakeShape(updates_shape.element_type(),
                                                  single_update_dimensions);

  const Shape& scalar_index_shape =
      ShapeUtil::MakeShape(indices_shape.element_type(), {num_indices});

  auto* comparison = ScalarSortingComparison(
      scatter->GetModule(),
      ShapeUtil::MakeShape(indices_shape.element_type(), {}),
      ShapeUtil::MakeShape(updates_shape.element_type(), {}),
      scatter_updates.size());

  std::vector<HloInstruction*> sort_operands = {scalar_indices};
  std::vector<Shape> sort_shapes = {scalar_index_shape};
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

  std::vector<HloInstruction*> sorted_updates(scatter_updates.size());
  for (int i = 0; i < scatter_updates.size(); i++) {
    sorted_updates[i] =
        parent->AddInstruction(HloInstruction::CreateGetTupleElement(
            scatter_updates[i]->shape(), sorting, i + 1));
  }
  std::vector<HloInstruction*> sorted_tensors = {sorted_scalar_indices};
  sorted_tensors.insert(sorted_tensors.end(), sorted_updates.begin(),
                        sorted_updates.end());
  return sorted_tensors;
}

// CreateScanWithIndices performs a prefix scan operation (akin to parallel
// prefix sum) on the updates and indices, to compute the accumulated updates in
// log(n) time.
//
// Algorithm (High-level):
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
    TF_ASSIGN_OR_RETURN(HloComputation * to_apply,
                        CallAndGetOutput(scatter->to_apply(), i));
    TF_ASSIGN_OR_RETURN(prefix_scans[i],
                        CreateScanWithIndices(parent, sorted_updates[i],
                                              sorted_scalar_indices, to_apply));
  }
  return prefix_scans;
}

static HloInstruction* FindLastOccurrenceIndices(
    HloInstruction* scatter_indices, HloInstruction* sorted_scalar_indices,
    HloInstruction* scatter, HloComputation* parent, int64_t num_indices) {
  int64_t indices_len = sorted_scalar_indices->shape().dimensions(0);
  auto sorted_expanded_indices =
      parent->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(sorted_scalar_indices->shape().element_type(),
                               {num_indices, 1}),
          sorted_scalar_indices));

  auto* sorted_indices_preceding_part =
      parent->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(scatter_indices->shape().element_type(),
                               {indices_len - 1}),
          sorted_scalar_indices, {0}, {indices_len - 1}, {1}));
  auto* sorted_indices_following_part =
      parent->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(scatter_indices->shape().element_type(),
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

  auto* out_of_bound_tensor =
      CreateOutOfBoundTensor(parent, scatter_indices, scatter->shape());

  auto* masked_indices = parent->AddInstruction(HloInstruction::CreateTernary(
      sorted_expanded_indices->shape(), HloOpcode::kSelect, indices_mask,
      sorted_expanded_indices, out_of_bound_tensor));
  return masked_indices;
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
  int64_t scatter_indices_count = ScatterCount(scatter);
  if (!IsInt32(scatter_indices_count)) {
    // 2147483647 is the maximum value for a 32-bit signed integer (INT32_MAX).
    return Unimplemented(
        "Scatter operations with more than 2147483647 scatter indices are not "
        "supported. This error occurred for %s.",
        scatter->ToString());
  }

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

  HloComputation* parent = scatter->parent();

  // Sort the scatter indices and updates together based on the scatter indices.
  int64_t num_indices = ShapeUtil::ElementsIn(scatter_updates[0]->shape());
  std::vector<HloInstruction*> sorted_tensors = SortIndicesAndUpdates(
      scatter_indices, scatter_updates, num_indices, scatter, parent);
  HloInstruction* sorted_scalar_indices = sorted_tensors[0];
  std::vector<HloInstruction*> sorted_updates(sorted_tensors.begin() + 1,
                                              sorted_tensors.end());

  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> prefix_scan_updates,
                      ComputePrefixScan(sorted_updates, sorted_scalar_indices,
                                        scatter, parent));

  HloInstruction* last_occurrence_indices = FindLastOccurrenceIndices(
      scatter_indices, sorted_scalar_indices, scatter, parent, num_indices);

  // Finally, recreate the scatter instruction with unique indices
  auto* new_scatter = parent->AddInstruction(HloInstruction::CreateScatter(
      scatter->shape(), scatter_operands, last_occurrence_indices,
      prefix_scan_updates, scatter->to_apply(), dim_numbers,
      true /*indices_are_sorted*/, true /*unique_indices*/));
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

void RecursivelyGetInputDependencies(
    const HloInstruction* instruction,
    std::unordered_set<const HloInstruction*>& dependencies) {
  if (instruction->opcode() == HloOpcode::kParameter) {
    dependencies.emplace(instruction);
    return;
  }
  for (HloInstruction* operand : instruction->operands()) {
    RecursivelyGetInputDependencies(operand, dependencies);
  }
}

// Check if the every output of the computation only depends on the
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
    std::unordered_set<const HloInstruction*> input_dependencies;
    RecursivelyGetInputDependencies(output, input_dependencies);
    // The input dependencies can be at most 2
    if (input_dependencies.size() > 2) {
      return false;
    }
    if (input_dependencies.size() == 2) {
      for (const HloInstruction* input : input_dependencies) {
        int64_t param_number = input->parameter_number();
        if (param_number != i && param_number != operand_size + i) {
          return false;
        }
      }
    }
  }
  return true;
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

  return indices_are_1d && updates_are_1d && !IsDeterministic(scatter) &&
         CheckOutputDependency(scatter->to_apply(),
                               scatter->scatter_operands().size());
}

}  // namespace xla