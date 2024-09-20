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

#include <memory>
#include <utility>

#include "xla/literal.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class ScatterDeterminismExpanderTest : public HloTestBase {};

TEST_F(ScatterDeterminismExpanderTest,
       DoNotEliminateScatterWithAssociativeCombiner) {
  const char* const kModuleStr = R"(
    HloModule scatter_determinism_expander

    scatter_computation {
      arg1.173 = s32[] parameter(1)
      arg0.172 = s32[] parameter(0)
      ROOT add.48 = s32[] add(arg0.172, arg1.173)
    }

    ENTRY fused_computation {
      bitcast.2335 = s32[1,4096] parameter(0)
      pad.96 = s32[4096,2] parameter(1)
     bitcast.2748 = s32[4096,1,1] parameter(2)
      ROOT scatter.48 = s32[1,4096] scatter(bitcast.2335, pad.96, bitcast.2748),
        update_window_dims={1,2}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0,1}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterDeterminismExpander scatter_determinism_expander;
  TF_ASSERT_OK_AND_ASSIGN(
      bool result, RunHloPass(&scatter_determinism_expander, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(ScatterDeterminismExpanderTest,
       EliminateScatterWithNonAssociativeCombiner) {
  const char* const kModuleStr = R"(
    HloModule scatter_determinism_expander

    scatter_computation {
      arg1.173 = f32[] parameter(1)
      arg0.172 = f32[] parameter(0)
      ROOT add.48 = f32[] add(arg0.172, arg1.173)
    }

    ENTRY fused_computation {
      bitcast.2335 = f32[4096] parameter(0)
      pad.96 = s32[4096,1] parameter(1)
     bitcast.2748 = f32[4096] parameter(2)
      ROOT scatter.48 = f32[4096] scatter(bitcast.2335, pad.96, bitcast.2748),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterDeterminismExpander scatter_determinism_expander;
  TF_ASSERT_OK_AND_ASSIGN(
      bool result, RunHloPass(&scatter_determinism_expander, module.get()));
  EXPECT_TRUE(result);
}

TEST_F(ScatterDeterminismExpanderTest,
       DoNotEliminateScatterWithAssociativeFp32Combiner) {
  const char* const kModuleStr = R"(
    HloModule scatter_determinism_expander

    scatter_computation {
      arg1.173 = f32[] parameter(1)
      arg0.172 = f32[] parameter(0)
      ROOT max.48 = f32[] maximum(arg0.172, arg1.173)
    }

    ENTRY fused_computation {
      bitcast.2335 = f32[1,4096] parameter(0)
      pad.96 = s32[4096,2] parameter(1)
     bitcast.2748 = f32[4096,1,1] parameter(2)
      ROOT scatter.48 = f32[1,4096] scatter(bitcast.2335, pad.96, bitcast.2748),
        update_window_dims={1,2}, inserted_window_dims={},
        scatter_dims_to_operand_dims={0,1}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterDeterminismExpander scatter_determinism_expander;
  TF_ASSERT_OK_AND_ASSIGN(
      bool result, RunHloPass(&scatter_determinism_expander, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(ScatterDeterminismExpanderTest, ScatterAddCorrectnessTest) {
  const char* const kModuleStr = R"(
    HloModule scatter_determinism_expander

    scatter_computation {
      arg1.173 = f32[] parameter(1)
      arg0.172 = f32[] parameter(0)
      ROOT add.48 = f32[] add(arg0.172, arg1.173)
    }

    ENTRY scatter_add_computation {
      operand = f32[4] constant({0, 0, 0, 0})
      indices = s32[7,1] constant({{0}, {1}, {2}, {3}, {1}, {1}, {2}})
      updates = f32[7] constant({2, 1, 5, 3, 8, 7, 9})
      ROOT scatter.48 = f32[4] scatter(operand, indices, updates),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterDeterminismExpander scatter_determinism_expander;
  TF_ASSERT_OK_AND_ASSIGN(
      bool result, RunHloPass(&scatter_determinism_expander, module.get()));

  EXPECT_TRUE(result);

  std::vector<float> expected_result = {2.0, 16.0, 14.0, 3.0};

  Literal result_literal = ExecuteAndTransfer(std::move(module), {});

  auto result_data = result_literal.data<float>();
  std::vector<float> actual_result(result_data.begin(), result_data.end());

  EXPECT_EQ(actual_result, expected_result);
}

TEST_F(ScatterDeterminismExpanderTest, ScatterAddOutOfBoundCorrectnessTest) {
  const char* const kModuleStr = R"(
    HloModule scatter_determinism_expander

    scatter_computation {
      arg1.173 = f32[] parameter(1)
      arg0.172 = f32[] parameter(0)
      ROOT add.48 = f32[] add(arg0.172, arg1.173)
    }

    ENTRY scatter_add_computation {
      operand = f32[4] constant({0, 0, 0, 0})
      indices = s32[7,1] constant({{0}, {1}, {5}, {4}, {1}, {1}, {2}})
      updates = f32[7] constant({2, 1, 5, 3, 8, 7, 9})
      ROOT scatter.48 = f32[4] scatter(operand, indices, updates),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterDeterminismExpander scatter_determinism_expander;
  TF_ASSERT_OK_AND_ASSIGN(
      bool result, RunHloPass(&scatter_determinism_expander, module.get()));

  EXPECT_TRUE(result);

  std::vector<float> expected_result = {2.0, 16.0, 9.0, 0.0};

  Literal result_literal = ExecuteAndTransfer(std::move(module), {});

  auto result_data = result_literal.data<float>();
  std::vector<float> actual_result(result_data.begin(), result_data.end());

  EXPECT_EQ(actual_result, expected_result);
}

TEST_F(ScatterDeterminismExpanderTest, ScatterAddReproducibilityTest) {
  const char* const kModuleStr = R"(
    HloModule scatter_determinism_expander

    scatter_computation {
      arg1.173 = f32[] parameter(1)
      arg0.172 = f32[] parameter(0)
      ROOT add.48 = f32[] add(arg0.172, arg1.173)
    }

    ENTRY scatter_add_computation {
      operand = f32[3] constant({0, 0, 0})
      indices = s32[100,1] constant({{0}, {3}, {0}, {1}, {0}, {3}, {1}, {2}, {1}, {2}, {2}, {2}, {0}, {2}, {1}, {0}, {1}, {1}, {2}, {0}, {2}, {1}, {2}, {1}, {2}, {2}, {3}, {2}, {2}, {0}, {3}, {0}, {3}, {2}, {0}, {3}, {3}, {3}, {3}, {3}, {2}, {3}, {3}, {0}, {0}, {3}, {3}, {3}, {2}, {3}, {2}, {3}, {0}, {0}, {2}, {0}, {1}, {3}, {1}, {3}, {2}, {2}, {2}, {1}, {0}, {3}, {1}, {1}, {1}, {1}, {1}, {2}, {2}, {3}, {0}, {2}, {2}, {0}, {2}, {1}, {0}, {2}, {2}, {2}, {0}, {2}, {0}, {1}, {3}, {0}, {2}, {3}, {3}, {2}, {0}, {3}, {3}, {2}, {3}, {2}})
      updates = f32[100] constant({0.02379167, 0.8527204, 0.8132185, 0.5140263, 0.17172801, 0.8026866, 0.5124631, 0.34838438, 0.50526905, 0.3370521, 0.10868239, 0.10520637, 0.83827364, 0.78986526, 0.34059846, 0.8349273, 0.24575627, 0.21387374, 0.02423227, 0.5617423, 0.28066766, 0.94366455, 0.61214995, 0.7383388, 0.52419806, 0.65466726, 0.41012764, 0.24028647, 0.74443066, 0.03544927, 0.851014, 0.02434528, 0.47239733, 0.72706807, 0.35055435, 0.6274171, 0.61077535, 0.06525731, 0.8091929, 0.21307838, 0.6465323, 0.3245015, 0.5538883, 0.8849807, 0.9591211, 0.83856845, 0.48919427, 0.11810577, 0.16933143, 0.83657074, 0.587505, 0.6867087, 0.95522237, 0.5797727, 0.28024232, 0.34749162, 0.5199702, 0.9811766, 0.5645981, 0.2446456, 0.68722725, 0.9616587, 0.480047, 0.88953114, 0.7083205, 0.948612, 0.67764974, 0.44131804, 0.36789334, 0.95148766, 0.30909216, 0.70908046, 0.8749926, 0.60973287, 0.60751855, 0.22647333, 0.5363518, 0.96195626, 0.08158326, 0.5266887, 0.85922587, 0.648262, 0.4657668, 0.31623375, 0.43507564, 0.48351157, 0.41285944, 0.73501325, 0.15267539, 0.67055714, 0.08459568, 0.04527426, 0.21078384, 0.4654404, 0.7363906, 0.23245859, 0.22119188, 0.99092937, 0.878675, 0.4102913})
      ROOT scatter.48 = f32[3] scatter(operand, indices, updates),
        update_window_dims={}, inserted_window_dims={0},
        scatter_dims_to_operand_dims={0}, index_vector_dim=1,
        to_apply=scatter_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ScatterDeterminismExpander scatter_determinism_expander;
  TF_ASSERT_OK_AND_ASSIGN(
      bool result, RunHloPass(&scatter_determinism_expander, module.get()));

  EXPECT_TRUE(result);

  auto cloned_module = module->Clone();
  Literal first_result_literal =
      ExecuteAndTransfer(std::move(cloned_module), {});
  auto first_result_span = first_result_literal.data<float>();
  std::vector<float> first_result(first_result_span.begin(),
                                  first_result_span.end());

  const int num_trials = 20;
  std::vector<std::vector<float>> results;

  for (int i = 0; i < num_trials; ++i) {
    auto cloned_module = module->Clone();

    Literal result_literal = ExecuteAndTransfer(std::move(cloned_module), {});

    auto result_data = result_literal.data<float>();
    std::vector<float> actual_result(result_data.begin(), result_data.end());

    EXPECT_EQ(actual_result, first_result)
        << "Results are not reproducible across trials!";
  }
}

}  // namespace
}  // namespace xla
