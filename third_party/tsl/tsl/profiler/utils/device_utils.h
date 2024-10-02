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

#ifndef TENSORFLOW_TSL_PROFILER_UTILS_DEVICE_UTILS_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_DEVICE_UTILS_H_

#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

enum class DeviceType {
  kUnknown,
  kCpu,
  kTpu,
  kGpu,
};

// Get DeviceType from XPlane.
DeviceType GetDeviceType(const tensorflow::profiler::XPlane& plane);

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_DEVICE_UTILS_H_