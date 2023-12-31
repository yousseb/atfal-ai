"""
 Copyright (c) 2018-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np
import logging as log
from openvino.runtime import AsyncInferQueue
from openvino.runtime import Core
from openvino.inference_engine import IECore


class Module:
    def __init__(self, core, model_path, model_type):
        self.core = core
        self.model_type = model_type
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)
        self.model_path = model_path
        self.active_requests = 0
        self.clear()

    def deploy(self, device, max_requests=1):
        self.max_requests = max_requests
        compiled_model = self.core.compile_model(self.model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_queue = AsyncInferQueue(compiled_model, self.max_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(self.model_type, self.model_path, device))

    def completion_callback(self, infer_request, id):
        self.outputs[id] = infer_request.results[self.output_tensor]

    def enqueue(self, input):
        if self.max_requests <= self.active_requests:
            log.warning('Processing request rejected - too many requests')
            return False

        self.infer_queue.start_async(input, self.active_requests)
        self.active_requests += 1
        return True

    async def wait(self):
        if self.active_requests <= 0:
            return
        self.infer_queue.wait_all()
        self.active_requests = 0

    async def get_outputs(self):
        await self.wait()
        return [v for _, v in sorted(self.outputs.items())]

    def clear(self):
        self.outputs = {}

    def infer(self, inputs):
        self.clear()
        self.start_async(*inputs)
        return self.postprocess()


class OutputTransform:
    def __init__(self, input_size, output_resolution):
        self.output_resolution = output_resolution
        if self.output_resolution:
            self.new_resolution = self.compute_resolution(input_size)

    def compute_resolution(self, input_size):
        self.input_size = input_size
        size = self.input_size[::-1]
        self.scale_factor = min(self.output_resolution[0] / size[0],
                                self.output_resolution[1] / size[1])
        return self.scale(size)

    def resize(self, image):
        if not self.output_resolution:
            return image
        curr_size = image.shape[:2]
        if curr_size != self.input_size:
            self.new_resolution = self.compute_resolution(curr_size)
        if self.scale_factor == 1:
            return image
        return cv2.resize(image, self.new_resolution)

    def scale(self, inputs):
        if not self.output_resolution or self.scale_factor == 1:
            return inputs
        return (np.array(inputs) * self.scale_factor).astype(np.int32)


class CoreManager(object):
    _core = None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CoreManager, cls).__new__(cls)
        return cls.instance

    def get_core(self):
        if self._core is None:
            ie = IECore()
            cpu_caps = ie.get_metric(metric_name="OPTIMIZATION_CAPABILITIES", device_name="CPU")
            print(f'OpenVino Available CPU Optimizations: {cpu_caps}')
            self._core = Core()
            self._core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "f32"})

            if 'BF16' in cpu_caps:
                self._core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "bf16"})
                self._core.set_property({'ENFORCE_BF16': 'YES'})
            return self._core
        else:
            return self._core


class CommonUtilsMixin(DownloaderBase):
    _core_manager = CoreManager()

    def get_core(self):
        return self._core_manager.get_core()
