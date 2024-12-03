# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.bindings import GptJsonConfig, KVCacheType
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
import tensorrt_llm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
# # class QWEN_TRTLLM:

#     def __init__(self, engine_dir):
#         self.session = self.get_session(engine_dir)
#         # config = read_config('encoder', engine_dir)

#     def get_session(self, engine_dir):
#         serialize_path = engine_dir / 'rank0.engine'
#         with open(serialize_path, 'rb') as f:
#             session = Session.from_serialized_engine(f.read())
#         return session

#     def run(self):
#         inputs = OrderedDict()
#         # fake input_ids, [B, T] B=2, T=4
#         inputs['input_ids'] = torch.ones(8, dtype=torch.int32).cuda()
#         inputs['input_ids'] = torch.ones(2, 4, dtype=torch.int32).cuda()
#         inputs['position_ids'] = torch.arange(4, dtype=torch.int32).cuda().expand(2, -1)
#         inputs['last_token_ids'] = torch.ones(2, dtype=torch.int32).cuda()
#         print(inputs['input_ids'].shape)
#         print(inputs['position_ids'].shape)
        
#         input_list = [
#             TensorInfo('input_ids', str_dtype_to_trt('int32'),
#                        inputs['input_ids'].shape),
#             TensorInfo('position_ids', str_dtype_to_trt('int32'),
#                        inputs['position_ids'].shape),
#             TensorInfo('last_token_ids', str_dtype_to_trt('int32'), inputs['last_token_ids'].shape),

#         ]

#         output_info = (self.session).infer_shapes(input_list)
#         print(output_info)
#         logger.debug(f'output info {output_info}')

#         outputs = {
#             t.name: torch.empty(tuple((2,1,151936)),
#                                 dtype=trt_dtype_to_torch(t.dtype),
#                                 device='cuda')
#             for t in output_info if t.name == 'logits'
#         }

#         stream = torch.cuda.current_stream()
#         ok = self.session.run(inputs=inputs,
#                               outputs=outputs,
#                               stream=stream.cuda_stream)
#         assert ok, 'Engine execution failed'
#         stream.synchronize()
#         return outputs

#     def generate(self):
#         if 'qwen2_vl' in self.model_type:
#             input_ids, input_lengths, ptuning_args, visual_features, mrope_args = self.preprocess(
#                 warmup, pre_prompt, post_prompt, image, other_vision_inputs)
#             mrope_params = MropeParams(
#                 mrope_rotary_sin_cos=mrope_args[0],
#                 mrope_position_deltas=mrope_args[1],
#             )
#         else:
#             input_ids, input_lengths, ptuning_args, visual_features = self.preprocess(
#                 warmup, pre_prompt, post_prompt, image, other_vision_inputs)
#         # use prompt tuning to pass multimodal features
#         # model.generate() expects the following params (see layers/embedding.py):
#         # args[0]: prompt embedding table, [batch_size, multimodal_len, hidden_size], later flattened to [batch_size * multimodal_len, hidden_size]
#         # args[1]: prompt task ids, [batch_size]. in multimodal case, arange(batch_size), i.e. in VILA batching mode 2, each image is treated separately in the batch instead of concated together (although the prompt embedding table has to be concated)
#         # args[2]: prompt task vocab size, [1]. assuming all table has the same length, which in multimodal case equals to multimodal_len

#         end_id = self.tokenizer.eos_token_id
#         if 'opt' in self.model_type and 'blip2' in self.model_type:
#             # For BLIP2-OPT, model outputs a "\n" at the end.
#             # we avoid it by using newline as the end token
#             end_id = self.tokenizer.encode("\n",
#                                             add_special_tokens=False)[0]

#         if self.model_type == 'cogvlm':
#             input_position_ids = self.prepare_position_ids_for_cogvlm(
#                 input_ids)

#         batch_size = len(input_ids)
#         prompt_tasks = ",".join(
#             np.arange(batch_size, dtype=np.int32).astype(str))
#         prompt_table = torch.stack([ptuning_args[0]])
#         prompt_table = prompt_table.view(batch_size, -1,
#                                             prompt_table.shape[-1])

#         output_ids = self.model.generate(
#             input_ids,
#             input_position_ids=input_position_ids
#             if self.model_type == 'cogvlm' else None,
#             mrope_params=mrope_params
#             if self.model_type == 'qwen2_vl' else None,
#             sampling_config=None,
#             prompt_table=prompt_table,
#             prompt_tasks=prompt_tasks,
#             max_new_tokens=max_new_tokens,
#             end_id=end_id,
#             pad_id=self.tokenizer.pad_token_id
#             if self.tokenizer.pad_token_id is not None else
#             self.tokenizer.all_special_ids[0],
#             top_k=self.args.top_k,
#             top_p=self.args.top_p,
#             temperature=self.args.temperature,
#             repetition_penalty=self.args.repetition_penalty,
#             num_beams=self.args.num_beams,
#             output_sequence_lengths=False,
#             return_dict=False)


#     def preprocess(self, warmup, pre_prompt, post_prompt, image,
#                    other_vision_inputs):
#         if self.model_type == 'kosmos-2':
#             input_ids = image['input_ids'].clone()
#             image_mask = image["image_embeds_position_mask"]
#             image = image['pixel_values']
#             input_ids += image_mask * (self.model_config.vocab_size - 4)
#             input_ids = input_ids.expand(self.args.batch_size,
#                                          *input_ids.shape[1:])
#             length = input_ids.shape[1]
        



#         visual_features, visual_atts = self.get_visual_features(
#             torch.stack(image['image_patches'], dim=0) if
#             self.model_type == 'fuyu' else image, other_vision_inputs)

#         if self.model_type == 'fuyu':
#             visual_features = visual_features.squeeze()
#             input_ids = image['input_ids'].to(torch.int32)
#             image_patches_indices = image['image_patches_indices'].to(
#                 torch.int32)

#             input_ids = input_ids.expand(self.args.batch_size,
#                                          *input_ids.shape[1:])
#             image_patches_indices = image_patches_indices.expand(
#                 self.args.batch_size, *image_patches_indices.shape[1:])

#             input_ids = self.ptuning_setup_fuyu(input_ids,
#                                                 image_patches_indices)
#             input_ids = torch.stack(input_ids, dim=0).to('cpu')
#             length = input_ids.shape[1]


#         elif self.model_type == 'phi-3-vision':
#             image_sizes = input["image_sizes"]
#             visual_features = self.vision_model.hd_feature_transform(
#                 visual_features, image_sizes)
#             input_ids = input["input_ids"].clone()
#             input_ids = input_ids.expand(self.args.batch_size,
#                                          *input_ids.shape[1:])
#             num_img_tokens = [visual_features.shape[0]]
#             input_ids = self.ptuning_setup_phi3(visual_features, input_ids,
#                                                 num_img_tokens)
#             visual_features = visual_features.unsqueeze(0).repeat(
#                 self.args.batch_size, 1, 1)
#             length = input_ids.shape[1]
#         elif self.model_type == 'llava_next':
#             visual_features = LlavaNextUtils.rearrange_image_features(
#                 visual_features, self.image_newlines["image_newline"],
#                 image_size)
#             input_ids = self.ptuning_setup_llava_next(visual_features,
#                                                       pre_prompt, post_prompt)
#             length = input_ids.shape[1]
#         else:
#             pre_input_ids = self.tokenizer(pre_prompt,
#                                            return_tensors="pt",
#                                            padding=True).input_ids
#             if post_prompt[0] is not None:
#                 post_input_ids = self.tokenizer(post_prompt,
#                                                 return_tensors="pt",
#                                                 padding=True).input_ids
#                 if self.model_type == 'video-neva':
#                     length = pre_input_ids.shape[1] + post_input_ids.shape[
#                         1] + visual_atts.shape[2] * visual_atts.shape[1]
#                 elif self.model_type == 'internvl':
#                     length = pre_input_ids.shape[1] + post_input_ids.shape[
#                         1] + visual_atts.shape[0] * visual_atts.shape[1]
#                 else:
#                     length = pre_input_ids.shape[1] + post_input_ids.shape[
#                         1] + visual_atts.shape[1]
#             else:
#                 post_input_ids = None
#                 length = pre_input_ids.shape[1] + visual_atts.shape[1]

#         input_lengths = torch.IntTensor([length] * self.args.batch_size).to(
#             torch.int32)

#         if self.model_type in [
#                 'fuyu', 'kosmos-2', 'phi-3-vision', 'llava_next'
#         ]:
#             return input_ids, input_lengths, [visual_features], visual_features

#         input_ids, ptuning_args = self.setup_fake_prompts(
#             visual_features, pre_input_ids, post_input_ids, input_lengths)

#         return input_ids, input_lengths, ptuning_args, visual_features



#     def _gen_tensorrt_llm_runtime(self,
#                                   log_level,
#                                   dtype,
#                                   world_size,
#                                   rank,
#                                   self.llm_config,
#                                   hf_llama,
#                                   model_name,
#                                   use_plugin,
#                                   batch_size,
#                                   beam_width,
#                                   input_len,
#                                   output_len,
#                                   use_refit,
#                                   fast_building=False,
#                                   context_fmha_flag=ContextFMHAType.disabled,
#                                   enable_remove_input_padding=False,
#                                   **opt_flags):
#         tensorrt_llm.logger.set_level(log_level)
#         mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)
#         engine_buffer = self._gen_tensorrt_llm_engine(
#             dtype, rank, world_size, self.llm_config, hf_llama, model_name,
#             use_plugin, batch_size, beam_width, input_len, output_len,
#             use_refit, fast_building, context_fmha_flag,
#             enable_remove_input_padding, **opt_flags)
#         runtime = tensorrt_llm.runtime.generation._Runtime(
#             engine_buffer, mapping)
#         return runtime, engine_buffer


#     def test_llama(self, use_refit, fast_building, context_fmha_flag,
#                    enable_remove_input_padding, dtype, num_kv_heads, hidden_act,
#                    opt_flags):

#         hf_llama = LlamaForCausalLM(self.llm_config).cuda().eval()
#         runtime, _ = self._gen_tensorrt_llm_runtime(
#             log_level, dtype, world_size, rank, self.llm_config, hf_llama, model,
#             use_plugin, batch_size, beam_width, input_len, output_len,
#             use_refit, fast_building, context_fmha_flag,
#             enable_remove_input_padding, **opt_flags)
#         key_value_cache_buffers = []
#         head_size = self.llm_config.hidden_size // self.llm_config.num_attention_heads
#         for i in range(self.llm_config.num_hidden_layers):
#             key_value_cache_buffers.append(
#                 torch.zeros((
#                     batch_size,
#                     2,
#                     self.llm_config.num_key_value_heads,
#                     max_seq_len,
#                     head_size,
#                 ),
#                             dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
#                             device='cuda'))

#         # compare context
#         step = 0
#         ctx_ids = torch.randint(100, (batch_size, input_len)).int().cuda()
#         ctx_context_lengths = input_len * torch.ones(
#             (batch_size), dtype=torch.int32, device='cuda')
#         ctx_position_ids = torch.tensor(range(input_len),
#                                         dtype=torch.int32).reshape([
#                                             1, input_len
#                                         ]).expand([batch_size,
#                                                    input_len]).cuda()
#         ctx_last_token_ids = ctx_context_lengths.clone()
#         ctx_host_request_types = torch.tensor([0] * batch_size,
#                                               dtype=torch.int32)

#         # We need sequence_lengths start as context_lengths for step 0,
#         # and it will be added one after each step.
#         sequence_length_buffer = ctx_context_lengths.detach().clone()

#         # with torch.no_grad():
#         #     hf_outputs = hf_llama.forward(ctx_ids)
#         # torch.cuda.synchronize()
#         # ref = hf_outputs.logits[:, -1, :]

#         # if enable_remove_input_padding:
#         #     ctx_ids = ctx_ids.view([batch_size * input_len])
#         #     ctx_position_ids = ctx_position_ids.view([batch_size * input_len])
#         #     ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()

#         cache_indirections = [
#             torch.full((
#                 batch_size,
#                 beam_width,
#                 max_seq_len,
#             ),
#                        0,
#                        dtype=torch.int32,
#                        device='cuda'),
#             torch.full((
#                 batch_size,
#                 beam_width,
#                 max_seq_len,
#             ),
#                        0,
#                        dtype=torch.int32,
#                        device='cuda')
#         ]  # ping-pong buffers

#         perf_knob_tensor_size = 16
#         # runtime_perf_knobs is not used in context phase
#         context_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
#                                                   dtype=torch.int64)
#         # if context_fmha_flag == ContextFMHAType.enabled_with_fp32_acc:
#         #     context_runtime_perf_knobs[1] = 1  # enable_context_fmha_fp32_acc

#         host_context_progress = torch.tensor([0], dtype=torch.int64)

#         ctx_buffer = {
#             'input_ids': ctx_ids,
#             'context_lengths': ctx_context_lengths,
#             'position_ids': ctx_position_ids,
#             'last_token_ids': ctx_last_token_ids,
#             'cache_indirection': cache_indirections[0],
#             'host_request_types': ctx_host_request_types,
#             'host_runtime_perf_knobs': context_runtime_perf_knobs,
#             'host_context_progress': host_context_progress,
#         }
#         # if enable_remove_input_padding:
#         #     ctx_buffer['host_context_lengths'] = ctx_context_lengths.cpu()

#         ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

#         kv_shape = (batch_size, 2, self.llm_config.num_key_value_heads,
#                     max_seq_len, head_size)
#         ctx_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
#             [max_seq_len] * self.llm_config.num_hidden_layers, dtype=torch.int32)
#         ctx_shape[f'host_max_attention_window_sizes'] = (
#             self.llm_config.num_hidden_layers, )
#         for i in range(self.llm_config.num_hidden_layers):
#             ctx_shape[f'past_key_value_{i}'] = kv_shape
#             ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
#             ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
#         ctx_buffer['sequence_length'] = sequence_length_buffer
#         ctx_shape['sequence_length'] = ctx_buffer['sequence_length'].shape
#         ctx_shape['host_past_key_value_lengths'] = (batch_size, )
#         ctx_buffer['host_past_key_value_lengths'] = torch.tensor(
#             [0] * batch_size, dtype=torch.int32)
#         ctx_shape['host_sink_token_length'] = (1, )
#         ctx_buffer['host_sink_token_length'] = torch.tensor([0],
#                                                             dtype=torch.int32)

#         context = runtime.ctx_context
#         runtime._set_shape(context, ctx_shape)
#         runtime._set_buffer(context, ctx_buffer)
#         runtime._run(context)
#         torch.cuda.synchronize()
#         res = ctx_buffer['logits']

#         # np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
#         #                            res.to(torch.float32).cpu().numpy(),
#         #                            atol=0.12)

        # compare generation
        # step = 1
        # step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        # gen_context_lengths = ctx_context_lengths.clone()
        # gen_position_ids = torch.ones_like(step1_id).int().cuda() * input_len
        # gen_last_token_ids = torch.zeros_like(gen_context_lengths).int().cuda()
        # gen_host_request_types = torch.tensor([1] * batch_size,
        #                                       dtype=torch.int32)

        # # with torch.no_grad():
        # #     hf_outputs = hf_llama.forward(
        # #         step1_id,
        # #         past_key_values=hf_outputs.past_key_values,
        # #         use_cache=True)
        # # torch.cuda.synchronize()
        # # ref = hf_outputs.logits[:, -1, :]

        # # if enable_remove_input_padding:
        # #     step1_id = step1_id.view([batch_size])
        # #     gen_position_ids = gen_position_ids.view([batch_size])
        # #     gen_last_token_ids = torch.ones_like(
        # #         gen_context_lengths).int().cuda()
        # #     gen_last_token_ids = torch.cumsum(gen_last_token_ids, dim=0).int()
        # gen_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
        #                                       dtype=torch.int64)
        # # if context_fmha_flag == ContextFMHAType.enabled_with_fp32_acc:
        # #     gen_runtime_perf_knobs[1] = 1  # enable_context_fmha_fp32_acc

        # step1_buffer = {
        #     'input_ids': step1_id,
        #     'context_lengths': gen_context_lengths,
        #     'position_ids': gen_position_ids,
        #     'last_token_ids': gen_last_token_ids,
        #     'host_request_types': gen_host_request_types,
        #     'cache_indirection': cache_indirections[1],
        #     'host_runtime_perf_knobs': gen_runtime_perf_knobs,
        #     'host_context_progress': host_context_progress,
        # }
        # # if enable_remove_input_padding:
        # #     step1_buffer['host_context_lengths'] = gen_context_lengths.cpu()

        # step1_shape = {k: v.shape for k, v in step1_buffer.items()}

        # step1_shape[f'host_max_attention_window_sizes'] = (
        #     self.llm_config.num_hidden_layers, )
        # for i in range(self.llm_config.num_hidden_layers):
        #     step1_shape[f'past_key_value_{i}'] = kv_shape
        # step1_shape['sequence_length'] = (batch_size, )
        # step1_shape['host_past_key_value_lengths'] = (batch_size, )
        # step1_shape['host_sink_token_length'] = (1, )
        # step1_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
        #     [max_seq_len] * self.llm_config.num_hidden_layers, dtype=torch.int32)
        # for i in range(self.llm_config.num_hidden_layers):
        #     step1_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
        #     step1_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        # step1_buffer[
        #     'host_past_key_value_lengths'] = sequence_length_buffer.cpu()
        # sequence_length_buffer = torch.add(sequence_length_buffer, step)
        # step1_buffer['sequence_length'] = sequence_length_buffer
        # step1_buffer['host_sink_token_length'] = torch.tensor([0],
        #                                                       dtype=torch.int32)

        # context = runtime.context_1
        # runtime._set_shape(context, step1_shape)
        # runtime._set_buffer(context, step1_buffer)
        # runtime._run(context)
        # torch.cuda.synchronize()
        # res = step1_buffer['logits']

        # np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
        #                            res.to(torch.float32).cpu().numpy(),
        #                            atol=0.12)
class QWEN_TRTLLM:
    def __init__(self, engine_dir, llm_config):
        serialize_path = engine_dir / 'rank0.engine'
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()
        rank = tensorrt_llm.mpi_rank()
        world_size = 1
        mapping = tensorrt_llm.Mapping(world_size, rank)

        self.runtime = tensorrt_llm.runtime.generation._Runtime(
            engine_buffer, mapping)
        self.llm_config = llm_config


    def run(self, ctx_ids, batch_size=1, input_len=5, dtype='float32', max_seq_len=512, beam_width=1):
        key_value_cache_buffers = []
        head_size = self.llm_config.hidden_size // self.llm_config.num_attention_heads
        for i in range(self.llm_config.num_hidden_layers):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    2,
                    self.llm_config.num_key_value_heads,
                    max_seq_len,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))
        
        ctx_context_lengths = input_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_position_ids = torch.tensor(range(input_len),
                                        dtype=torch.int32).reshape([
                                            1, input_len
                                        ]).expand([batch_size,
                                                   input_len]).cuda()
        ctx_last_token_ids = ctx_context_lengths.clone()
        ctx_host_request_types = torch.tensor([0] * batch_size,
                                              dtype=torch.int32)

        # We need sequence_lengths start as context_lengths for step 0,
        # and it will be added one after each step.
        sequence_length_buffer = ctx_context_lengths.detach().clone()

        # if enable_remove_input_padding:
        #     ctx_ids = ctx_ids.view([batch_size * input_len])
        #     ctx_position_ids = ctx_position_ids.view([batch_size * input_len])
        #     ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                max_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                max_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda')
        ]  # ping-pong buffers

        perf_knob_tensor_size = 16
        # runtime_perf_knobs is not used in context phase
        context_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                                  dtype=torch.int64)

        host_context_progress = torch.tensor([0], dtype=torch.int64)

        ctx_buffer = {
            'input_ids': ctx_ids,
            'context_lengths': ctx_context_lengths,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'cache_indirection': cache_indirections[0],
            'host_request_types': ctx_host_request_types,
            'host_runtime_perf_knobs': context_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
        }
        # if enable_remove_input_padding:
        #     ctx_buffer['host_context_lengths'] = ctx_context_lengths.cpu()

        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

        kv_shape = (batch_size, 2, self.llm_config.num_key_value_heads,
                    max_seq_len, head_size)
        ctx_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
            [max_seq_len] * self.llm_config.num_hidden_layers, dtype=torch.int32)
        ctx_shape[f'host_max_attention_window_sizes'] = (
            self.llm_config.num_hidden_layers, )
        for i in range(self.llm_config.num_hidden_layers):
            ctx_shape[f'past_key_value_{i}'] = kv_shape
            ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        ctx_buffer['sequence_length'] = sequence_length_buffer
        ctx_shape['sequence_length'] = ctx_buffer['sequence_length'].shape
        ctx_shape['host_past_key_value_lengths'] = (batch_size, )
        ctx_buffer['host_past_key_value_lengths'] = torch.tensor(
            [0] * batch_size, dtype=torch.int32)
        ctx_shape['host_sink_token_length'] = (1, )
        ctx_buffer['host_sink_token_length'] = torch.tensor([0],
                                                            dtype=torch.int32)

        context = self.runtime.ctx_context
        self.runtime._set_shape(context, ctx_shape)
        self.runtime._set_buffer(context, ctx_buffer)

        self.runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']
        return res




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_dir', type=str, default='trt_engines_qwen2.5-0.5B-Instruct')
    args = parser.parse_args()

    model_name = "/home/scratch.yuekaiz_wwfo_1/Qwen2.5-0.5B-Instruct"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).cuda().eval()   

    batch_size, input_len = 1, 5
    # torch.manual_seed(6)
    # ctx_ids = torch.randint(100, (batch_size, input_len)).int().cuda()
    text_inputs = "阿里巴巴集团是一家以商务电商为主的公司"
    ctx_ids = tokenizer.encode(text_inputs, return_tensors="pt").int().cuda()
    ctx_ids = ctx_ids[:, :input_len]
	# tokens = tokenizer.encode(text_inputs, return_tensors="pt").to(
	# 	device=codec_decoder.get_input_embeddings().weight.device)

    qwen = QWEN_TRTLLM(Path(args.engine_dir), hf_model.config)

    for i in range(10):
        start_time = time.time()
        res = qwen.run(ctx_ids, batch_size, input_len)
        end_time = time.time()

        with torch.no_grad():
            hf_outputs = hf_model.forward(ctx_ids)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]
        print(f"Time cost of trt-llm is {end_time - start_time:.4f}")
        print(f"Time cost of hf-llm is {time.time() - end_time:.4f}")


    np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                res.to(torch.float32).cpu().numpy(),
                                atol=0.12)
    # compute top 10 index of the logits
    print(f"The top 10 index of ref is {torch.topk(ref, 10).indices}")
    print(f"The top 10 index of res is {torch.topk(res, 10).indices}")
    print(ref[:20])
    print(res[:20])
    print("Test passed!")


                