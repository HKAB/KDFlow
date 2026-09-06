import os
import queue
import time
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
from sglang.srt.entrypoints.engine import Engine as _SglEngine
from sglang.srt.managers.scheduler import run_scheduler_process as _original_run_scheduler_process

from kdflow.utils.logging_utils import init_logger
from kdflow.backend.sglang.hidden_state_alignment import (
    loss_mask_start_positions,
    select_loss_hidden_states,
)

logger = init_logger(__name__)


os.environ["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "true"

def _patched_run_scheduler_process(*args, **kwargs):
    from kdflow.backend.sglang.monkey_patch import apply_patch
    if not apply_patch():
        raise RuntimeError(
            "KDFlow could not install its SGLang hidden-state transfer patch. "
            "Verify that the runtime uses the supported SGLang 0.5.17 API."
        )
    return _original_run_scheduler_process(*args, **kwargs)


class PatchedEngine(_SglEngine):
    """
    SGLang Engine that applies monkey patch in scheduler subprocesses.
    Motivation: SGLang Engine supports returning hidden states, but the existing implementation use .tolist() to convert hidden states from GPU tensor to Python list, which is very inefficient. This monkey patch replaces the original .tolist() with a more efficient operation .numpy().
    """
    run_scheduler_process_func = staticmethod(_patched_run_scheduler_process)


@dataclass
class EngineConfig:
    """Configuration for SGLang Engine."""
    model_path: str
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    chunked_prefill_size: int = -1
    disable_radix_cache: bool = True
    enable_return_hidden_states: bool = True
    enable_memory_saver: bool = True
    enable_weights_cpu_backup: bool = True
    mem_fraction_static: float = 0.8
    quantization: str = None
    offload_tags: Optional[str] = "all"
    base_gpu_id: int = 0
    # for multi-node tp/pp
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: Optional[str] = None


def _engine_worker(config: EngineConfig, request_queue: Queue, response_queue: Queue, hidden_queue: Queue):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if config.nnodes > 1:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

    engine = None

    try:
        engine = PatchedEngine(
            model_path=config.model_path,
            tp_size=config.tp_size,
            ep_size=config.ep_size,
            pp_size=config.pp_size,
            chunked_prefill_size=config.chunked_prefill_size,
            disable_radix_cache=config.disable_radix_cache,
            enable_return_hidden_states=config.enable_return_hidden_states,
            enable_memory_saver=config.enable_memory_saver,
            enable_weights_cpu_backup=config.enable_weights_cpu_backup,
            quantization=config.quantization,
            mem_fraction_static=config.mem_fraction_static,
            base_gpu_id=config.base_gpu_id,
            nnodes=config.nnodes,
            node_rank=config.node_rank,
            dist_init_addr=config.dist_init_addr,
            disable_custom_all_reduce=True,
        )

        response_queue.put({
            "type": "init_done", 
            "success": True,
        })

        while True:
            request = request_queue.get()
            if request is None:
                break

            req_type = request.get("type")

            try:
                if req_type == "generate":
                    _handle_generate(engine, request, hidden_queue, response_queue)
                elif req_type == "sleep":
                    _handle_sleep(engine, request, config, response_queue)
                elif req_type == "wakeup":
                    _handle_wakeup(engine, request, config, response_queue)
                elif req_type == "update_weights_from_tensor":
                    _handle_update_weights_from_tensor(engine, request, response_queue)
                elif req_type == "flush_cache":
                    engine.flush_cache()
                    response_queue.put({"type": "flush_cache", "success": True})
                else:
                    response_queue.put({"type": req_type, "success": False,
                                        "error": f"Unknown request type: {req_type}"})
            except Exception:
                import traceback
                response_queue.put({"type": req_type, "success": False,
                                    "error": traceback.format_exc()})

    except Exception:
        import traceback
        response_queue.put({"type": "init_done", "success": False,
                            "error": traceback.format_exc()})
    finally:
        if engine:
            try:
                engine.shutdown()
            except Exception:
                pass


def _normalize_tags(tags):
    """Convert tags to the format SGLang expects (None, or list of strings)."""
    if tags is None or tags == "all":
        return None
    if isinstance(tags, str):
        return [tags]
    return tags


def _handle_generate(engine, request, hidden_queue, response_queue):
    """Handle a generate request and stream hidden states via shared memory."""
    kwargs = request["kwargs"]

    generate_kwargs = {
        "input_ids": kwargs["input_ids"],
        "sampling_params": kwargs["sampling_params"],
        "return_hidden_states": kwargs.get("return_hidden_states", True),
    }
    cache_prefix_limits = None
    if kwargs.get("protect_loss_hidden_states", False):
        cache_prefix_limits = loss_mask_start_positions(kwargs["loss_masks"])
        # In SGLang 0.5.17, logprob_start_len is also the public per-request
        # upper bound for radix-prefix matching. We discard the logprobs; this
        # ensures that all KD-required hidden states are newly computed.
        generate_kwargs.update(
            return_logprob=True,
            logprob_start_len=cache_prefix_limits,
            top_logprobs_num=0,
        )
    if kwargs.get("image_data") is not None:
        generate_kwargs["image_data"] = kwargs["image_data"]

    outputs = engine.generate(**generate_kwargs, stream=True)
    num_samples = len(kwargs["loss_masks"])

    response_queue.put({
        "type": "generate",
        "success": True,
        "num_samples": num_samples,
    })

    received_indices = set()
    try:
        for output in outputs:
            meta_info = output["meta_info"]
            if meta_info.get("finish_reason") is None:
                continue

            idx = output["index"]
            if idx in received_indices:
                raise RuntimeError(f"Duplicate SGLang output index: {idx}")

            mask = kwargs["loss_masks"][idx]
            hidden_states = meta_info.get("hidden_states")
            if not hidden_states:
                raise RuntimeError(
                    "SGLang returned an empty hidden_states result "
                    f"(index={idx}, finish_reason={meta_info.get('finish_reason')}, "
                    f"prompt_tokens={meta_info.get('prompt_tokens')}, "
                    f"completion_tokens={meta_info.get('completion_tokens')})"
                )
            raw_hidden_states = np.asarray(hidden_states[0])
            if (
                kwargs.get("image_data") is not None
                and raw_hidden_states.shape[0] != mask.shape[0]
            ):
                # Preserve the established multimodal expansion behavior in the
                # colocated mode. persistent_remote rejects multimodal inputs.
                num_loss_tokens = int(mask.sum())
                if num_loss_tokens >= raw_hidden_states.shape[0]:
                    raise ValueError(
                        f"Cannot select {num_loss_tokens} multimodal loss tokens "
                        f"from {raw_hidden_states.shape[0]} hidden states"
                    )
                hs_np = raw_hidden_states[-num_loss_tokens - 1:-1]
                alignment = {
                    "input_length": int(mask.shape[0]),
                    "returned_length": int(raw_hidden_states.shape[0]),
                    "selected_length": num_loss_tokens,
                    "multimodal_tail_selection": True,
                }
            else:
                hs_np, alignment = select_loss_hidden_states(
                    raw_hidden_states, mask, sample_index=idx
                )
                if cache_prefix_limits is not None:
                    alignment["radix_cache_prefix_limit"] = cache_prefix_limits[idx]

            if not hs_np.flags['C_CONTIGUOUS']:
                hs_np = np.ascontiguousarray(hs_np)

            hs_tensor = torch.from_numpy(hs_np).share_memory_()
            metadata = {
                key: value
                for key, value in meta_info.items()
                if key != "hidden_states"
                and isinstance(value, (str, int, float, bool, type(None)))
            }
            metadata["hidden_state_alignment"] = alignment
            hidden_queue.put((idx, hs_tensor, metadata))
            received_indices.add(idx)

        if len(received_indices) != num_samples:
            raise RuntimeError(
                f"SGLang returned {len(received_indices)} final outputs for "
                f"{num_samples} samples"
            )
        hidden_queue.put(None)
    except Exception:
        import traceback
        error = traceback.format_exc()
        logger.error(f"[_handle_generate] ERROR:\n{error}")
        hidden_queue.put(error)


def _handle_sleep(engine, request, config, response_queue):
    """Handle a sleep request: offload GPU memory."""
    tags = request.get("tags", config.offload_tags)
    torch.cuda.empty_cache()
    engine.release_memory_occupation(tags=_normalize_tags(tags))
    response_queue.put({"type": "sleep", "success": True, "tags": tags})


def _handle_wakeup(engine, request, config, response_queue):
    """Handle a wakeup request: restore GPU memory."""
    tags = request.get("tags", config.offload_tags)
    torch.cuda.empty_cache()
    engine.resume_memory_occupation(tags=_normalize_tags(tags))
    response_queue.put({"type": "wakeup", "success": True, "tags": tags})


def _handle_update_weights_from_tensor(engine, request, response_queue):
    """Handle a update_weights_from_tensor request: update weights from student (for self-distillation)."""
    serialized_named_tensors = request["kwargs"]["serialized_named_tensors"]
    load_format = request["kwargs"]["load_format"]
    flush_cache = request["kwargs"]["flush_cache"]
    engine.update_weights_from_tensor(
        named_tensors=serialized_named_tensors,
        load_format=load_format,
        flush_cache=flush_cache,
    )
    response_queue.put({"type": "update_weights_from_tensor", "success": True})


class SGLangEngineService:
    """Manages SGLang Engine in a subprocess with torch multiprocessing communication."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.process: Optional[mp.Process] = None
        self.request_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self.hidden_queue: Optional[Queue] = None
        self._started = False

    def start(self, timeout: float = 1800.0):
        """Start the SGLang Engine in a subprocess."""
        if self._started:
            raise RuntimeError("Service already started")

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        self.hidden_queue = mp.Queue(maxsize=2)

        self.process = mp.Process(
            target=_engine_worker,
            args=(self.config, self.request_queue, self.response_queue, self.hidden_queue),
        )
        self.process.start()

        try:
            response = self.response_queue.get(timeout=timeout)
            if response.get("type") == "init_done" and response.get("success"):
                self._started = True
            else:
                raise RuntimeError(f"Init failed: {response.get('error')}")
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Engine initialization failed: {e}")

    def generate(
        self,
        input_ids: List[List[int]],
        loss_masks: List[np.ndarray],
        sampling_params: Dict[str, Any],
        return_hidden_states: bool = True,
        image_data=None,
        return_metadata: bool = False,
        protect_loss_hidden_states: bool = False,
    ) -> List[np.ndarray]:
        """Run generation and return hidden states via shared-memory tensors.
        
        Args:
            input_ids: Per-sample token-id lists fed directly to SGLang (avoids a
                re-tokenization round trip, so hidden-state length matches the mask).
            loss_masks: Pre-computed boolean masks for selecting response hidden states.
            sampling_params: Sampling parameters (e.g. max_new_tokens=0 for prefill-only).
            return_hidden_states: Whether to return hidden states.
            image_data: Optional list of image data for multimodal models.
            return_metadata: Return ``(hidden_states, metadata)`` per sample.
            protect_loss_hidden_states: Prevent radix reuse from extending into
                positions selected by ``loss_masks``. This uses SGLang's public
                ``logprob_start_len`` cache boundary.
        """
        if not self._started:
            raise RuntimeError("Service not started")

        if len(input_ids) != len(loss_masks):
            raise ValueError(
                f"input_ids and loss_masks must have the same batch size, got "
                f"{len(input_ids)} and {len(loss_masks)}"
            )

        # Check if subprocess is still alive before sending request
        if self.process and not self.process.is_alive():
            raise RuntimeError(
                f"[SGLangEngineService] Engine subprocess (PID={self.process.pid}) is dead! "
                f"exitcode={self.process.exitcode}"
            )

        kwargs = {
            "input_ids": input_ids,
            "loss_masks": loss_masks,
            "sampling_params": sampling_params,
            "return_hidden_states": return_hidden_states,
            "protect_loss_hidden_states": protect_loss_hidden_states,
        }
        if image_data is not None:
            kwargs["image_data"] = image_data

        self.request_queue.put({"type": "generate", "kwargs": kwargs})

        response = self._get_response(req_type="generate", timeout=600)
        if not response.get("success"):
            raise RuntimeError(f"Generate failed: {response.get('error')}")

        num_samples = response["num_samples"]
        hidden_states = [None] * num_samples
        received_count = 0
        t_recv_start = time.time()
        while True:
            try:
                message = self.hidden_queue.get(timeout=300)
                if message is None:
                    break
                if isinstance(message, str):
                    raise RuntimeError(f"Generate failed in engine subprocess:\n{message}")

                idx, hs_tensor, metadata = message
                hs_np = hs_tensor.numpy().copy()
                del hs_tensor
                hidden_states[idx] = (
                    (hs_np, metadata) if return_metadata else hs_np
                )
                received_count += 1
            except queue.Empty:
                elapsed_total = time.time() - t_recv_start
                raise RuntimeError(
                    f"Hidden state recv timeout while receiving hidden states, "
                    f"received={received_count}/{num_samples}, "
                    f"total_elapsed={elapsed_total:.1f}s"
                )

        return hidden_states

    def flush_cache(self):
        """Flush the engine radix cache."""
        if not self._started:
            return
        self.request_queue.put({"type": "flush_cache"})
        response = self._get_response(req_type="flush_cache", timeout=300)
        if not response.get("success"):
            raise RuntimeError(f"flush_cache failed: {response.get('error')}")

    def sleep(self, tags: Optional[str] = "all"):
        """Release GPU memory."""
        if not self._started:
            return
        self.request_queue.put({"type": "sleep", "tags": tags})
        response = self._get_response(req_type="sleep", timeout=300)
        if not response.get("success"):
            raise RuntimeError(f"Sleep failed: {response.get('error')}")
        return response.get("tags")

    def wakeup(self, tags: Optional[str] = "all"):
        """Resume GPU memory."""
        if not self._started:
            return
        self.request_queue.put({"type": "wakeup", "tags": tags})
        response = self._get_response(req_type="wakeup", timeout=300)
        if not response.get("success"):
            raise RuntimeError(f"Wakeup failed: {response.get('error')}")
        return response.get("tags")
    
    def update_weights_from_tensor(
        self, serialized_named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None, flush_cache: bool = True):
        kwargs = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        self.request_queue.put({"type": "update_weights_from_tensor", "kwargs": kwargs})
        response = self._get_response(req_type="update_weights_from_tensor", timeout=300)
        if not response.get("success"):
            raise RuntimeError(f"update_weights_from_tensor failed: {response.get('error')}")

    def _get_response(self, req_type="unknown", timeout=600, check_interval=10):
        elapsed = 0
        while elapsed < timeout:
            try:
                return self.response_queue.get(timeout=check_interval)
            except queue.Empty:
                elapsed += check_interval
                if self.process and not self.process.is_alive():
                    raise RuntimeError(
                        f"Engine subprocess (PID={self.process.pid}) died during '{req_type}'! "
                        f"exitcode={self.process.exitcode}"
                    )
        raise RuntimeError(f"Response timeout after {timeout}s during '{req_type}'")

    def shutdown(self):
        """Shutdown the subprocess gracefully."""
        if not self._started:
            return
        self._started = False
        self._cleanup()

    def _cleanup(self):
        """Clean up subprocess, queues and shared memory."""
        if self.request_queue:
            try:
                self.request_queue.put(None)
            except Exception:
                pass

        if self.process:
            self.process.join(timeout=30)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()

        self.process = None
        self.request_queue = None
        self.response_queue = None
        self.hidden_queue = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
