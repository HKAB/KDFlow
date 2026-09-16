import asyncio
import time
from typing import Any, Dict, List, Optional

from kdflow.trainer.data_processor import RolloutDataProcessor
from kdflow.utils.logging_utils import init_logger
from kdflow.utils.structured_output import find_invalid_rollout_regex_outputs

logger = init_logger(__name__)


class RolloutManager:
    """Manage batched rollout generation and data processing."""

    def __init__(
        self,
        strategy,
        rollout_group,
        is_same_tokenizer: bool,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.args = strategy.args
        self.rollout_group = rollout_group
        self.generate_kwargs = generate_kwargs
        self.max_concurrent = (
            self.args.rollout.rollout_engine_concurrency
            * self.rollout_group.num_actors
        )
        if self.max_concurrent <= 0:
            raise ValueError(
                f"max_concurrent must be positive, got {self.max_concurrent}"
            )

        self.data_processor = RolloutDataProcessor(strategy, is_same_tokenizer)
        self.image_key = self.data_processor.image_key
        self.dp_size = (
            self.args.train.num_nodes * self.args.train.num_gpus_per_node
        ) // self.args.model.ring_attn_size
        self.train_sample_alignment = (
            self.dp_size * self.args.train.micro_train_batch_size
        )

    def rollout(
        self,
        prompt_batch: List[Dict[str, Any]],
        global_step: int,
        mode: str = "train",
        **kwargs,
    ) -> tuple[List[dict], Dict[str, float]]:
        """Generate rollout samples and convert them into micro-batches."""
        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported rollout mode: {mode!r}")
        if not prompt_batch:
            return [], {}

        n_samples = self.args.rollout.n_samples_per_prompt if mode == "train" else 1
        if n_samples <= 0:
            raise ValueError(f"n_samples_per_prompt must be positive, got {n_samples}")

        should_sleep = self.args.train.enable_sleep
        if should_sleep:
            self.rollout_group.wakeup()

        try:
            stu_prompts = [
                item["stu_prompt"]
                for item in prompt_batch
                for _ in range(n_samples)
            ]
            tea_prompts = [
                item["tea_prompt"]
                for item in prompt_batch
                for _ in range(n_samples)
            ]
            labels = [
                item["label"] for item in prompt_batch for _ in range(n_samples)
            ]
            images = None
            if self.image_key:
                images = [
                    item.get("images")
                    for item in prompt_batch
                    for _ in range(n_samples)
                ]
            teacher_routing_keys = None
            if self.args.kd.multi_teacher_config:
                teacher_routing_keys = [
                    item.get("teacher_routing_key")
                    for item in prompt_batch
                    for _ in range(n_samples)
                ]

            sampling_params = kwargs or self.generate_kwargs
            if not sampling_params:
                raise ValueError("sampling_params must not be empty")

            outputs, timing_metrics = self._generate(
                stu_prompts,
                sampling_params,
                image_data=images,
            )
            outputs, invalid_indices, structured_metrics = (
                self._retry_invalid_structured_outputs(
                    stu_prompts,
                    outputs,
                    sampling_params,
                    image_data=images,
                )
            )

            valid_indices = [
                index
                for index in range(len(outputs))
                if index not in invalid_indices
            ]
            valid_before_alignment = len(valid_indices)
            if mode == "train":
                usable_count = (
                    len(valid_indices)
                    // self.train_sample_alignment
                    * self.train_sample_alignment
                )
                valid_indices = valid_indices[:usable_count]
            if not valid_indices:
                raise RuntimeError(
                    "No valid structured rollout samples remain after retries"
                )

            def select(values):
                if values is None:
                    return None
                return [values[index] for index in valid_indices]

            total_outputs = len(outputs)
            outputs = select(outputs)
            stu_prompts = select(stu_prompts)
            tea_prompts = select(tea_prompts)
            labels = select(labels)
            images = select(images)
            teacher_routing_keys = select(teacher_routing_keys)
            structured_metrics["structured_output/dropped_ratio"] = (
                total_outputs - len(outputs)
            ) / total_outputs
            structured_metrics["structured_output/dp_alignment_drop_ratio"] = (
                valid_before_alignment - len(outputs)
            ) / total_outputs

            micro_batches, rollout_metrics = self.data_processor.process(
                stu_prompts=stu_prompts,
                tea_prompts=tea_prompts,
                outputs=outputs,
                labels=labels,
                sampling_params=sampling_params,
                global_step=global_step,
                mode=mode,
                images=images,
                teacher_routing_keys=teacher_routing_keys,
            )
            rollout_metrics.update(timing_metrics)
            rollout_metrics.update(structured_metrics)
            return micro_batches, rollout_metrics
        finally:
            if should_sleep:
                self.rollout_group.sleep()

    def _retry_invalid_structured_outputs(
        self,
        prompts: List[str],
        outputs: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        image_data: Optional[List] = None,
    ) -> tuple[List[Dict[str, Any]], set[int], Dict[str, float]]:
        """Retry regex-invalid samples and return any persistent failures."""
        regex = sampling_params.get("regex")
        initial_invalid = find_invalid_rollout_regex_outputs(
            (output["text"] for output in outputs), regex
        )
        invalid_indices = set(initial_invalid)
        retry_requests = 0
        retry_start = time.perf_counter()

        retry_params = dict(sampling_params)
        base_temperature = float(retry_params.get("temperature", 0.0))
        retry_temperature = self.args.rollout.rollout_regex_retry_temperature
        retry_params["temperature"] = (
            min(base_temperature, retry_temperature)
            if base_temperature > 0
            else retry_temperature
        )

        for attempt in range(self.args.rollout.rollout_regex_max_retries):
            if not invalid_indices:
                break
            retry_indices = sorted(invalid_indices)
            retry_requests += len(retry_indices)
            logger.warning(
                "Retrying %d regex-invalid rollout samples (attempt %d/%d)",
                len(retry_indices),
                attempt + 1,
                self.args.rollout.rollout_regex_max_retries,
            )
            retry_outputs, _ = self._generate(
                [prompts[index] for index in retry_indices],
                retry_params,
                image_data=(
                    [image_data[index] for index in retry_indices]
                    if image_data is not None
                    else None
                ),
            )
            for index, retry_output in zip(retry_indices, retry_outputs):
                outputs[index] = retry_output
            invalid_indices = set(
                find_invalid_rollout_regex_outputs(
                    (output["text"] for output in outputs), regex
                )
            )

        if invalid_indices:
            details = []
            for index in sorted(invalid_indices)[:5]:
                output = outputs[index]
                finish_reason = output.get("meta_info", {}).get("finish_reason")
                details.append(
                    f"{index}:tokens={len(output.get('output_ids', []))},"
                    f"finish={finish_reason!r}"
                )
            logger.warning(
                "Skipping %d persistently invalid rollout samples (%s)",
                len(invalid_indices),
                "; ".join(details),
            )

        total = len(outputs)
        recovered = len(initial_invalid) - len(invalid_indices)
        metrics = {
            "structured_output/initial_invalid_ratio": len(initial_invalid) / total,
            "structured_output/recovered_ratio": recovered / total,
            "structured_output/persistent_invalid_ratio": (
                len(invalid_indices) / total
            ),
            "structured_output/retry_requests": float(retry_requests),
            "timing/structured_output_retry": time.perf_counter() - retry_start,
        }
        return outputs, invalid_indices, metrics

    def _generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        image_data: Optional[List] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Run ordered concurrent generation requests."""
        if not prompts:
            return [], {}
        if image_data is not None and len(image_data) != len(prompts):
            raise ValueError("image_data and prompts must have the same length")

        max_concurrent = min(len(prompts), self.max_concurrent)
        try:
            return asyncio.run(
                self._async_generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    max_concurrent=max_concurrent,
                    image_data=image_data,
                )
            )
        except Exception:
            try:
                engine_health = self.rollout_group.health_check()
            except Exception as health_error:
                engine_health = f"unavailable ({health_error!r})"
            logger.exception(
                "Rollout generation failed: prompts=%d, max_concurrent=%d, "
                "engine_health=%s",
                len(prompts),
                max_concurrent,
                engine_health,
            )
            raise

    async def _async_generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        max_concurrent: int,
        image_data: Optional[List] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Schedule single-sample generation requests concurrently."""
        import aiohttp

        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(prompts)
        rollout_times = [0.0] * len(prompts)

        async def run_request(
            index: int, prompt: str, session: aiohttp.ClientSession
        ) -> None:
            try:
                async with semaphore:
                    start = time.perf_counter()
                    results[index] = await self.rollout_group.generate_one(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        session=session,
                        image_data=(
                            image_data[index] if image_data is not None else None
                        ),
                    )
                    rollout_times[index] = time.perf_counter() - start
            except Exception as error:
                raise RuntimeError(
                    f"Rollout generation failed for request {index}"
                ) from error

        connector = aiohttp.TCPConnector(limit=max_concurrent)
        timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=60)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            await asyncio.gather(
                *(
                    run_request(index, prompt, session)
                    for index, prompt in enumerate(prompts)
                )
            )

        timing_metrics = {
            "timing/rollout_per_sample/mean": sum(rollout_times) / len(rollout_times),
            "timing/rollout_per_sample/max": max(rollout_times),
        }
        return results, timing_metrics
