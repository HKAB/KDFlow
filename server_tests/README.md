# Remote teacher server tests

Run these in order with the same container used for training:

1. `sbatch server_tests/run_teacher_prefix_cache_benchmark.slurm`
2. Compare the JSON `request` records for cache enabled and disabled. Send back
   the complete stdout/stderr logs; the useful fields are latency, exact prompt
   LCP, and `metadata.hidden_state_alignment`.
3. Start `examples/on_policy_kd/run_gemma4_31b_teacher_head.slurm` and copy the
   printed `TEACHER_RAY_ADDRESS`.
4. Submit the student smoke test with
   `sbatch --export=ALL,TEACHER_RAY_ADDRESS=<ip>:6379,MAX_SAMPLES=64 examples/on_policy_kd/run_gemma4_e2b_student_rollout.slurm`.
   Remove `MAX_SAMPLES` for the full run.

The benchmark intentionally runs on the real 31B teacher and is not part of the
local unit-test suite. The local alignment tests contain no model or GPU work.
