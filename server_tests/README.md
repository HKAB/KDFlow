# Remote teacher server tests

Run these in order with the same container used for training:

1. `sbatch server_tests/run_teacher_prefix_cache_benchmark.slurm`
2. Compare the JSON `request` records for cache enabled and disabled. Send back
   the complete stdout/stderr logs; the useful fields are latency, exact prompt
   LCP, and `metadata.hidden_state_alignment`.
3. For one 8-GPU worker, run the preferred single-node smoke test:
   `sbatch --export=ALL,MAX_SAMPLES=64 examples/on_policy_kd/run_gemma4_31b_remote_teacher.slurm`.
4. For two separate workers instead, start
   `examples/on_policy_kd/run_gemma4_31b_teacher_head.slurm` and copy the
   printed `TEACHER_RAY_ADDRESS` and `TEACHER_HOSTNAME`.
5. Submit the separate student smoke test with
   `sbatch --exclude=<teacher-hostname> --export=ALL,TEACHER_RAY_ADDRESS=<ip>:6379,TEACHER_HOSTNAME=<teacher-hostname>,MAX_SAMPLES=64 examples/on_policy_kd/run_gemma4_e2b_student_rollout.slurm`.
   Remove `MAX_SAMPLES` for the full run.

The benchmark intentionally runs on the real 31B teacher and is not part of the
local unit-test suite. The local alignment tests contain no model or GPU work.
