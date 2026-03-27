# ❌ Training Failed

- **Date**: Mar 27, 2026 19:34 UTC
- **Command**: `python3 scripts/train.py`
- **Duration**: 3s
- **GPU**: NVIDIA GeForce RTX 4090
- **Exit code**: 1

## Error Log (last 20 lines)

```
    self.tracker.start_run(run_name, run_config)
  File "/workspace/project/src/handflow/utils/experiment_tracker.py", line 426, in start_run
    tracker.start_run(run_name, config)
  File "/workspace/project/src/handflow/utils/experiment_tracker.py", line 106, in start_run
    self._run = self._wandb.init(
  File "/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py", line 1595, in init
    get_sentry().reraise(e)
  File "/usr/local/lib/python3.10/dist-packages/wandb/analytics/sentry.py", line 190, in reraise
    raise exc.with_traceback(tb)
  File "/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py", line 1516, in init
    wi.maybe_login(init_settings)
  File "/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py", line 193, in maybe_login
    wandb_login._login(
  File "/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_login.py", line 190, in _login
    auth = _find_or_prompt_for_key(
  File "/usr/local/lib/python3.10/dist-packages/wandb/sdk/w
```

Trained on Akash Network decentralized GPU cloud.
