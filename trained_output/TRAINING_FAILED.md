# ❌ Training Failed

- **Date**: Mar 27, 2026 20:02 UTC
- **Command**: `python3 scripts/train.py`
- **Duration**: 4s
- **GPU**: NVIDIA H200
- **Exit code**: 1

## Error Log (last 20 lines)

```
2026-03-27 20:01:59 | INFO     | handflow.trainer |   touch: 527 samples, weight=0.846
2026-03-27 20:01:59 | INFO     | handflow.trainer | Class weighting enabled for training
2026-03-27 20:01:59 | INFO     | handflow.trainer | Callbacks will monitor: val_loss (loss), val_accuracy (accuracy)
2026-03-27 20:01:59 | INFO     | handflow.trainer | Early stopping patience: 25 epochs
2026-03-27 20:01:59 | INFO     | handflow.trainer | Training with on-the-fly augmentation...
/usr/local/lib/python3.10/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/1000
Traceback (most recent call last):
  File "/workspace/project/scripts/train.py", line 241, in <module>
    main()
  File "/workspace/project/s
```

Trained on Akash Network decentralized GPU cloud.
