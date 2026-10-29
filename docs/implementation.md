# Implementation details

The core functions in `controllable_generation_TV_lora.py` have the same syntax trees as those in the supplied source archive. Imports that were not used by these functions were removed. The following choices are retained:

- NCSN++ configuration `AAPM_256_ncsnpp_continuous`, VE-SDE, noise scales 0.01 to 378.
- Predictor followed by one Langevin corrector for each slice batch; concatenate batches before the volume correction.
- Three circular forward differences and the supplied transpose routines.
- One ADMM correction and one CG recurrence per reverse step.
- Batched SVD on `[S,1,HW]`, equivalent to independent slice group shrinkage.
- Final normalized back-projection correction when enabled. This is one correction, not a guarantee of an exactly zero measurement residual.
- The sampler's effective reverse-time endpoint is `1e-10`, because the supplied function overwrites its `eps` argument. The release retains that behavior.
- The supplied `CT.AT` interpolation backprojector is retained; this implementation does not substitute an autograd-generated transpose.
- The legacy whole-volume min-max export is retained, while the raw reconstruction is also saved.

The old `lam2=0.02` field was passed to an unused `lamb_2` argument and was embedded in directory names. It was not the active spectral weight. The active spectral weight is passed as `lamb_lowrank=0.5`. New records use explicit `lambda_tv`, `lambda_lr`, `rho_tv`, and `rho_lr` names.

The release entry point runs TV-LoRA only. It avoids the old invalid default `TGV`, does not import other reconstruction methods, does not overwrite the requested batch size, and fails if a checkpoint is missing. Model restoration is strict except for the intentionally regenerated noise-scale buffer. It checks the full EMA count and all EMA tensor shapes before use.

The default native FIR resampling uses the supplied native PyTorch implementation on CPU or CUDA. It avoids eager compilation during import. Optional compiled CUDA kernels remain available, but CUDA-backend compilation and numerical equivalence were not tested in the CPU review environment. Backend choice is explicit and recorded.

Experiment profiles distinguish primary-comparison parameters from the legacy CTHD qualitative parameters. The supplied source archive itself does not contain the historical launch command, initial random state, or experiment manifests needed to establish bitwise agreement with every published saved array. A profile states the requested parameters; it does not certify a reconstruction result before execution.
