"""Service-generate orchestration entrypoint for handler decomposition."""

from typing import Any, Dict, List, Optional, Union

import torch


class ServiceGenerateMixin:
    """Orchestrate service-level latent generation from normalized request inputs."""

    @torch.inference_mode()
    def service_generate(
        self,
        captions: Union[str, List[str]],
        lyrics: Union[str, List[str]],
        keys: Optional[Union[str, List[str]]] = None,
        target_wavs: Optional[torch.Tensor] = None,
        refer_audios: Optional[List[List[torch.Tensor]]] = None,
        metas: Optional[Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]] = None,
        vocal_languages: Optional[Union[str, List[str]]] = None,
        infer_steps: int = 60,
        guidance_scale: float = 7.0,
        seed: Optional[Union[int, List[int]]] = None,
        return_intermediate: bool = False,
        repainting_start: Optional[Union[float, List[float]]] = None,
        repainting_end: Optional[Union[float, List[float]]] = None,
        instructions: Optional[Union[str, List[str]]] = None,
        audio_cover_strength: float = 1.0,
        cover_noise_strength: float = 0.0,
        use_adg: bool = False,
        cfg_interval_start: float = 0.0,
        cfg_interval_end: float = 1.0,
        shift: float = 1.0,
        audio_code_hints: Optional[Union[str, List[str]]] = None,
        infer_method: str = "ode",
        timesteps: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Generate music latents from text/audio conditioning inputs."""
        _ = return_intermediate
        normalized = self._normalize_service_generate_inputs(
            captions=captions,
            lyrics=lyrics,
            keys=keys,
            metas=metas,
            vocal_languages=vocal_languages,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            instructions=instructions,
            audio_code_hints=audio_code_hints,
            infer_steps=infer_steps,
            seed=seed,
        )
        batch = self._prepare_batch(
            captions=normalized["captions"],
            lyrics=normalized["lyrics"],
            keys=normalized["keys"],
            target_wavs=target_wavs,
            refer_audios=refer_audios,
            metas=normalized["metas"],
            vocal_languages=normalized["vocal_languages"],
            repainting_start=normalized["repainting_start"],
            repainting_end=normalized["repainting_end"],
            instructions=normalized["instructions"],
            audio_code_hints=normalized["audio_code_hints"],
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
        )
        payload = self._unpack_service_processed_data(self.preprocess_batch(batch))
        seed_param = self._resolve_service_seed_param(normalized["seed_list"])
        self._ensure_silence_latent_on_device()
        generate_kwargs = self._build_service_generate_kwargs(
            payload=payload,
            seed_param=seed_param,
            infer_steps=normalized["infer_steps"],
            guidance_scale=guidance_scale,
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
            infer_method=infer_method,
            use_adg=use_adg,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            shift=shift,
            timesteps=timesteps,
        )
        outputs, encoder_hidden_states, encoder_attention_mask, context_latents = (
            self._execute_service_generate_diffusion(
                payload=payload,
                generate_kwargs=generate_kwargs,
                seed_param=seed_param,
                infer_method=infer_method,
                shift=shift,
                audio_cover_strength=audio_cover_strength,
            )
        )
        return self._attach_service_generate_outputs(
            outputs=outputs,
            payload=payload,
            batch=batch,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )
