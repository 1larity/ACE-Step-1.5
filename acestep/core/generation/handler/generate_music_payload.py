"""Success payload builders for ``generate_music`` orchestration."""

from typing import Any, Dict

from loguru import logger

from acestep.core.generation.handler.repaint_waveform_splice import (
    apply_repaint_waveform_splice,
)


class GenerateMusicPayloadMixin:
    """Build audio/metadata payload structures returned by ``generate_music``."""

    def _build_generate_music_success_payload(
        self,
        outputs: Dict[str, Any],
        pred_wavs,
        pred_latents_cpu,
        time_costs: Dict[str, Any],
        seed_value_for_ui: int,
        actual_batch_size: int,
        progress: Any,
        source_wavs=None,
        repainting_starts=None,
        repainting_ends=None,
        repaint_wav_crossfade_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """Assemble final success response from decoded tensors and model outputs.

        Args:
            outputs: Service output payload containing intermediate generation tensors.
            pred_wavs: Decoded waveform tensor shaped ``[batch, channels, samples]``.
            pred_latents_cpu: CPU latent tensor preserved for extra outputs.
            time_costs: Updated time-cost payload including decode/offload timings.
            seed_value_for_ui: Seed value displayed in UI outputs.
            actual_batch_size: Effective generation batch size.
            progress: Optional progress callback.
            source_wavs: Optional source-audio tensor aligned to repaint output length.
            repainting_starts: Optional batch repaint start times in seconds.
            repainting_ends: Optional batch repaint end times in seconds.
            repaint_wav_crossfade_sec: Optional waveform splice crossfade duration.

        Returns:
            Dict[str, Any]: Standard success payload returned by ``generate_music``.
        """
        logger.info("[generate_music] VAE decode completed. Preparing audio tensors...")
        if progress:
            progress(0.99, desc="Preparing audio data...")

        if (
            source_wavs is not None
            and repainting_starts is not None
            and repainting_ends is not None
        ):
            pred_wavs = apply_repaint_waveform_splice(
                pred_wavs=pred_wavs,
                src_wavs=source_wavs,
                repainting_starts=repainting_starts,
                repainting_ends=repainting_ends,
                sample_rate=self.sample_rate,
                crossfade_duration=max(0.0, float(repaint_wav_crossfade_sec)),
            )

        audio_tensors = []
        for index in range(actual_batch_size):
            audio_tensor = pred_wavs[index].cpu()
            audio_tensors.append(audio_tensor)
        # Free the GPU waveform tensor now that all per-sample CPU copies are done.
        del pred_wavs

        status_message = "Generation completed successfully!"
        logger.info(f"[generate_music] Done! Generated {len(audio_tensors)} audio tensors.")

        src_latents = outputs.get("src_latents")
        target_latents_input = outputs.get("target_latents_input")
        chunk_masks = outputs.get("chunk_masks")
        spans = outputs.get("spans", [])
        latent_masks = outputs.get("latent_masks")

        encoder_hidden_states = outputs.get("encoder_hidden_states")
        encoder_attention_mask = outputs.get("encoder_attention_mask")
        context_latents = outputs.get("context_latents")
        lyric_token_idss = outputs.get("lyric_token_idss")

        extra_outputs = {
            "pred_latents": pred_latents_cpu,
            "target_latents": target_latents_input.detach().cpu() if target_latents_input is not None else None,
            "src_latents": src_latents.detach().cpu() if src_latents is not None else None,
            "chunk_masks": chunk_masks.detach().cpu() if chunk_masks is not None else None,
            "latent_masks": latent_masks.detach().cpu() if latent_masks is not None else None,
            "spans": spans,
            "time_costs": time_costs,
            "seed_value": seed_value_for_ui,
            "encoder_hidden_states": (
                encoder_hidden_states.detach().cpu()
                if encoder_hidden_states is not None
                else None
            ),
            "encoder_attention_mask": (
                encoder_attention_mask.detach().cpu()
                if encoder_attention_mask is not None
                else None
            ),
            "context_latents": context_latents.detach().cpu() if context_latents is not None else None,
            "lyric_token_idss": lyric_token_idss.detach().cpu() if lyric_token_idss is not None else None,
        }

        audios = []
        for audio_tensor in audio_tensors:
            audios.append({"tensor": audio_tensor, "sample_rate": self.sample_rate})

        return {
            "audios": audios,
            "status_message": status_message,
            "extra_outputs": extra_outputs,
            "success": True,
            "error": None,
        }
