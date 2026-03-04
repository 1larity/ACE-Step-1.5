"""Dataset-related training API route registration."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException

from acestep.api import train_api_models
from acestep.api.train_api_dataset_models import (
    PreprocessDatasetRequest,
    SaveDatasetRequest,
    UpdateSampleRequest,
    _serialize_samples,
)
from acestep.api.train_api_dataset_auto_label_routes import register_training_dataset_auto_label_routes
from acestep.api.train_api_dataset_scan_load_routes import register_training_dataset_scan_load_routes
from acestep.api.train_api_runtime import RuntimeComponentManager
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler


def register_training_dataset_routes(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
    wrap_response: Callable[[Any, int, Optional[str]], Dict[str, Any]],
    temporary_llm_model: Callable[[FastAPI, LLMHandler, Optional[str]], Any],
    atomic_write_json: Callable[[str, Dict[str, Any]], None],
    append_jsonl: Callable[[str, Dict[str, Any]], None],
) -> None:
    """Register dataset APIs used by training workflows."""

    register_training_dataset_scan_load_routes(
        app=app,
        verify_api_key=verify_api_key,
        wrap_response=wrap_response,
    )

    register_training_dataset_auto_label_routes(
        app=app,
        verify_api_key=verify_api_key,
        wrap_response=wrap_response,
        temporary_llm_model=temporary_llm_model,
        atomic_write_json=atomic_write_json,
        append_jsonl=append_jsonl,
    )

    @app.get("/v1/dataset/preprocess_status")
    async def get_preprocess_status_latest(_: None = Depends(verify_api_key)):
        """Get latest preprocess task status."""

        with train_api_models._preprocess_lock:
            latest_task_id = train_api_models._preprocess_latest_task_id
            if latest_task_id is None:
                return wrap_response(
                    {
                        "task_id": None,
                        "status": "idle",
                        "progress": "",
                        "current": 0,
                        "total": 0,
                    }
                )

            task = train_api_models._preprocess_tasks.get(latest_task_id)
            if task is None:
                return wrap_response(
                    {
                        "task_id": latest_task_id,
                        "status": "idle",
                        "progress": "",
                        "current": 0,
                        "total": 0,
                    }
                )

            response_data = {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "current": task.current,
                "total": task.total,
            }

            if task.status == "completed" and task.result:
                response_data["result"] = task.result
            elif task.status == "failed" and task.error:
                response_data["error"] = task.error
            return wrap_response(response_data)

    @app.get("/v1/dataset/auto_label_status")
    async def get_auto_label_status_latest(_: None = Depends(verify_api_key)):
        """Get latest auto-label task status."""

        with train_api_models._auto_label_lock:
            latest_task_id = train_api_models._auto_label_latest_task_id
            if latest_task_id is None:
                return wrap_response(
                    {
                        "task_id": None,
                        "status": "idle",
                        "progress": "",
                        "current": 0,
                        "total": 0,
                    }
                )
            task = train_api_models._auto_label_tasks.get(latest_task_id)
            if task is None:
                return wrap_response(
                    {
                        "task_id": latest_task_id,
                        "status": "idle",
                        "progress": "",
                        "current": 0,
                        "total": 0,
                    }
                )

            response_data = {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "current": task.current,
                "total": task.total,
                "save_path": task.save_path,
                "last_updated_index": task.last_updated_index,
                "last_updated_sample": task.last_updated_sample,
            }
            if task.status == "completed" and task.result:
                response_data["result"] = task.result
            elif task.status == "failed" and task.error:
                response_data["error"] = task.error
            return wrap_response(response_data)

    @app.post("/v1/dataset/save")
    async def save_dataset(request: SaveDatasetRequest, _: None = Depends(verify_api_key)):
        """Save dataset to JSON file."""

        builder = app.state.dataset_builder
        if builder is None:
            raise HTTPException(status_code=400, detail="No dataset to save")

        try:
            if request.custom_tag is not None:
                builder.metadata.custom_tag = request.custom_tag
            if request.tag_position is not None:
                builder.metadata.tag_position = request.tag_position
            if request.all_instrumental is not None:
                builder.metadata.all_instrumental = request.all_instrumental
            if request.genre_ratio is not None:
                builder.metadata.genre_ratio = request.genre_ratio

            status = builder.save_dataset(request.save_path.strip(), request.dataset_name)

            if status.startswith("✅"):
                app.state.dataset_json_path = request.save_path.strip()

            if status.startswith("✅"):
                return wrap_response({"message": status, "save_path": request.save_path})
            return wrap_response(None, code=400, error=status)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Save failed: {exc}")

    @app.post("/v1/dataset/preprocess")
    async def preprocess_dataset(request: PreprocessDatasetRequest, _: None = Depends(verify_api_key)):
        """Preprocess dataset to tensor files for training."""

        builder = app.state.dataset_builder
        if builder is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        handler: AceStepHandler = app.state.handler
        if handler is None or handler.model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        preprocess_notes = []
        llm: LLMHandler = app.state.llm_handler
        mgr = RuntimeComponentManager(handler=handler, llm=llm, app_state=app.state)
        mgr.offload_decoder_to_cpu()
        mgr.unload_llm()

        try:
            output_paths, status = await asyncio.to_thread(
                builder.preprocess_to_tensors,
                dit_handler=handler,
                output_dir=request.output_dir.strip(),
                skip_existing=request.skip_existing,
                progress_callback=None,
            )

            if status.startswith("✅"):
                if mgr.llm_unloaded:
                    status += "\nℹ️ LLM was temporarily unloaded during preprocessing and restored afterward."
                if mgr.decoder_moved:
                    status += "\nℹ️ Decoder was temporarily offloaded during preprocessing and restored afterward."
                if preprocess_notes:
                    status += "\n" + "\n".join(preprocess_notes)

                return wrap_response(
                    {
                        "message": status,
                        "output_dir": request.output_dir,
                        "num_tensors": len(output_paths),
                    }
                )
            return wrap_response(None, code=400, error=status)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Preprocessing failed: {exc}")
        finally:
            mgr.restore()

    @app.post("/v1/dataset/preprocess_async")
    async def preprocess_dataset_async(request: PreprocessDatasetRequest, _: None = Depends(verify_api_key)):
        """Start preprocessing task asynchronously and return task_id immediately."""

        builder = app.state.dataset_builder
        if builder is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        handler: AceStepHandler = app.state.handler
        if handler is None or handler.model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        task_id = str(uuid4())

        labeled_samples = [sample for sample in builder.samples if sample.labeled]
        total = len(labeled_samples)

        if total == 0:
            return wrap_response(
                {
                    "task_id": task_id,
                    "message": "No labeled samples to preprocess",
                    "total": 0,
                }
            )

        with train_api_models._preprocess_lock:
            train_api_models._preprocess_tasks[task_id] = train_api_models.PreprocessTask(
                task_id=task_id,
                status="running",
                progress="Starting preprocessing...",
                current=0,
                total=total,
                created_at=time.time(),
            )
            train_api_models._preprocess_latest_task_id = task_id

        def run_preprocessing() -> None:
            mgr = RuntimeComponentManager(handler=handler, llm=app.state.llm_handler, app_state=app.state)

            try:
                preprocess_notes = []
                mgr.offload_decoder_to_cpu()
                mgr.unload_llm()

                def progress_callback(msg: str):
                    with train_api_models._preprocess_lock:
                        task = train_api_models._preprocess_tasks.get(task_id)
                        if task:
                            import re

                            match = re.match(r"Preprocessing (\d+)/(\d+)", msg)
                            if match:
                                task.current = int(match.group(1))
                                task.progress = msg

                output_paths, status = builder.preprocess_to_tensors(
                    dit_handler=handler,
                    output_dir=request.output_dir.strip(),
                    skip_existing=request.skip_existing,
                    progress_callback=progress_callback,
                )

                if mgr.llm_unloaded:
                    status += "\nℹ️ LLM was temporarily unloaded during preprocessing and restored afterward."
                if mgr.decoder_moved:
                    status += "\nℹ️ Decoder was temporarily offloaded during preprocessing and restored afterward."
                if preprocess_notes:
                    status += "\n" + "\n".join(preprocess_notes)

                with train_api_models._preprocess_lock:
                    task = train_api_models._preprocess_tasks.get(task_id)
                    if task:
                        task.status = "completed"
                        task.progress = status
                        task.current = task.total
                        task.result = {
                            "message": status,
                            "output_dir": request.output_dir,
                            "num_tensors": len(output_paths),
                        }
            except Exception as exc:
                with train_api_models._preprocess_lock:
                    task = train_api_models._preprocess_tasks.get(task_id)
                    if task:
                        task.status = "failed"
                        task.error = str(exc)
                        task.progress = f"Failed: {exc}"
            finally:
                mgr.restore()

        import threading

        thread = threading.Thread(target=run_preprocessing, daemon=True)
        thread.start()

        return wrap_response(
            {
                "task_id": task_id,
                "message": "Preprocessing task started",
                "total": total,
            }
        )

    @app.get("/v1/dataset/preprocess_status/{task_id}")
    async def get_preprocess_status(task_id: str, _: None = Depends(verify_api_key)):
        """Get preprocessing task status and progress."""

        with train_api_models._preprocess_lock:
            task = train_api_models._preprocess_tasks.get(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            response_data = {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "current": task.current,
                "total": task.total,
            }

            if task.status == "completed" and task.result:
                response_data["result"] = task.result
            elif task.status == "failed" and task.error:
                response_data["error"] = task.error

            return wrap_response(response_data)

    @app.get("/v1/dataset/samples")
    async def get_all_samples(_: None = Depends(verify_api_key)):
        """Get all samples in the current dataset."""

        builder = app.state.dataset_builder
        if builder is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        return wrap_response(
            {
                "dataset_name": builder.metadata.name,
                "num_samples": len(builder.samples),
                "labeled_count": builder.get_labeled_count(),
                "samples": _serialize_samples(builder),
            }
        )

    @app.get("/v1/dataset/sample/{sample_idx}")
    async def get_sample(sample_idx: int, _: None = Depends(verify_api_key)):
        """Get a specific sample by index."""

        builder = app.state.dataset_builder
        if builder is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        if sample_idx < 0 or sample_idx >= len(builder.samples):
            raise HTTPException(status_code=404, detail=f"Sample index {sample_idx} out of range")

        sample = builder.samples[sample_idx]
        payload = sample.to_dict()
        payload["index"] = sample_idx
        return wrap_response(payload)

    @app.put("/v1/dataset/sample/{sample_idx}")
    async def update_sample(sample_idx: int, request: UpdateSampleRequest, _: None = Depends(verify_api_key)):
        """Update a sample's metadata."""

        builder = app.state.dataset_builder
        if builder is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")

        try:
            sample, status = builder.update_sample(
                sample_idx,
                caption=request.caption,
                genre=request.genre,
                prompt_override=request.prompt_override,
                lyrics=request.lyrics if not request.is_instrumental else "[Instrumental]",
                bpm=request.bpm,
                keyscale=request.keyscale,
                timesignature=request.timesignature,
                language="unknown" if request.is_instrumental else request.language,
                is_instrumental=request.is_instrumental,
                labeled=True,
            )

            if status.startswith("✅"):
                sample_payload = sample.to_dict()
                sample_payload["index"] = sample_idx
                return wrap_response({"message": status, "sample": sample_payload})
            return wrap_response(None, code=400, error=status)
        except Exception as exc:
            return wrap_response(None, code=500, error=f"Update failed: {exc}")

