"""WebSocket server handler for Aseprite plugin communication."""

import json
import asyncio
import logging

import websockets
import numpy as np

from server.config import ServerConfig
from server.inference.pipeline import InferencePipeline
from server.utils.image_io import image_to_rgba_bytes

logger = logging.getLogger(__name__)


class GenerationHandler:
    """Handles WebSocket connections and dispatches generation requests."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.pipeline: InferencePipeline | None = None
        self._cancel = False

    def load_pipeline(self):
        logger.info("Loading inference pipeline...")
        self.pipeline = InferencePipeline(self.config)
        logger.info("Pipeline ready.")

    async def handle_connection(self, websocket):
        """Handle a single WebSocket connection."""
        logger.info("Client connected")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_text(websocket, message)
                else:
                    logger.warning("Unexpected binary message from client")
        except websockets.ConnectionClosed:
            logger.info("Client disconnected")

    async def _handle_text(self, websocket, message: str):
        """Parse JSON request and dispatch."""
        try:
            req = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            return

        action = req.get("action")

        if action == "generate":
            await self._handle_generate(websocket, req)
        elif action == "cancel":
            self._cancel = True
        elif action == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown action: {action}",
            }))

    async def _handle_generate(self, websocket, req: dict):
        """Run generation and send results back."""
        self._cancel = False

        if self.pipeline is None:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Pipeline not loaded",
            }))
            return

        prompt = req.get("prompt", "pixel art sprite")
        size = req.get("size", 128)
        palette = req.get("palette")
        guidance_scale = req.get("guidance_scale", self.config.default_guidance_scale)
        steps = req.get("steps", self.config.default_num_steps)
        seed = req.get("seed", -1)
        dither_mode = req.get("dither_mode")
        outline_cleanup = req.get("outline_cleanup", True)
        num_colors = req.get("num_colors", 16)

        if size not in self.config.supported_sizes:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unsupported size: {size}. Use {self.config.supported_sizes}",
            }))
            return

        async def progress_callback(step, total):
            if self._cancel:
                raise asyncio.CancelledError()
            await websocket.send(json.dumps({
                "type": "progress",
                "step": step,
                "total": total,
            }))

        try:
            # Run generation in thread pool to not block event loop
            loop = asyncio.get_event_loop()

            def sync_progress(step, total):
                asyncio.run_coroutine_threadsafe(
                    progress_callback(step, total), loop
                )

            image, palette_out = await loop.run_in_executor(
                None,
                lambda: self.pipeline.generate(
                    prompt=prompt,
                    size=size,
                    palette_hex=palette,
                    guidance_scale=guidance_scale,
                    num_steps=steps,
                    seed=seed,
                    dither_mode=dither_mode,
                    outline_cleanup=outline_cleanup,
                    num_colors=num_colors,
                    progress_callback=sync_progress,
                ),
            )

            # Send result header
            palette_hex = [
                f"#{r:02x}{g:02x}{b:02x}"
                for r, g, b in palette_out.tolist()
            ]
            await websocket.send(json.dumps({
                "type": "result",
                "width": image.shape[1],
                "height": image.shape[0],
                "palette": palette_hex,
            }))

            # Send image data as binary
            await websocket.send(image_to_rgba_bytes(image))

        except asyncio.CancelledError:
            await websocket.send(json.dumps({
                "type": "cancelled",
            }))
        except Exception as e:
            logger.exception("Generation failed")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e),
            }))


async def start_server(config: ServerConfig):
    """Start the WebSocket server."""
    handler = GenerationHandler(config)
    handler.load_pipeline()

    logger.info(f"Starting server on {config.host}:{config.port}")

    async with websockets.serve(
        handler.handle_connection,
        config.host,
        config.port,
        max_size=10 * 1024 * 1024,  # 10MB max message
    ):
        await asyncio.Future()  # run forever
