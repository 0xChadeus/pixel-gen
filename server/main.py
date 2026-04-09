"""Entry point for the pixel_gen backend server."""

import argparse
import asyncio
import logging

from server.config import ServerConfig
from server.ws_handler import start_server


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Pixel Gen inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9847)
    parser.add_argument("--checkpoint", default="", help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    args = parser.parse_args()

    config = ServerConfig(
        host=args.host,
        port=args.port,
        checkpoint_path=args.checkpoint,
        device=args.device,
        clip_model=args.clip_model,
    )

    asyncio.run(start_server(config))


if __name__ == "__main__":
    main()
