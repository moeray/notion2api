"""Request-local image bytes, kept outside JSON history and durable session state."""
from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass

MAX_IMAGE_BYTES = 20 * 1024 * 1024
# Notion accepts 20 content blocks including the bridge's text block.
MAX_TURN_IMAGES = 19
IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}


class ImageError(ValueError):
    pass


@dataclass(frozen=True)
class Image:
    media_type: str
    data: bytes


class ImagePrompt(str):
    def __new__(cls, text, images):
        value = super().__new__(cls, text)
        value.images = images
        return value


class FilePrompt(str):
    def __new__(cls, text, file_ids):
        value = super().__new__(cls, text)
        value.file_ids = file_ids
        return value


def normalize_image(block, images):
    source = block.get("source")
    if not isinstance(source, dict) or source.get("type") != "base64":
        raise ImageError("Images require a base64 source; URL and file_id image sources are not supported")
    media_type, encoded = source.get("media_type"), source.get("data")
    if not isinstance(media_type, str) or media_type not in IMAGE_EXTENSIONS:
        raise ImageError("Supported image formats: PNG, JPEG, GIF and WebP")
    if not isinstance(encoded, str) or not encoded:
        raise ImageError("Image base64 data must be a non-empty string")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageError("Each image must be at most 20 MiB")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ImageError("Invalid image base64 data") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError("Each image must be at most 20 MiB")
    signatures = {
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": data.startswith(b"\xff\xd8\xff"),
        "image/gif": data.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
    }
    if not signatures[media_type]:
        raise ImageError("Image data does not match its declared media_type")
    image_id = hashlib.sha256(media_type.encode() + b"\0" + data).hexdigest()
    images[image_id] = Image(media_type, data)
    return {"type": "image", "image_id": image_id, "media_type": media_type}


def images_in_messages(messages, images):
    """Visit only normalized content, never arbitrary tool arguments or user text."""
    selected = {}

    def visit(blocks):
        for block in blocks:
            if block["type"] == "image":
                image_id = block["image_id"]
                selected[image_id] = images[image_id]
            elif block["type"] == "tool_result":
                visit(block["content"])

    for message in messages:
        visit(message["content"])
    return selected
