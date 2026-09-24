import os
import uuid
import tempfile
from typing import Tuple, Optional
from PIL import Image
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_FORENSIC_DIR = os.path.join(PROJECT_ROOT, "outputs", "forensic")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/pjpeg"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit

def ensure_output_directories() -> str:
    """Ensures the outputs/forensic directory exists and returns its absolute path."""
    os.makedirs(OUTPUT_FORENSIC_DIR, exist_ok=True)
    return OUTPUT_FORENSIC_DIR

def validate_image_bytes(file_bytes: bytes, filename: str, content_type: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Strict validation of uploaded image bytes:
    1. Size within limit.
    2. Extension allowed.
    3. Magic byte header verification.
    4. Image decode test with PIL without executing or writing to disk.
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "Uploaded file is empty (0 bytes)."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"File exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB)."

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    # Validate Magic Bytes
    if ext in {".jpg", ".jpeg"}:
        if not file_bytes.startswith(b"\xff\xd8\xff"):
            return False, "File header does not match a valid JPEG structure."
    elif ext == ".png":
        if not file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return False, "File header does not match a valid PNG structure."
    elif ext == ".webp":
        if not (file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP"):
            return False, "File header does not match a valid WEBP structure."

    # Verify PIL decode
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
    except Exception as e:
        return False, f"Corrupted or unreadable image data: {str(e)}"

    return True, None

def generate_unique_prefix() -> str:
    """Generates an unguessable unique prefix for forensic artifacts to prevent overwrites."""
    return f"{uuid.uuid4().hex[:12]}_{int(uuid.uuid1().time_low)}"

def get_artifact_paths(prefix: str, artifact_name: str, ext: str = ".jpg") -> Tuple[str, str]:
    """
    Returns (absolute_filesystem_path, public_url_path).
    Example:
    'outputs/forensic/a1b2c3_ela.jpg', '/outputs/forensic/a1b2c3_ela.jpg'
    """
    ensure_output_directories()
    safe_filename = f"{prefix}_{artifact_name}{ext}"
    abs_path = os.path.join(OUTPUT_FORENSIC_DIR, safe_filename)
    url_path = f"/outputs/forensic/{safe_filename}"
    return abs_path, url_path
