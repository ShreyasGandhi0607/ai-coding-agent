from pathlib import Path

def resolve_path(base: str | Path, path: str | Path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()   
    
    return Path(base).resolve() / path # Users/Shreyas/Desktop/ai-agent tools/base.py 

def is_binary_file(path: Path) -> bool:
    try:
        # check for null byte in first 8192 bytes of the file
        with open(path, 'rb') as file:
            chunk = file.read(8192)
            return f"\x00" in chunk
    except (OSError, IOError):
        return False
