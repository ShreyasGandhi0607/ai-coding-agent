from pathlib import Path

def resolve_path(base: str | Path, path: str | Path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()   
    
    return Path(base).resolve() / path 

def display_path_rel_to_cwd( path: Path | None,cwd: Path,) -> str:
    try:
        p = Path(path)
    except Exception:
        return path
    
    if cwd:
        try:
            return p.relative_to(cwd)
        except ValueError:
            pass
    
    return str(p)

def is_binary_file(path: Path) -> bool:
    try:
        # check for null byte in first 8192 bytes of the file
        with open(path, 'rb') as file:
            chunk = file.read(8192)
            return b"\x00" in chunk
    except (OSError, IOError):
        return False
