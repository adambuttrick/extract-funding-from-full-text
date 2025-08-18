import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def find_markdown_files(directory: str) -> List[str]:
    md_files = []
    path = Path(directory)
    
    if not path.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    for file_path in path.rglob('*.md'):
        md_files.append(str(file_path))
    
    return sorted(md_files)


def get_file_hash(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    return {
        'processed_files': {},
        'last_update': None,
        'total_processed': 0
    }


def save_checkpoint(checkpoint_file: str, checkpoint_data: Dict[str, Any]):
    checkpoint_data['last_update'] = datetime.now().isoformat()
    checkpoint_data['total_processed'] = len(checkpoint_data.get('processed_files', {}))
    
    temp_file = checkpoint_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    os.replace(temp_file, checkpoint_file)




def merge_results(existing_results: Dict, new_results: List) -> Dict:
    for result in new_results:
        if hasattr(result, 'filename'):
            existing_results[result.filename] = result
    
    return existing_results


def format_elapsed_time(start_time: float, end_time: float) -> str:
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"