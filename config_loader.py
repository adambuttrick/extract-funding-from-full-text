import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_config_path(config_type: str, filename: str, custom_config_dir: Optional[str] = None) -> Path:
    if custom_config_dir:
        base_dir = Path(custom_config_dir)
    else:
        base_dir = Path(__file__).parent / 'configs'
    
    return base_dir / config_type / filename


def load_queries(queries_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> Dict[str, str]:
    if queries_file:
        config_path = Path(queries_file)
    else:
        config_path = get_config_path('queries', 'default.yaml', custom_config_dir)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Query configuration file not found at '{config_path}'. "
            f"Please ensure the config file exists or provide a valid path."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        return data.get('queries', {})


def load_extraction_prompt(prompt_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> str:
    if prompt_file:
        config_path = Path(prompt_file)
    else:
        config_path = get_config_path('prompts', 'extraction_prompt.txt', custom_config_dir)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Extraction prompt file not found at '{config_path}'. "
            f"Please ensure the config file exists or provide a valid path."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_extraction_examples(examples_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    if examples_file:
        config_path = Path(examples_file)
    else:
        config_path = get_config_path('prompts', 'extraction_examples.json', custom_config_dir)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Extraction examples file not found at '{config_path}'. "
            f"Please ensure the config file exists or provide a valid path."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_funding_patterns(patterns_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> List[str]:
    if patterns_file:
        config_path = Path(patterns_file)
    else:
        config_path = get_config_path('patterns', 'funding_patterns.yaml', custom_config_dir)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Funding patterns file not found at '{config_path}'. "
            f"Please ensure the config file exists or provide a valid path."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        return data.get('patterns', [])