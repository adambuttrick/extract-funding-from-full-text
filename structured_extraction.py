import os
import concurrent.futures
from typing import List, Optional

import langextract as lx

from models import FunderEntity, ExtractionResult
from providers import ModelProvider, get_provider_config
from config_loader import load_extraction_prompt, load_extraction_examples


def create_extraction_prompt(prompt_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> str:
    return load_extraction_prompt(prompt_file, custom_config_dir)


def create_funding_examples(examples_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> List[lx.data.ExampleData]:
    examples_data = load_extraction_examples(examples_file, custom_config_dir)
    
    examples = []
    for example in examples_data:
        extractions = []
        for ext in example.get('extractions', []):
            extraction = lx.data.Extraction(
                extraction_class=ext['class'],
                extraction_text=ext['text']
            )
            if 'attributes' in ext:
                extraction.attributes = ext['attributes']
            extractions.append(extraction)
        
        examples.append(lx.data.ExampleData(
            text=example['text'],
            extractions=extractions
        ))
    
    return examples


def get_language_model_class(provider: ModelProvider):
    if provider == ModelProvider.GEMINI:
        return lx.inference.GeminiLanguageModel
    elif provider == ModelProvider.OLLAMA:
        return lx.inference.OllamaLanguageModel
    elif provider in (ModelProvider.OPENAI, ModelProvider.LOCAL_OPENAI):
        return lx.inference.OpenAILanguageModel
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def extract_structured_entities(
    funding_statement: str,
    provider: ModelProvider = ModelProvider.GEMINI,
    model_id: Optional[str] = None,
    model_url: Optional[str] = None,
    api_key: Optional[str] = None,
    skip_model_validation: bool = False,
    timeout: int = 60,
    debug: bool = False,
    prompt_file: Optional[str] = None,
    examples_file: Optional[str] = None,
    custom_config_dir: Optional[str] = None,
) -> ExtractionResult:
    config = get_provider_config(provider)
    
    if model_id is None:
        model_id = config.default_model
    if model_url is None:
        model_url = config.default_url
    
    prompt = create_extraction_prompt(prompt_file, custom_config_dir)
    examples = create_funding_examples(examples_file, custom_config_dir)
    
    extract_params = {
        "text_or_documents": funding_statement,
        "prompt_description": prompt,
        "examples": examples,
        "temperature": 0.1,
        "extraction_passes": 3,
        "max_workers": 1,
        "fence_output": False,
        "use_schema_constraints": True,
        "debug": debug,
    }
    
    if provider == ModelProvider.GEMINI:
        extract_params["language_model_type"] = get_language_model_class(provider)
        extract_params["model_id"] = model_id
        extract_params["api_key"] = api_key
        extract_params["language_model_params"] = {
            "timeout": timeout,
        }
    elif provider == ModelProvider.OLLAMA:
        extract_params["language_model_type"] = get_language_model_class(provider)
        extract_params["language_model_params"] = {
            "model": model_id,
            "model_url": model_url or "http://localhost:11434",
            "timeout": timeout,
        }
    elif provider in (ModelProvider.OPENAI, ModelProvider.LOCAL_OPENAI):
        if provider == ModelProvider.LOCAL_OPENAI or (model_url and not model_url.startswith("https://api.openai.com")):
            if model_url:
                os.environ["OPENAI_BASE_URL"] = model_url
            else:
                os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000")
            
            openai_model_class = get_language_model_class(provider)
            language_model = openai_model_class(
                model_id=model_id,
                api_key=api_key or "dummy-key",
                timeout=timeout,
            )
            extract_params["model"] = language_model
        else:
            extract_params["language_model_type"] = get_language_model_class(provider)
            extract_params["model_id"] = model_id
            extract_params["api_key"] = api_key
            extract_params["language_model_params"] = {
                "model_id": model_id,
                "api_key": api_key,
                "timeout": timeout,
            }
        
        extract_params["fence_output"] = True
        extract_params["use_schema_constraints"] = False
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lx.extract, **extract_params)
            result = future.result(timeout=timeout)
            return _convert_extractions_to_result(result.extractions, funding_statement)
    except concurrent.futures.TimeoutError:
        print(f"Request timed out after {timeout} seconds, returning empty result")
        return ExtractionResult(statement=funding_statement, funders=[])
    except Exception as e:
        raise e


def _convert_extractions_to_result(
    extractions: List[lx.data.Extraction],
    funding_statement: str
) -> ExtractionResult:
    funders_map = {}
    
    for extraction in extractions:
        if extraction.extraction_class == "funder_name":
            funder_name = extraction.extraction_text
            if funder_name not in funders_map:
                funders_map[funder_name] = FunderEntity(
                    funder_name=funder_name,
                    funding_scheme=None,
                    award_ids=[],
                    award_title=None
                )
    
    for extraction in extractions:
        attrs = extraction.attributes or {}
        
        if extraction.extraction_class == "award_ids":
            funder_name = attrs.get("funder_name")
            if funder_name and funder_name in funders_map:
                funders_map[funder_name].add_award_id(extraction.extraction_text)
            elif funder_name:
                funders_map[funder_name] = FunderEntity(
                    funder_name=funder_name,
                    funding_scheme=None,
                    award_ids=[extraction.extraction_text],
                    award_title=None
                )
        
        elif extraction.extraction_class == "funding_scheme":
            funder_name = attrs.get("funder_name")
            if funder_name and funder_name in funders_map:
                funders_map[funder_name].funding_scheme = extraction.extraction_text
        
        elif extraction.extraction_class == "award_title":
            funder_name = attrs.get("funder_name")
            if funder_name and funder_name in funders_map:
                funders_map[funder_name].award_title = extraction.extraction_text
    
    return ExtractionResult(
        statement=funding_statement,
        funders=list(funders_map.values())
    )