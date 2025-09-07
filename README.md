# Extract Funding Metadata from Full-Text

Python utility for extracting and structuring funding information from markdown documents using semantic search/late interaction models and LLM-based entity extraction. Built with [pylate](https://github.com/lightonai/pylate) and [langextract](https://github.com/google/langextract).


## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Process a single markdown file with default settings
python funding_extractor.py -i document.md -o results.json

# Process a directory of markdown files
python funding_extractor.py -i /path/to/documents -o results.json

# Use a specific LLM provider
python funding_extractor.py -i docs/ -o results.json --provider gemini --api-key YOUR_KEY
```

## Configuration

The tool uses configuration files located in the `configs/` directory for its extraction:

### Query Configuration
`configs/queries/default.yaml` - Defines the semantic search queries for finding funding statements

### Pattern Configuration  
`configs/patterns/funding_patterns.yaml` - Regular expressions for identifying the sections of text with funding details

### Extraction Configuration
- `configs/prompts/extraction_prompt.txt` - Prompt for funding extraction
- `configs/prompts/extraction_examples.json` - Examples for few-shot learning

Custom queries and prompts can be configured via the arguments.

## Usage Examples

### Basic Extraction
```bash
python funding_extractor.py -i paper.md -o funding.json
```

### With Text Normalization
```bash
python funding_extractor.py -i docs/ -o results.json --normalize
```

### Using Ollama
```bash
python funding_extractor.py -i docs/ -o results.json \
  --provider ollama --model llama3.2 \
  --model-url http://localhost:11434
```

### Skip Structured Extraction
```bash
python funding_extractor.py -i docs/ -o results.json --skip-structured
```

## Command-Line Options

### Required Arguments
- `-i, --input` - Input markdown file or directory
- `-o, --output` - Output JSON file (default: funding_results.json)

### Configuration Options
- `-q, --queries` - Custom queries YAML file
- `--config-dir` - Custom configuration directory
- `--patterns-file` - Custom funding patterns YAML
- `--prompt-file` - Custom extraction prompt
- `--examples-file` - Custom extraction examples

### Processing Options
- `--normalize` - Enable text normalization
- `--skip-extraction` - Skip funding statement extraction
- `--skip-structured` - Skip structured funding entity extraction
- `--batch-size` - Documents per batch (default: 10)
- `--timeout` - LLM request timeout in seconds (default: 60)

### LLM Provider Options
- `--provider` - LLM provider: gemini, ollama, openai, local_openai
- `--model` - Model ID to use
- `--model-url` - API endpoint URL
- `--api-key` - API key for the provider

### ColBERT Options
- `--colbert-model` - ColBERT model (default: [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1))
- `--threshold` - Minimum relevance score (default: 28.0)
- `--top-k` - Top paragraphs per query (default: 5)

### Checkpoint Options
- `--checkpoint-file` - Custom checkpoint file path
- `--resume` - Resume from previous checkpoint
- `--force` - Force reprocessing all files

### Other Options
- `-v, --verbose` - Enable verbose output
- `--skip-model-validation` - Skip model validation
- `--debug` - Enable debug output

## Output Format

The tool generates a JSON file with the following structure:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "parameters": {
    "input_path": "/path/to/documents",
    "normalize": true,
    "provider": "gemini",
    "model": "gemini-2.5-flash-lite",
    "threshold": 28.0,
    "top_k": 5
  },
  "results": {
    "document.md": {
      "funding_statements": [
        {
          "statement": "This work was supported by NSF grant 12345 and NIH grant 67890.",
          "original": "Original text before normalization",
          "score": 35.2,
          "query": "funding_statement",
          "is_problematic": false
        }
      ],
      "extractions": [
        {
          "statement": "This work was supported by NSF grant 12345 and NIH grant 67890.",
          "funders": [
            {
              "funder_name": "NSF",
              "funding_scheme": null,
              "award_ids": ["12345"],
              "award_title": null
            },
            {
              "funder_name": "NIH",
              "funding_scheme": null,
              "award_ids": ["67890"],
              "award_title": null
            }
          ]
        }
      ]
    }
  },
  "summary": {
    "total_files": 10,
    "files_with_funding": 8,
    "total_statements": 25,
    "total_funders": 45
  }
}
```

## Provider Setup

### Gemini
```bash
export GEMINI_API_KEY=your_api_key
python funding_extractor.py -i docs/ -o results.json --provider gemini
```

### OpenAI
```bash
export OPENAI_API_KEY=your_api_key
python funding_extractor.py -i docs/ -o results.json --provider openai --model gpt-4o-mini
```

### Ollama
```bash
# Start Ollama server
ollama serve

# Pull a model
ollama pull llama3.2

# Run extraction
python funding_extractor.py -i docs/ -o results.json --provider ollama --model llama3.2
```

### Local OpenAI-Compatible Server
```bash
python funding_extractor.py -i docs/ -o results.json \
  --provider local_openai \
  --model-url http://localhost:8000 \
  --model your-model-name
```

## Advanced Features

### Checkpoint and Resume
Progress is automatically save for large batch operations:
- Checkpoints are saved after each batch
- Use `--resume` to continue from last checkpoint
- Use `--force` to reprocess all files

### Text Normalization
The `--normalize` flag attemtps to normalize the extracted funding statement by:
- Correcting malformed accents and special characters
- Removing line numbers
- Cleans whitespace issues
- Detecting and flagging funding statements that appear to contain errant content (tables, etc.)

## Notes on Input and Markdown Conversion

This tool processes markdown files as input. When converting PDF documents to markdown, the quality and structure of the output can vary significantly depending on the conversion library used and the source document format. Some options:

### Markdown Conversion Libraries

- [pdfplumber](https://github.com/jsvine/pdfplumber) - Good for basic text extract from PDFs with decent support for complex layouts and tables.
  ```python
  import pdfplumber
  with pdfplumber.open("document.pdf") as pdf:
      text = '\n'.join(page.extract_text() for page in pdf.pages)
  ```

- [Docling](https://github.com/DS4SD/docling) - Advanced document conversion with robust support for complex layouts, tables, and figures. Works great for scientific papers.
  ```python
  from docling.document_converter import DocumentConverter
  converter = DocumentConverter()
  result = converter.convert("document.pdf")  # supports local path or URL
  markdown = result.document.export_to_markdown()
  ```

- [DOTS OCR](https://github.com/rednote-hilab/dots.ocr) - Vision-language model based conversion using multimodal LLMs. Exciting new approach for challenging documents!
  ```python
  from transformers import AutoModelForCausalLM, AutoProcessor
  model = AutoModelForCausalLM.from_pretrained("DotsOCR", trust_remote_code=True)
  processor = AutoProcessor.from_pretrained("DotsOCR", trust_remote_code=True)
  # Process images with model to extract text/layout
  ```



