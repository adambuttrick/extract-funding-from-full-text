import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from models import (
    FundingStatement,
    FunderEntity,
    ExtractionResult,
    DocumentResult,
    ProcessingResults,
    ProcessingParameters
)
from extraction import extract_funding_statements
from normalization import normalize_funding_statement, is_improperly_formatted
from structured_extraction import extract_structured_entities
from providers import ModelProvider, get_provider_config, validate_provider_requirements
from utils import find_markdown_files, load_checkpoint, save_checkpoint, get_file_hash
from config_loader import load_queries


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract funding information from markdown documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single markdown file
  %(prog)s -i document.md -o results.json

  # Process a directory with Gemini
  %(prog)s -i /path/to/md/files -o results.json --provider gemini --api-key YOUR_KEY

  # Use an Ollama model with normalization
  %(prog)s -i docs/ -o results.json --provider ollama --model llama3.2 --normalize

  # Skip structured extraction (only extract funding statements)
  %(prog)s -i docs/ -o results.json --skip-structured
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input markdown file or directory containing markdown files')
    parser.add_argument('-o', '--output', default='funding_results.json',
                        help='Output JSON file (default: funding_results.json)')
    parser.add_argument(
        '-q', '--queries', help='YAML file containing search queries (uses defaults if not provided)')
    parser.add_argument(
        '--config-dir', help='Custom directory containing configuration files')
    parser.add_argument('--patterns-file',
                        help='YAML file containing funding detection patterns')
    parser.add_argument(
        '--prompt-file', help='Text file containing extraction prompt')
    parser.add_argument('--examples-file',
                        help='JSON file containing extraction examples')

    parser.add_argument('--normalize', action='store_true',
                        help='Normalize funding statements (fix whitespace, accents, etc.)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip semantic extraction (use existing results file)')
    parser.add_argument('--skip-structured', action='store_true',
                        help='Skip structured entity extraction (only extract statements)')

    parser.add_argument('--provider', choices=['gemini', 'ollama', 'openai', 'local_openai'],
                        default='gemini', help='LLM provider for structured extraction (default: gemini)')
    parser.add_argument(
        '--model', help='Model ID to use (defaults to provider default)')
    parser.add_argument(
        '--model-url', help='API endpoint URL (for ollama or local_openai)')
    parser.add_argument('--api-key', help='API key for the model provider')

    parser.add_argument('--colbert-model', default='lightonai/GTE-ModernColBERT-v1',
                        help='ColBERT model for semantic extraction (default: lightonai/GTE-ModernColBERT-v1)')
    parser.add_argument('--threshold', type=float, default=28.0,
                        help='Minimum score threshold for relevance (default: 28.0)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top paragraphs to analyze per query (default: 5)')

    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of documents to process per batch (default: 10)')
    parser.add_argument('--workers', type=int,
                        help='Number of parallel workers (auto-detected if not specified)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in seconds for LLM requests (default: 60)')

    parser.add_argument('--checkpoint-file',
                        help='Checkpoint file for saving progress')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous checkpoint')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of all files (ignore checkpoint)')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--skip-model-validation',
                        action='store_true', help='Skip model validation checks')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output from langextract')

    return parser.parse_args()


def load_existing_results(output_file: str) -> Optional[ProcessingResults]:
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                return ProcessingResults.from_dict(data)
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    return None


def save_results(results: ProcessingResults, output_file: str):
    temp_file = output_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
    os.replace(temp_file, output_file)


def process_markdown_file(
    file_path: str,
    args: argparse.Namespace,
    queries: Dict[str, str]
) -> Optional[DocumentResult]:

    filename = os.path.basename(file_path)

    if args.verbose:
        print(f"Processing: {filename}")

    result = DocumentResult(filename=filename)

    if not args.skip_extraction:
        statements = extract_funding_statements(
            file_path=file_path,
            queries=queries,
            model_name=args.colbert_model,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=32,
            patterns_file=args.patterns_file,
            custom_config_dir=args.config_dir
        )

        if not statements:
            if args.verbose:
                print(f"  No funding statements found in {filename}")
            result.funding_statements = []
            result.structured_entities = []
            return result

        for stmt in statements:
            if args.normalize:
                normalized = normalize_funding_statement(stmt.statement)
                stmt.original = stmt.statement
                stmt.statement = normalized

            if is_improperly_formatted(stmt.statement):
                stmt.is_problematic = True

        result.funding_statements = statements

        if args.verbose:
            print(f"  Found {len(statements)} funding statements")

    if not args.skip_structured and result.funding_statements:
        unique_statements = list(
            set(stmt.statement for stmt in result.funding_statements))

        extraction_results = []
        for statement in unique_statements:
            if args.verbose:
                print(f"  Extracting entities from statement...")

            try:
                extraction_result = extract_structured_entities(
                    funding_statement=statement,
                    provider=ModelProvider(args.provider),
                    model_id=args.model,
                    model_url=args.model_url,
                    api_key=args.api_key,
                    skip_model_validation=args.skip_model_validation,
                    timeout=args.timeout,
                    debug=args.debug if hasattr(args, 'debug') else False,
                    prompt_file=args.prompt_file,
                    examples_file=args.examples_file,
                    custom_config_dir=args.config_dir
                )
                extraction_results.append(extraction_result)
            except Exception as e:
                print(f"  Warning: Entity extraction failed: {e}")

        result.extraction_results = extraction_results

        if args.verbose:
            total_funders = sum(len(r.funders) for r in extraction_results)
            print(f"  Extracted {total_funders} funders from {len(extraction_results)} statements")
    else:
        if not hasattr(result, 'extraction_results') or result.extraction_results is None:
            result.extraction_results = []

    return result


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_file = args.output
    checkpoint_file = args.checkpoint_file or f"{output_file}.checkpoint"

    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)

    queries = load_queries(args.queries, args.config_dir)

    if not args.skip_structured:
        provider = ModelProvider(args.provider)
        config = get_provider_config(provider)

        if not args.model:
            args.model = config.default_model
        if not args.model_url:
            args.model_url = config.default_url

        if not args.api_key:
            if provider == ModelProvider.OPENAI:
                args.api_key = os.environ.get('OPENAI_API_KEY')
            elif provider == ModelProvider.GEMINI:
                args.api_key = os.environ.get('GEMINI_API_KEY')

        try:
            validate_provider_requirements(
                provider=provider,
                api_key=args.api_key,
                model_url=args.model_url,
                model_id=args.model,
                skip_model_validation=args.skip_model_validation
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    if input_path.is_file():
        md_files = [str(input_path)]
    else:
        md_files = find_markdown_files(str(input_path))

    if not md_files:
        print("No markdown files found")
        sys.exit(0)

    print(f"Found {len(md_files)} markdown files to process")

    checkpoint_data = {}
    if args.resume and not args.force:
        checkpoint_data = load_checkpoint(checkpoint_file)
        print(f"Resuming from checkpoint: {len(checkpoint_data.get('processed_files', {}))} files already processed")

    files_to_process = []
    for file_path in md_files:
        file_hash = get_file_hash(file_path)
        if args.force or file_hash not in checkpoint_data.get('processed_files', {}):
            files_to_process.append(file_path)

    if not files_to_process:
        print("All files already processed")
        return

    print(f"Processing {len(files_to_process)} files...")

    results = load_existing_results(output_file) or ProcessingResults(
        timestamp=datetime.now().isoformat(),
        parameters=ProcessingParameters(
            input_path=str(input_path),
            normalize=args.normalize,
            provider=args.provider if not args.skip_structured else None,
            model=args.model if not args.skip_structured else None,
            threshold=args.threshold,
            top_k=args.top_k
        ),
        results={},
        summary={}
    )

    batch_results = []
    processed_count = 0

    try:
        for i, file_path in enumerate(files_to_process):
            doc_result = process_markdown_file(file_path, args, queries)

            if doc_result:
                batch_results.append(doc_result)

            file_hash = get_file_hash(file_path)
            if 'processed_files' not in checkpoint_data:
                checkpoint_data['processed_files'] = {}
            checkpoint_data['processed_files'][file_hash] = {
                'path': file_path,
                'processed_at': datetime.now().isoformat(),
                'found_funding': doc_result is not None and len(doc_result.funding_statements) > 0
            }
            processed_count += 1

            if len(batch_results) >= args.batch_size or i == len(files_to_process) - 1:
                for doc in batch_results:
                    results.results[doc.filename] = doc

                results.update_summary()

                save_results(results, output_file)
                save_checkpoint(checkpoint_file, checkpoint_data)

                print(f"Processed {processed_count}/{len(files_to_process)} files, saved checkpoint")
                batch_results = []

    except KeyboardInterrupt:
        print("\nInterrupted! Saving progress...")
        for doc in batch_results:
            results.results[doc.filename] = doc
        results.update_summary()
        save_results(results, output_file)
        save_checkpoint(checkpoint_file, checkpoint_data)
        print("Progress saved. Use --resume to continue.")
        sys.exit(1)

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total files processed: {results.summary.get('total_files', 0)}")
    print(f"Files with funding: {results.summary.get('files_with_funding', 0)}")
    print(f"Total statements: {results.summary.get('total_statements', 0)}")
    if not args.skip_structured:
        print(f"Total funders: {results.summary.get('total_funders', 0)}")
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
