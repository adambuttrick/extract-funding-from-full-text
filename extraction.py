"""
Semantic extraction of funding statements using ColBERT.
"""

import re
import os
from typing import List, Dict, Optional

import torch
from pylate import models, rank

from models import FundingStatement
from config_loader import load_funding_patterns


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def is_likely_funding_statement(
    paragraph: str, 
    score: float, 
    threshold: float = 28.0,
    patterns_file: Optional[str] = None,
    custom_config_dir: Optional[str] = None
) -> bool:
    """Check if a paragraph is likely a funding statement based on patterns and score."""
    if score < threshold:
        return False
    
    funding_patterns = load_funding_patterns(patterns_file, custom_config_dir)
    
    paragraph_lower = paragraph.lower()
    return any(re.search(pattern, paragraph_lower) for pattern in funding_patterns)


def extract_funding_sentences(paragraph: str) -> List[str]:
    """Extract sentences that contain funding information from a paragraph."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
    funding_sentences = []
    
    for i, sentence in enumerate(sentences):
        if re.search(r'\b(?:acknowledg|fund|support|grant|award|project)\w*\b', sentence, re.IGNORECASE):
            sentence = sentence.strip()
            
            # Handle cases where grant numbers start on next line
            if i + 1 < len(sentences) and sentence.endswith('No'):
                sentence = sentence + ' ' + sentences[i + 1].strip()
            
            funding_sentences.append(sentence)
    
    return funding_sentences


def should_extract_full_paragraph(paragraph: str, score: float) -> bool:
    """
    Determine if the full paragraph should be extracted based on smart heuristics.
    
    Args:
        paragraph: The paragraph text
        score: The semantic similarity score from ColBERT
    
    Returns:
        True if full paragraph should be extracted, False otherwise
    """
    # High semantic score indicates the entire paragraph is relevant
    if score > 25.0:
        return True
    
    # Short paragraphs are likely cohesive units
    if len(paragraph) < 500:
        return True
    
    # Calculate funding keyword density
    funding_keywords = [
        'fund', 'grant', 'support', 'acknowledg', 'sponsor', 
        'award', 'financial', 'foundation', 'scholarship', 'fellowship'
    ]
    
    words = paragraph.lower().split()
    if not words:
        return False
        
    keyword_count = sum(1 for word in words 
                       if any(kw in word for kw in funding_keywords))
    density = keyword_count / len(words)
    
    # High density of funding-related terms
    if density > 0.05:  # More than 5% funding-related words
        return True
    
    # Check for clustering of funding keywords
    # If multiple funding terms appear within a 50-word window, likely cohesive
    text_lower = paragraph.lower()
    keyword_positions = []
    for kw in funding_keywords:
        start = 0
        while True:
            pos = text_lower.find(kw, start)
            if pos == -1:
                break
            keyword_positions.append(pos)
            start = pos + 1
    
    if len(keyword_positions) >= 3:
        keyword_positions.sort()
        # Check if keywords are clustered (within 300 characters of each other)
        for i in range(len(keyword_positions) - 2):
            if keyword_positions[i + 2] - keyword_positions[i] < 300:
                return True
    
    return False


def extract_funding_from_long_paragraph(paragraph: str) -> str:
    """
    Extract funding-relevant portion from a long paragraph using sliding window.
    
    Args:
        paragraph: The long paragraph text
    
    Returns:
        The extracted funding-relevant portion
    """
    funding_keywords = [
        'fund', 'grant', 'support', 'acknowledg', 'sponsor',
        'award', 'financial', 'foundation', 'scholarship', 'fellowship'
    ]
    
    text_lower = paragraph.lower()
    
    # Find all positions of funding keywords
    keyword_positions = []
    for kw in funding_keywords:
        start = 0
        while True:
            pos = text_lower.find(kw, start)
            if pos == -1:
                break
            keyword_positions.append(pos)
            start = pos + 1
    
    if not keyword_positions:
        return paragraph  # Fallback to full paragraph
    
    keyword_positions.sort()
    
    # Find the span that covers all keywords with minimal extra content
    start_pos = max(0, keyword_positions[0] - 100)  # Add some context before
    end_pos = min(len(paragraph), keyword_positions[-1] + 200)  # Add context after
    
    # Expand to sentence boundaries if possible
    # Look for sentence start before start_pos
    for i in range(start_pos, max(0, start_pos - 50), -1):
        if i == 0 or (i > 0 and paragraph[i-1] in '.!?\n'):
            start_pos = i
            break
    
    # Look for sentence end after end_pos
    for i in range(end_pos, min(len(paragraph), end_pos + 100)):
        if paragraph[i] in '.!?':
            end_pos = i + 1
            break
    
    return paragraph[start_pos:end_pos].strip()


def extract_funding_statements(
    file_path: str,
    queries: Dict[str, str],
    model_name: str = 'lightonai/GTE-ModernColBERT-v1',
    top_k: int = 5,
    threshold: float = 28.0,
    batch_size: int = 32,
    patterns_file: Optional[str] = None,
    custom_config_dir: Optional[str] = None
) -> List[FundingStatement]:
    """
    Extract funding statements from a markdown file using semantic search.
    
    Args:
        file_path: Path to the markdown file
        queries: Dictionary of query names and texts
        model_name: ColBERT model to use
        top_k: Number of top paragraphs to analyze per query
        threshold: Minimum score threshold
        batch_size: Batch size for encoding
    
    Returns:
        List of extracted funding statements
    """
    # Set environment to avoid warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Load the ColBERT model
    model = models.ColBERT(model_name_or_path=model_name)
    
    # Read and split document into paragraphs
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    paragraphs = split_into_paragraphs(content)
    if not paragraphs:
        return []
    
    # Encode all paragraphs
    documents_embeddings = model.encode(
        paragraphs,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False
    )
    
    # Track unique statements to avoid duplicates
    seen_statements = set()
    funding_statements = []
    
    # Process each query
    for query_name, query_text in queries.items():
        # Encode query
        query_embeddings = model.encode(
            [query_text],
            batch_size=1,
            is_query=True,
            show_progress_bar=False
        )
        
        # Rank documents
        doc_ids = list(range(len(paragraphs)))
        reranked = rank.rerank(
            documents_ids=[doc_ids],
            queries_embeddings=query_embeddings,
            documents_embeddings=[documents_embeddings]
        )
        
        if reranked and len(reranked) > 0:
            top_results = reranked[0][:top_k]
            
            for result in top_results:
                para_id = result['id']
                score = float(result['score'])
                paragraph = paragraphs[para_id]
                
                # Check if this is likely a funding statement
                if is_likely_funding_statement(paragraph, score, threshold, patterns_file, custom_config_dir):
                    # Determine extraction strategy based on smart heuristics
                    if should_extract_full_paragraph(paragraph, score):
                        # Extract the full paragraph
                        statement_text = paragraph.strip()
                    elif len(paragraph) > 1000:
                        # For very long paragraphs, use sliding window extraction
                        statement_text = extract_funding_from_long_paragraph(paragraph)
                    else:
                        # Fall back to sentence extraction for medium-length, low-density paragraphs
                        funding_sentences = extract_funding_sentences(paragraph)
                        # Join sentences if multiple found (preserves context)
                        if funding_sentences:
                            statement_text = ' '.join(funding_sentences)
                        else:
                            continue
                    
                    # Add the statement if it's not a duplicate and has meaningful content
                    if statement_text not in seen_statements and len(statement_text) > 20:
                        seen_statements.add(statement_text)
                        
                        stmt = FundingStatement(
                            statement=statement_text,
                            score=score,
                            query=query_name,
                            paragraph_idx=para_id
                        )
                        funding_statements.append(stmt)
    
    return funding_statements