from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class FundingStatement(BaseModel):
    statement: str = Field(description="The funding statement text (possibly normalized)")
    original: Optional[str] = Field(default=None, description="Original text before normalization")
    score: float = Field(description="Relevance score from semantic search")
    query: str = Field(description="Query that matched this statement")
    paragraph_idx: Optional[int] = Field(default=None, description="Index of source paragraph")
    is_problematic: bool = Field(default=False, description="Whether statement has formatting issues")


class FunderEntity(BaseModel):
    funder_name: str = Field(description="Name of the funding organization")
    funding_scheme: Optional[str] = Field(default=None, description="Specific funding program or scheme")
    award_ids: List[str] = Field(default_factory=list, description="List of grant/award identifiers")
    award_title: Optional[str] = Field(default=None, description="Title of the award if provided")
    
    def add_award_id(self, award_id: str) -> None:
        if award_id not in self.award_ids:
            self.award_ids.append(award_id)


class ExtractionResult(BaseModel):
    statement: str = Field(description="The original funding statement text")
    funders: List[FunderEntity] = Field(default_factory=list, description="List of funders extracted")


class DocumentResult(BaseModel):
    filename: str = Field(description="Name of the markdown file")
    funding_statements: List[FundingStatement] = Field(
        default_factory=list,
        description="Extracted funding statements"
    )
    extraction_results: List[ExtractionResult] = Field(
        default_factory=list,
        description="Structured extraction results with funders"
    )
    
    def has_funding(self) -> bool:
        return len(self.funding_statements) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'funding_statements': [
                {
                    'statement': stmt.statement,
                    'original': stmt.original,
                    'score': stmt.score,
                    'query': stmt.query,
                    'is_problematic': stmt.is_problematic
                }
                for stmt in self.funding_statements
            ],
            'extractions': [
                {
                    'statement': result.statement,
                    'funders': [
                        {
                            'funder_name': funder.funder_name,
                            'funding_scheme': funder.funding_scheme,
                            'award_ids': funder.award_ids,
                            'award_title': funder.award_title
                        }
                        for funder in result.funders
                    ]
                }
                for result in self.extraction_results
            ]
        }


class ProcessingParameters(BaseModel):
    input_path: str
    normalize: bool = False
    provider: Optional[str] = None
    model: Optional[str] = None
    threshold: float = 28.0
    top_k: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_path': self.input_path,
            'normalize': self.normalize,
            'provider': self.provider,
            'model': self.model,
            'threshold': self.threshold,
            'top_k': self.top_k
        }


class ProcessingResults(BaseModel):
    timestamp: str
    parameters: ProcessingParameters
    results: Dict[str, DocumentResult]
    summary: Dict[str, Any]
    
    def update_summary(self):
        total_files = len(self.results)
        files_with_funding = sum(1 for doc in self.results.values() if doc.has_funding())
        total_statements = sum(len(doc.funding_statements) for doc in self.results.values())
        total_funders = sum(
            len(result.funders) 
            for doc in self.results.values() 
            for result in doc.extraction_results
        )
        
        self.summary = {
            'total_files': total_files,
            'files_with_funding': files_with_funding,
            'total_statements': total_statements,
            'total_funders': total_funders
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'parameters': self.parameters.to_dict(),
            'results': {
                filename: doc.to_dict()
                for filename, doc in self.results.items()
            },
            'summary': self.summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingResults':
        parameters = ProcessingParameters(**data['parameters'])
        results = {}
        
        for filename, doc_data in data.get('results', {}).items():
            doc = DocumentResult(filename=filename)
            for stmt_data in doc_data.get('funding_statements', []):
                stmt = FundingStatement(
                    statement=stmt_data['statement'],
                    original=stmt_data.get('original'),
                    score=stmt_data.get('score', 0.0),
                    query=stmt_data.get('query', ''),
                    is_problematic=stmt_data.get('is_problematic', False)
                )
                doc.funding_statements.append(stmt)

            for extraction_data in doc_data.get('extractions', []):
                funders = []
                for funder_data in extraction_data.get('funders', []):
                    funder = FunderEntity(
                        funder_name=funder_data['funder_name'],
                        funding_scheme=funder_data.get('funding_scheme'),
                        award_ids=funder_data.get('award_ids', []),
                        award_title=funder_data.get('award_title')
                    )
                    funders.append(funder)
                
                result = ExtractionResult(
                    statement=extraction_data.get('statement', extraction_data.get('text', '')),
                    funders=funders
                )
                doc.extraction_results.append(result)
            
            results[filename] = doc
        
        return cls(
            timestamp=data['timestamp'],
            parameters=parameters,
            results=results,
            summary=data.get('summary', {})
        )


class ProcessingStats(BaseModel):
    total_documents: int = Field(description="Total documents processed")
    successful: int = Field(description="Successfully processed documents") 
    failed: int = Field(description="Failed documents")
    total_entities: int = Field(description="Total entities extracted")