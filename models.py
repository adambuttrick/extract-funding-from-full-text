from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class Grant(BaseModel):
    grant_id: str = Field(description="Grant or award identifier")


class FundingStatement(BaseModel):
    statement: str = Field(description="The funding statement text (possibly normalized)")
    original: Optional[str] = Field(default=None, description="Original text before normalization")
    score: float = Field(description="Relevance score from semantic search")
    query: str = Field(description="Query that matched this statement")
    paragraph_idx: Optional[int] = Field(default=None, description="Index of source paragraph")
    is_problematic: bool = Field(default=False, description="Whether statement has formatting issues")


class StructuredEntity(BaseModel):
    funder: str = Field(description="Name of the funding organization")
    grants: List[Grant] = Field(default_factory=list, description="List of grants from this funder")
    extraction_texts: List[str] = Field(
        default_factory=list, 
        description="Original texts where entity was found"
    )
    
    def add_grant(self, grant_id: str) -> None:
        for grant in self.grants:
            if grant.grant_id == grant_id:
                return
        self.grants.append(Grant(grant_id=grant_id))


class DocumentResult(BaseModel):
    filename: str = Field(description="Name of the markdown file")
    funding_statements: List[FundingStatement] = Field(
        default_factory=list,
        description="Extracted funding statements"
    )
    structured_entities: List[StructuredEntity] = Field(
        default_factory=list,
        description="Structured entities extracted from statements"
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
            'structured_entities': [
                {
                    'funder': entity.funder,
                    'grants': [g.grant_id for g in entity.grants],
                    'extraction_texts': entity.extraction_texts
                }
                for entity in self.structured_entities
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
        total_entities = sum(len(doc.structured_entities) for doc in self.results.values())
        
        self.summary = {
            'total_files': total_files,
            'files_with_funding': files_with_funding,
            'total_statements': total_statements,
            'total_entities': total_entities
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

            for entity_data in doc_data.get('structured_entities', []):
                entity = StructuredEntity(
                    funder=entity_data['funder'],
                    extraction_texts=entity_data.get('extraction_texts', [])
                )
                for grant_id in entity_data.get('grants', []):
                    entity.add_grant(grant_id)
                doc.structured_entities.append(entity)
            
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