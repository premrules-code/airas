"""
Extract metadata from SEC filings for filtering.
"""

import re
import logging
from collections import defaultdict
from typing import Any, ClassVar, Dict, List, Optional, Sequence
from pathlib import Path

from pydantic import PrivateAttr
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)


class SECMetadataExtractor:
    """
    Extract structured metadata from SEC 10-K/10-Q filings.
    
    Metadata enables precise filtering:
    - Document type, ticker, filing date
    - Section (Income Statement, Balance Sheet, etc)
    - Metrics mentioned (revenue, assets, debt)
    - Fiscal period
    """
    
    def __init__(self):
        # Financial statement sections
        self.sections = {
            'income_statement': [
                'consolidated statements of operations',
                'consolidated statements of income',
                'income statement',
                'statement of operations',
                'statement of earnings'
            ],
            'balance_sheet': [
                'consolidated balance sheets',
                'balance sheet',
                'statement of financial position'
            ],
            'cash_flow': [
                'consolidated statements of cash flows',
                'cash flow statement',
                'statement of cash flows'
            ],
            'shareholders_equity': [
                'consolidated statements of shareholders',
                'statement of stockholders equity',
                'shareholders equity'
            ]
        }
        
        # Financial metrics keywords
        self.metrics = {
            'revenue': ['revenue', 'net sales', 'total revenue', 'sales'],
            'income': ['net income', 'earnings', 'profit', 'net earnings'],
            'assets': ['total assets', 'current assets', 'non-current assets'],
            'liabilities': ['total liabilities', 'current liabilities'],
            'equity': ['shareholders equity', 'stockholders equity', 'total equity'],
            'cash_flow': ['operating cash flow', 'free cash flow', 'cash generated'],
            'debt': ['total debt', 'long-term debt', 'term debt'],
            'margin': ['gross margin', 'operating margin', 'net margin', 'profit margin']
        }
    
    def extract_from_filename(self, filename: str) -> Dict:
        """
        Extract metadata from SEC filing filename.
        
        Expected format: TICKER_TYPE_DATE.txt
        Example: AAPL_10K_2023-11-03.txt
        
        Returns:
            Base metadata dict
        """
        
        parts = Path(filename).stem.split('_')
        
        metadata = {
            'source_file': filename,
            'ticker': None,
            'document_type': None,
            'filing_date': None
        }
        
        if len(parts) >= 1:
            metadata['ticker'] = parts[0].upper()
        
        if len(parts) >= 2:
            doc_type = parts[1].upper().replace('-', '')
            if 'K' in doc_type:
                metadata['document_type'] = '10-K'
            elif 'Q' in doc_type:
                metadata['document_type'] = '10-Q'
            else:
                metadata['document_type'] = doc_type
        
        if len(parts) >= 3:
            metadata['filing_date'] = parts[2]
        
        return metadata
    
    def extract_from_content(self, content: str, base_metadata: Dict) -> Dict:
        """
        Extract content-specific metadata.
        
        Args:
            content: Text chunk
            base_metadata: Metadata from filename
            
        Returns:
            Enhanced metadata with section, metrics, period
        """
        
        content_lower = content.lower()
        
        # Detect section
        section = self._detect_section(content_lower)
        
        # Detect metrics
        metrics = self._detect_metrics(content_lower)
        
        # Detect fiscal period
        fiscal_period = self._detect_fiscal_period(content)
        
        # Check for numbers
        has_numbers = bool(re.search(
            r'\$?\d{1,3}(,\d{3})*(\.\d+)?\s*(million|billion|thousand)?',
            content
        ))
        
        content_type = 'financial_data' if (has_numbers and metrics) else 'narrative'
        
        # Combine all metadata
        enhanced = {
            **base_metadata,
            'section': section,
            'metric_types': metrics,
            'fiscal_period': fiscal_period,
            'content_type': content_type,
            'has_numbers': has_numbers,
            'word_count': len(content.split())
        }
        
        return enhanced
    
    def _detect_section(self, content: str) -> Optional[str]:
        """Detect financial statement section."""
        
        for section_name, keywords in self.sections.items():
            for keyword in keywords:
                if keyword in content:
                    return section_name
        return None
    
    def _detect_metrics(self, content: str) -> List[str]:
        """Detect financial metrics mentioned."""
        
        detected = []
        for metric_name, keywords in self.metrics.items():
            for keyword in keywords:
                if keyword in content:
                    detected.append(metric_name)
                    break
        return detected
    
    def _detect_fiscal_period(self, content: str) -> Optional[str]:
        """Detect fiscal period mentioned."""
        
        patterns = [
            r'fiscal\s+year\s+(\d{4})',
            r'fiscal\s+(\d{4})',
            r'FY\s*(\d{4})',
            r'Q([1-4])\s+(\d{4})',
            r'year\s+ended.*?(\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None


class SECMetadataTransform(BaseExtractor):
    """
    LlamaIndex IngestionPipeline transform that enriches nodes with SEC metadata.

    Wraps SECMetadataExtractor to work as a pipeline transformation step.
    Each node gets: section, metric_types, fiscal_period, content_type,
    has_numbers, word_count, chunk_index, total_chunks.
    """

    _extractor: SECMetadataExtractor = PrivateAttr(default_factory=SECMetadataExtractor)

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        # Group nodes by source_file to compute correct total_chunks per document
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, node in enumerate(nodes):
            source = node.metadata.get("source_file", "unknown")
            groups[source].append(i)

        metadata_list: List[Dict] = [{} for _ in nodes]

        for _source_file, indices in groups.items():
            total = len(indices)
            base_meta = {k: v for k, v in nodes[indices[0]].metadata.items()
                         if k in ("source_file", "ticker", "document_type", "filing_date")}

            for chunk_idx, node_idx in enumerate(indices):
                node = nodes[node_idx]
                content_meta = self._extractor.extract_from_content(node.text, base_meta)

                # Remove keys already in base metadata to avoid duplication
                for key in ("source_file", "ticker", "document_type", "filing_date"):
                    content_meta.pop(key, None)

                content_meta["chunk_index"] = chunk_idx
                content_meta["total_chunks"] = total

                metadata_list[node_idx] = content_meta

        return metadata_list