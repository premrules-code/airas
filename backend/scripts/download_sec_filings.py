#!/usr/bin/env python3
"""Download SEC filings from EDGAR."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import shutil
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup

from config.settings import get_settings
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


class SECDownloader:
    """Download and clean SEC filings."""

    def __init__(self, output_dir: Path = None):
        self.settings = get_settings()
        self.output_dir = output_dir or self.settings.raw_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = Downloader(
            company_name="AIRAS",
            email_address=self.settings.sec_user_email,
            download_folder=str(self.output_dir / "temp_downloads")
        )

    def download_filings(self, ticker: str, filing_types: list = None, num_filings: int = 3):
        """Download filings for ticker.

        Args:
            ticker: Company ticker symbol (e.g. AAPL)
            filing_types: List of filing types (default: ["10-K"])
            num_filings: Number of filings per type to download
        """
        if filing_types is None:
            filing_types = ["10-K"]

        logger.info(f"Downloading {filing_types} for {ticker}")

        for filing_type in filing_types:
            try:
                self.downloader.get(filing_type, ticker, limit=num_filings)
                self._process_downloads(ticker, filing_type)
            except Exception as e:
                logger.error(f"Error downloading {filing_type} for {ticker}: {e}")

        # Cleanup temp directory
        temp_dir = self.output_dir / "temp_downloads"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def _process_downloads(self, ticker: str, filing_type: str):
        """Process downloaded filing files into clean text."""
        temp_dir = (
            self.output_dir
            / "temp_downloads"
            / "sec-edgar-filings"
            / ticker
            / filing_type
        )

        if not temp_dir.exists():
            logger.warning(f"No downloads found at {temp_dir}")
            return

        for folder in sorted(temp_dir.iterdir()):
            if not folder.is_dir():
                continue

            submission_file = folder / "full-submission.txt"
            if not submission_file.exists():
                continue

            with open(submission_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()

            # Extract filing date from SEC header
            date = self._extract_filing_date(raw_content, folder.name)

            # Extract the primary document HTML from the SGML wrapper
            clean_text = self._extract_and_clean(raw_content)

            if len(clean_text) < 1000:
                logger.warning(f"Skipping {folder.name} - too short ({len(clean_text)} chars)")
                continue

            output_filename = f"{ticker}_{filing_type.replace('-', '')}_{date}.txt"
            output_path = self.output_dir / output_filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)

            size_kb = output_path.stat().st_size / 1024
            logger.info(f"   Saved: {output_filename} ({size_kb:.1f} KB)")

    def _extract_filing_date(self, content: str, accession_number: str) -> str:
        """Extract filing date from SEC submission header."""
        import re
        match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', content)
        if match:
            d = match.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        match = re.search(r'FILED AS OF DATE:\s*(\d{8})', content)
        if match:
            d = match.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return accession_number

    def _extract_and_clean(self, raw_content: str) -> str:
        """Extract the primary document from SGML and clean HTML to text."""
        import re
        # Find the first <DOCUMENT> block with the filing type (e.g. 10-K)
        doc_pattern = re.compile(
            r'<DOCUMENT>\s*<TYPE>10-K.*?<TEXT>(.*?)</TEXT>',
            re.DOTALL | re.IGNORECASE
        )
        match = doc_pattern.search(raw_content)
        if match:
            html_content = match.group(1)
        else:
            # Fallback: use everything after the header
            html_content = raw_content

        return self._clean_html(html_content)

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML to plain text."""
        soup = BeautifulSoup(html_content, 'lxml')

        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        text = soup.get_text()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Download SEC filings from EDGAR")
    parser.add_argument("--ticker", help="Single ticker (e.g. AAPL)")
    parser.add_argument("--tickers", nargs="+", help="Multiple tickers (e.g. AAPL MSFT GOOGL)")
    parser.add_argument("--type", default="10-K", help="Filing type: 10-K, 10-Q, 8-K (default: 10-K)")
    parser.add_argument("--amount", type=int, default=3, help="Number of filings per ticker (default: 3)")

    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    if not args.ticker and not args.tickers:
        parser.error("Provide --ticker or --tickers")

    tickers = [args.ticker] if args.ticker else args.tickers

    downloader = SECDownloader()

    print(f"\nDownloading {args.type} filings for: {', '.join(tickers)}")
    print(f"Amount per ticker: {args.amount}\n")

    for ticker in tickers:
        downloader.download_filings(ticker.upper(), [args.type], args.amount)

    print("\nDone. Files saved to:", downloader.output_dir)


if __name__ == "__main__":
    main()
