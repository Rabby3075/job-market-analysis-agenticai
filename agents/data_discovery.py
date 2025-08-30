"""
Data Discovery Agent
Automatically discovers and downloads datasets from URLs
"""

import logging
import os
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDiscoveryAgent:
    """Agent responsible for discovering datasets from URLs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def discover_datasets(self, url: str) -> List[Dict]:
        """
        Discover available datasets from a given URL
        
        Args:
            url: The URL to scrape for datasets
            
        Returns:
            List of discovered datasets with metadata
        """
        try:
            logger.info(f"Discovering datasets from: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            datasets = []
            
            # Look for common download patterns - ONLY dataset files
            download_patterns = [
                # ABS specific patterns - only actual data files
                r'download.*\.xlsx',
                r'download.*\.csv',
                r'download.*\.xls',
                # General patterns - only dataset file extensions
                r'\.xlsx$',
                r'\.xls$',
                r'\.csv$',
                r'\.json$'
            ]
            
            # Find download links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').lower()
                text = link.get_text().lower()
                
                # Check if this looks like a download link - ONLY dataset files
                is_download = any(re.search(pattern, href) for pattern in download_patterns)
                
                # Only accept links that clearly point to dataset files
                # Reject HTML, ZIP, and other non-dataset files
                is_dataset_file = (
                    href.lower().endswith(('.xlsx', '.xls', '.csv', '.json')) and
                    not href.lower().endswith(('.html', '.htm', '.zip', '.pdf'))
                )
                
                # Check for clear dataset indicators in text
                has_dataset_text = any(word in text.lower() for word in ['xlsx', 'xls', 'csv', 'json'])
                
                if is_dataset_file or (is_download and has_dataset_text):
                    dataset_info = self._extract_dataset_info(link, url)
                    if dataset_info:
                        datasets.append(dataset_info)
            
            # Only look for actual dataset files, not HTML tables
            # (HTML table extraction removed as per user request)
            
            logger.info(f"Discovered {len(datasets)} potential datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"Error discovering datasets from {url}: {str(e)}")
            return []
    
    def _extract_dataset_info(self, link, base_url: str) -> Optional[Dict]:
        """Extract metadata from a download link"""
        try:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # REJECT non-dataset files immediately
            href_lower = href.lower()
            if any(ext in href_lower for ext in ['.html', '.htm', '.zip', '.pdf', '.doc', '.txt']):
                return None
            
            # Only accept dataset file types
            file_type = 'unknown'
            
            # Check URL for dataset file extensions only
            if '.xlsx' in href_lower:
                file_type = 'xlsx'
            elif '.xls' in href_lower:
                file_type = 'xls'
            elif '.csv' in href_lower:
                file_type = 'csv'
            elif '.json' in href_lower:
                file_type = 'json'
            
            # If still unknown, check link text for dataset indicators only
            if file_type == 'unknown':
                text_lower = text.lower()
                if 'excel' in text_lower or 'xlsx' in text_lower or 'xls' in text_lower:
                    file_type = 'xlsx'
                elif 'csv' in text_lower:
                    file_type = 'csv'
                elif 'json' in text_lower:
                    file_type = 'json'
            
            # Reject if we still can't determine it's a dataset
            if file_type == 'unknown':
                return None
            
            # Build full URL
            full_url = urljoin(base_url, href)
            
            # Extract size if available
            size_text = link.find_next_sibling(string=re.compile(r'\d+\.?\d*\s*[KM]B'))
            size = size_text.strip() if size_text else 'unknown'
            
            return {
                'name': text or f'Dataset from {urlparse(base_url).netloc}',
                'type': file_type,
                'url': full_url,
                'description': f'Downloadable {file_type.upper()} file',
                'size': size
            }
            
        except Exception as e:
            logger.error(f"Error extracting dataset info: {str(e)}")
            return None
    
    def download_dataset(self, dataset_info: Dict, output_dir: str = 'data') -> Optional[str]:
        """
        Download a discovered dataset
        
        Args:
            dataset_info: Dataset information from discovery
            output_dir: Directory to save the file
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            url = dataset_info['url']
            file_type = dataset_info['type']
            
            # Only handle actual dataset files, not HTML tables
            
            # Download file
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            # Try to detect file type from Content-Type header if still unknown
            if file_type == 'unknown':
                content_type = response.headers.get('content-type', '').lower()
                if 'excel' in content_type or 'spreadsheet' in content_type:
                    file_type = 'xlsx'
                elif 'csv' in content_type:
                    file_type = 'csv'
                elif 'json' in content_type:
                    file_type = 'json'
                elif 'zip' in content_type:
                    file_type = 'zip'
                elif 'pdf' in content_type:
                    file_type = 'pdf'
                elif 'text/html' in content_type:
                    file_type = 'html'
            
            # Generate filename with detected type
            filename = f"{dataset_info['name'].replace(' ', '_')}.{file_type}"
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)  # Remove invalid chars
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # If file type is still unknown, try to detect from content
            if file_type == 'unknown':
                detected_type = self._detect_file_type_from_content(filepath)
                if detected_type != 'unknown':
                    # Rename file with correct extension
                    new_filepath = filepath.rsplit('.', 1)[0] + '.' + detected_type
                    try:
                        os.rename(filepath, new_filepath)
                        filepath = new_filepath
                        logger.info(f"Renamed file to correct extension: {filepath}")
                    except Exception as e:
                        logger.warning(f"Could not rename file: {str(e)}")
            
            # Check for multiple sheets in Excel files
            if file_type in ['xlsx', 'xls']:
                try:
                    import pandas as pd
                    excel_file = pd.ExcelFile(filepath)
                    sheet_count = len(excel_file.sheet_names)
                    if sheet_count > 1:
                        logger.info(f"Excel file contains {sheet_count} sheets: {excel_file.sheet_names}")
                except Exception as e:
                    logger.warning(f"Could not check Excel sheets: {str(e)}")
            
            logger.info(f"Downloaded dataset to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            return None
    
    def _detect_file_type_from_content(self, filepath: str) -> str:
        """Detect file type by examining file content (magic bytes)"""
        try:
            with open(filepath, 'rb') as f:
                # Read first few bytes to detect file type
                header = f.read(8)
                
                # Excel files (XLSX)
                if header.startswith(b'PK\x03\x04'):
                    return 'xlsx'
                
                # Excel files (XLS)
                if header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                    return 'xls'
                
                # CSV files (check if it's text and has comma-separated values)
                f.seek(0)
                first_line = f.readline().decode('utf-8', errors='ignore')
                if ',' in first_line and len(first_line.split(',')) > 1:
                    return 'csv'
                
                # JSON files
                f.seek(0)
                try:
                    content = f.read().decode('utf-8', errors='ignore')
                    if content.strip().startswith('{') or content.strip().startswith('['):
                        return 'json'
                except:
                    pass
                
                # ZIP files
                if header.startswith(b'PK\x03\x04'):
                    return 'zip'
                
                # PDF files
                if header.startswith(b'%PDF'):
                    return 'pdf'
                
                # HTML files
                f.seek(0)
                try:
                    content = f.read(100).decode('utf-8', errors='ignore').lower()
                    if '<html' in content or '<!doctype' in content:
                        return 'html'
                except:
                    pass
                
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Error detecting file type from content: {str(e)}")
            return 'unknown'
    
    def _extract_html_table(self, dataset_info: Dict, output_dir: str) -> Optional[str]:
        """Extract data from HTML table and save as CSV"""
        try:
            response = self.session.get(dataset_info['url'])
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table (this is simplified - in practice you'd need more logic)
            tables = soup.find_all('table')
            if not tables:
                return None
            
            # Use the first table for now
            table = tables[0]
            
            # Extract headers
            headers = []
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Extract data rows
            rows = []
            for row in table.find_all('tr')[1:]:  # Skip header
                row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                if row_data:
                    rows.append(row_data)
            
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame(rows, columns=headers)
            
            filename = f"{dataset_info['name'].replace(' ', '_')}.csv"
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            filepath = os.path.join(output_dir, filename)
            
            df.to_csv(filepath, index=False)
            logger.info(f"Extracted HTML table to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error extracting HTML table: {str(e)}")
            return None
