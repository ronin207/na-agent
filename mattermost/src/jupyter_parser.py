import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class JupyterParser:
    """Parser for Jupyter notebook (.ipynb) files."""
    
    def __init__(self, include_outputs: bool = True, include_raw_code: bool = False):
        """
        Initialize the Jupyter notebook parser.
        
        Args:
            include_outputs: Whether to include cell outputs in the parsed content
            include_raw_code: Whether to include raw code cells in the parsed content
        """
        self.include_outputs = include_outputs
        self.include_raw_code = include_raw_code
    
    def parse(self, file_path: str) -> str:
        """
        Parse a Jupyter notebook file and return its content as text.
        
        Args:
            file_path: Path to the .ipynb file
            
        Returns:
            Extracted text content from the notebook
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            content_parts = []
            
            # Extract notebook metadata if available
            if 'metadata' in notebook:
                title = notebook['metadata'].get('title', '')
                if title:
                    content_parts.append(f"# {title}\n")
            
            # Process each cell
            for cell in notebook.get('cells', []):
                cell_type = cell.get('cell_type', '')
                source = cell.get('source', [])
                
                # Handle different types of cells
                if isinstance(source, list):
                    source = ''.join(source)
                
                if cell_type == 'markdown':
                    # Preserve LaTeX blocks
                    processed_source = self._preserve_latex(source)
                    content_parts.append(processed_source)
                    content_parts.append('\n\n')
                
                elif cell_type == 'code':
                    if self.include_raw_code:
                        content_parts.append('```python\n')
                        content_parts.append(source)
                        content_parts.append('\n```\n\n')
                    
                    if self.include_outputs and 'outputs' in cell:
                        outputs = self._process_outputs(cell['outputs'])
                        if outputs:
                            content_parts.append('Output:\n')
                            content_parts.append(outputs)
                            content_parts.append('\n\n')
            
            return '\n'.join(content_parts).strip()
        
        except Exception as e:
            logger.error(f"Error parsing Jupyter notebook {file_path}: {e}")
            return ""
    
    def _preserve_latex(self, text: str) -> str:
        """
        Preserve LaTeX expressions in markdown text.
        
        Args:
            text: The markdown text containing LaTeX
            
        Returns:
            Processed text with preserved LaTeX
        """
        # Function to handle LaTeX block matches
        def handle_latex_block(match):
            content = match.group(1)
            # Ensure proper LaTeX delimiters
            if not content.strip().startswith('$$') and not content.strip().endswith('$$'):
                return f'$${content}$$'
            return match.group(0)
        
        # Function to handle inline LaTeX matches
        def handle_inline_latex(match):
            content = match.group(1)
            # Ensure proper LaTeX delimiters
            if not content.strip().startswith('$') and not content.strip().endswith('$'):
                return f'$${content}$$'
            return match.group(0)
        
        # Preserve display math blocks ($$...$$)
        text = re.sub(r'\$\$(.*?)\$\$', handle_latex_block, text, flags=re.DOTALL)
        
        # Preserve inline math ($...$)
        text = re.sub(r'\$(.*?)\$', handle_inline_latex, text)
        
        # Handle special markdown formatting
        text = re.sub(r'<font color="([^"]+)">(.*?)</font>', r'\2', text)  # Remove font color tags
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Handle bold text
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Handle italic text
        
        return text
    
    def _process_outputs(self, outputs: List[Dict[str, Any]]) -> str:
        """
        Process cell outputs and convert them to text.
        
        Args:
            outputs: List of output cells from the notebook
            
        Returns:
            Processed output text
        """
        output_parts = []
        
        for output in outputs:
            output_type = output.get('output_type', '')
            
            if output_type == 'stream':
                text = output.get('text', [])
                if isinstance(text, list):
                    text = ''.join(text)
                output_parts.append(text)
            
            elif output_type in ['execute_result', 'display_data']:
                # Handle text/plain output
                if 'data' in output and 'text/plain' in output['data']:
                    text = output['data']['text/plain']
                    if isinstance(text, list):
                        text = ''.join(text)
                    output_parts.append(text)
                
                # Handle LaTeX output
                if 'data' in output and 'text/latex' in output['data']:
                    latex = output['data']['text/latex']
                    if isinstance(latex, list):
                        latex = ''.join(latex)
                    output_parts.append(f'$${latex}$$')
            
            elif output_type == 'error':
                # Include error messages for debugging purposes
                ename = output.get('ename', 'Error')
                evalue = output.get('evalue', '')
                output_parts.append(f"{ename}: {evalue}")
        
        return '\n'.join(output_parts).strip()

    @staticmethod
    def can_handle(file_path: str) -> bool:
        """
        Check if the given file is a Jupyter notebook.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is a Jupyter notebook, False otherwise
        """
        return Path(file_path).suffix.lower() == '.ipynb' 