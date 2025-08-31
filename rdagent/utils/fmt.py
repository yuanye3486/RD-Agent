"""
Tools that support generating better formats.
"""

def get_xgboost_params(use_gpu=True):
    """
    Get XGBoost parameters based on hardware availability.
    
    Args:
        use_gpu (bool): Whether to try to use GPU if available
        
    Returns:
        dict: XGBoost parameters
    """
    # Check if GPU is available and should be used
    gpu_available = False
    if use_gpu:
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            try:
                import subprocess
                result = subprocess.run(["nvidia-smi"], 
                                      capture_output=True, text=True, timeout=10)
                gpu_available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                gpu_available = False
    
    if gpu_available:
        return {
            "tree_method": "gpu_hist",
            "device": "cuda",
        }
    else:
        return {
            "tree_method": "hist",
            "device": "cpu",
        }


def shrink_text(
    text: str, context_lines: int = 200, line_len: int = 5000, *, row_shrink: bool = True, col_shrink: bool = True
) -> str:
    """
    When the context is too long, hide the part in the middle.

    >>> shrink_text("line1\\nline2\\nline3", context_lines=2, line_len=5)
    'line1\\n... (1 lines are hidden) ...\\nline3'

    >>> shrink_text("line1\\nline2\\nline3", context_lines=2, line_len=5, row_shrink=False)
    'line1\\nline2\\nline3'

    >>> shrink_text("short line", context_lines=2, line_len=5)
    'sh... (5 chars are hidden) ...ine'

    >>> shrink_text("a" * 5010, context_lines=2, line_len=10)
    'aaaaa... (5000 chars are hidden) ...aaaaa'
    """

    lines = text.splitlines()
    total_lines = len(lines)

    new_lines = []
    for line in lines:
        if col_shrink and len(line) > line_len:
            # If any line is longer than line_len, we can't shrink it
            line = f"{line[:line_len // 2]}... ({len(line) - line_len} chars are hidden) ...{line[- line_len + line_len // 2:]}"
        new_lines.append(line)
    lines = new_lines

    if not row_shrink or total_lines <= context_lines:
        return "\n".join(lines)

    # shrink row only when it is enabled and the total number of lines is greater than context_lines
    # Calculate how many lines to show from start and end
    half_lines = context_lines // 2
    start = "\n".join(lines[:half_lines])
    end = "\n".join(lines[-half_lines:])

    # Count the number of lines we're hiding
    hidden_lines = total_lines - half_lines * 2

    return f"{start}\n... ({hidden_lines} lines are hidden) ...\n{end}"