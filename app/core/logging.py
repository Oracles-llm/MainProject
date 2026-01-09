"""
Standard logging configuration for the LLM RAG Platform.
Provides a reusable logger that can be imported across the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "Oracales llm",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up and configure a logger with console and file handlers.
    
    Args:
        name: Logger name (default: "Oracales llm")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name. If None, uses {name}.log
        log_dir: Directory to store log files (default: "logs")
    
    Returns:
        Configured logger instance
    """

    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    if log_file or log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        if log_file:
            file_path = log_path / log_file
        else:
            file_path = log_path / f"{name}.log"
        
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. If name is provided, returns a child logger.
    If no name is provided, returns the root logger.
    
    Args:
        name: Optional logger name (typically __name__ of the calling module)
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"Oracales llm.{name}")
    return logging.getLogger("Oracales llm")



