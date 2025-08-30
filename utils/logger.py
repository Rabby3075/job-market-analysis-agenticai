"""
Professional Logging System for Job Vacancies Agent
Provides structured logging with file output, rotation, and different log levels
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


class ProfessionalLogger:
    """Professional logging system with file output and rotation"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self.setup_logging()
    
    def setup_logging(self):
        """Setup professional logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console Handler (INFO level and above) - Windows compatible
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Set encoding for Windows compatibility
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except:
                pass  # Fallback if reconfigure not available
        
        # File Handler for all logs (DEBUG level and above) - Windows compatible
        all_logs_file = os.path.join(self.log_dir, f"{self.name}_all.log")
        file_handler = logging.handlers.RotatingFileHandler(
            all_logs_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'  # Windows compatibility
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Error Handler for errors only - Windows compatible
        error_logs_file = os.path.join(self.log_dir, f"{self.name}_errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_logs_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'  # Windows compatibility
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Performance Handler for timing logs - Windows compatible
        performance_logs_file = os.path.join(self.log_dir, f"{self.name}_performance.log")
        performance_handler = logging.handlers.RotatingFileHandler(
            performance_logs_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'  # Windows compatibility
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(detailed_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(performance_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def performance(self, operation: str, duration: float, details: str = ""):
        """Log performance metrics"""
        message = f"PERFORMANCE | {operation} | Duration: {duration:.3f}s | {details}"
        self.logger.info(message)
    
    def data_operation(self, operation: str, dataset: str, details: str = ""):
        """Log data operations"""
        message = f"DATA_OP | {operation} | Dataset: {dataset} | {details}"
        self.logger.info(message)
    
    def api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Log API requests"""
        message = f"API | {method} {endpoint} | Status: {status_code} | Duration: {duration:.3f}s"
        if status_code >= 400:
            self.logger.warning(message)
        else:
            self.logger.info(message)

# Create logger instances for different components
def get_logger(name: str) -> ProfessionalLogger:
    """Get a logger instance for a specific component"""
    return ProfessionalLogger(name)

# Convenience functions for common logging patterns
def log_function_call(func_name: str, args: dict = None):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.debug(f"Calling {func_name} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {str(e)}")
                raise
        return wrapper
    return decorator

def log_execution_time(operation: str):
    """Decorator to log execution time"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.performance(operation, duration)
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"{operation} failed after {duration:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator
