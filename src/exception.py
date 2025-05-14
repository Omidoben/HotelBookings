import sys
from src.logger import logging

# This function formats the error message with file name and line number
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Get traceback details
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get file where error occurred
    error_message = "Error occurred in Python script [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

# Custom exception class - Defines a custom exception that inherits from Python’s built-in Exception.
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message