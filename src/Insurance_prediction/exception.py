import sys
from src.Insurance_prediction.logger import logging
import traceback

def error_message_detail(error, error_detail=None):
    # It's better to get the current exception info directly, if not provided
    if error_detail is None:
        error_detail = sys.exc_info()
    exc_type, exc_value, exc_tb = error_detail
    # If traceback is available, extract file name and line number
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = "Error occurred in script [{0}] at line [{1}]: [{2}]".format(
            file_name, line_number, str(error))
    else:
        # If no traceback, return a more generic message
        error_message = "An error occurred: {}".format(str(error))
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_details=None):
        super().__init__(error_message)
        if error_details:
            self.error_message = f"{error_message}\nDetails: {traceback.format_exc()}"
        else:
            self.error_message = error_message

    def __str__(self):
        return self.error_message

