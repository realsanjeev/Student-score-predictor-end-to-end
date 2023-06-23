import sys
from src.logger import logging

def error_message_details(error, error_details: sys):
    error_class, error_desc, exc_obj = error_details.exc_info()
    python_file_name = exc_obj.tb_frame.f_code.co_filename
    error_message = f"[ERROR] Error occurred in python script name \
        [[{python_file_name}]] line number [{exc_obj.tb_lineno}] error message [{error}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error = error
        self.error_detail = error_detail
        self.error_message = error_message_details(self.error, self.error_detail)

    def __str__(self) -> str:
        return self.error_message

if __name__ == "__main__":
    logging.info("Exception.py man is executed")
    try:
        a = 1 / 0
        print("value of a:", a)
    except Exception as err:
        print("*"*100)
        raise CustomException(err, sys)
    print("*"*100)
