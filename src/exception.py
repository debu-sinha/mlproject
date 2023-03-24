import sys


def error_handler(err, err_detail: sys):
    """
    This function is used to handle error and return the error message
    """
    _, _, exec_tb = err_detail.exc_info()
    filename = exec_tb.tb_frame.f_code.co_filename
    err_message = f"Error: {filename} at line {exec_tb.tb_lineno} error message {err}"
    return err_message


class CustomException(Exception):
    """
    This class is used to handle custom exception
    """

    def __init__(self, err, err_detail: sys):
        super().__init__(err)
        self.err_detail = error_handler(err, err_detail)

    def __str__(self):
        return self.err_detail
