"""
Common global variables / functions.
"""

PKL_LST_EXT = '.pkl_lst'
PKL_DIC_EXT = '.pkl_dic'
LOG_EXT = '.log'
# NATIVE_COUNTRIES = {'Australia', 'England', 'UK', 'US'}
NATIVE_COUNTRIES = {'Australia', 'England', 'Ireland', 'Scotland', 'Wales', 'UK', 'New Zealand', 'US', 'Canada'}
# NATIVE_COUNTRIES = {'Australia', 'England', 'Ireland', 'Scotland', 'Wales', 'UK', 'US'}
DATE_STR_SHORT = "%Y-%m-%d %H-%M-%S"
DATE_STR_LONG = "%Y-%m-%d %H-%M-%S.%f"


def print_log(msg, file=None, to_console=True, end='\n'):
    def _print_log(*args, **kwargs):
        if to_console:
            print(*args, **kwargs)
        if file:
            with open(file, 'a', encoding='utf-8') as log_f:
                print(*args, **kwargs, file=log_f)
    _print_log(msg, end=end)
