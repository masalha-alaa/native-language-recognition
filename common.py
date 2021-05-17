"""
Common global variables / functions.
"""

PKL_LST_EXT = '.pkl_lst'
PKL_DIC_EXT = '.pkl_dic'
LOG_EXT = '.log'
# NATIVE_COUNTRIES = {'Australia', 'England', 'UK', 'US'}
NATIVE_COUNTRIES = {'Australia', 'England', 'Ireland', 'Scotland', 'Wales', 'UK', 'New Zealand', 'US'}
# NATIVE_COUNTRIES = {'Australia', 'England', 'Ireland', 'Scotland', 'Wales', 'UK', 'US'}


def print_log(msg, filename=None, to_console=True):
    def _print_log(*args, **kwargs):
        if to_console:
            print(*args, **kwargs)
        if filename:
            with open(filename, 'a', encoding='utf-8') as log_f:
                print(*args, **kwargs, file=log_f)
    _print_log(msg)
