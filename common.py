"""
Common global variables / functions.
"""

PKL_LST_EXT = '.pkl_lst'
PKL_DIC_EXT = '.pkl_dic'
LOG_EXT = '.log'
NATIVE_COUNTRIES = {'Australia', 'England', 'Ireland', 'Scotland', 'Wales', 'UK', 'New Zealand', 'US', 'Canada'}
COUNTRIES_ORDER = ['UK', 'Scotland', 'England', 'Wales', 'Ireland',
                   'New Zealand', 'Australia',
                   'US', 'Canada',
                   'Austria', 'Germany', 'Netherlands', 'Norway', 'Sweden',
                   'Bulgaria', 'Croatia', 'Slovenia', 'Serbia', 'lithuania', 'Poland', 'Latvia', 'Russia', 'Slovakia', 'Czechia', 'Czech Republic', 'Slobenia', 'Ukraine',
                   'France', 'Italy', 'Portugal', 'Romania', 'Spain',
                   'Greece', 'Estonia', 'Finland', 'Turkey', 'Israel', 'Palestine', 'Hungary']


class LanguageFamilies:
    # Sets inside dictionaries so the lookup will be fast. The order here doesn't matter.
    _families = {'English': {'Australia', 'England', 'Ireland', 'Scotland', 'Wales', 'UK', 'New Zealand', 'US',
                             'Canada', },
                 'Germanic': {'Austria', 'Germany', 'Denmark', 'Netherlands', 'Norway', 'Sweden', 'Iceland'},
                 'Slavic': {'Bosnia', 'Bulgaria', 'Croatia', 'Czech Republic', 'Latvia', 'Lithuania', 'Poland',
                            'Russia', 'Serbia', 'Slovakia', 'Slovenia', 'Ukraine', },
                 'Romance': {'France', 'Italy', 'Portugal', 'Romania', 'Spain', 'Mexico', }}

    @staticmethod
    def get_family(family_name):
        return LanguageFamilies._families.get(family_name)

    @staticmethod
    def get_country_fam(country_name):
        for k, v in LanguageFamilies._families.items():
            if country_name in v:
                return k
        return None


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
