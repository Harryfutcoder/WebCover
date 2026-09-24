import re
import string
import random

RAW_TYPE_DICT = {
    # Random Input
    'search': ['search'],
    # Input Vocabulary
    'repeatedpassword': ['repeatedpassword', 'repeatpassword', 'passwordconfirm'],
    'newpassword': ['newpassword'],
    'password': ['password'],
    'email': ['email'],
    'username': ['username', 'user', 'login'],
    'phone': ['phone'],
    'bankname': ['bankname'],
    'routingnumber': ['routingnumber'],
    'accountnumber': ['accountnumber'],
    'number': ['number', 'amount', 'total', 'expense'],
    'startdate': ['startdate'],
    'enddate': ['enddate'],
    'date': ['date'],
}


class InputType:
    def __init__(self):
        self.type_dict = self.reverse_dict(RAW_TYPE_DICT)
        self.random_types = ['search', 'normal']

    @staticmethod
    def reverse_dict(dictionary: dict) -> dict:
        reversed_dict = {}
        for key, values in dictionary.items():
            for value in values:
                reversed_dict[value] = key
        return reversed_dict

    def get_input_type_by_html(self, input_html: str) -> str:
        cleaned_input_html = re.sub(r'[^a-zA-Z]', '', input_html).lower()
        for type_key in self.type_dict:
            if type_key in cleaned_input_html:
                return self.type_dict[type_key]
        return 'normal'


class InputVocabulary:
    def __init__(self, valid_dict):
        self.valid_dict = valid_dict
        # self.valid_dict = {
        #     'repeatedpassword': 'secret',
        #     'newpassword': 'secret',
        #     'password': 'secret',
        #     'email': 'webcover@example.com',
        #     'username': 'secret',
        #     'phone': '123123123',
        #     'number': '1',
        #     'startdate': '2024-03-30',
        #     'enddate': '2024-04-12',
        #     'date': '2024-01-29',
        # }


class InputRandom:
    def get_random_input(self, input_type: str = None) -> str:
        if input_type == 'search':
            return self.get_random_str(0, 1)
        elif input_type == 'email':
            return self.generate_random_email()
        elif input_type == 'date':
            return self.generate_random_date()
        elif input_type == 'number':
            return self.get_random_number(1, 10)
        else:
            return self.get_random_str(4, 8)

    @staticmethod
    def generate_random_email(max_length=10):
        chars = string.ascii_letters + string.digits + '_'
        domain_list = ['example.com', 'test.org', 'sample.net']
        domain = random.choice(domain_list)
        local_part = ''.join(random.choice(chars) for _ in range(random.randint(1, max_length)))
        return f"{local_part}@{domain}"

    @staticmethod
    def generate_random_date(start=2021, end=2022):
        # date_formate_list = [f'%Y-%m-%d', f'%m/%d/%Y']  # todo
        # date_formate = date_formate_list[random.randint(0, len(date_formate_list) - 1)]
        year = random.randint(start, end)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date = f"{year:04d}/{month:02d}/{day:02d}"
        return date

    @staticmethod
    def get_random_number(min_value: int, max_value: int) -> str:
        # return str(random.uniform(min_value, max_value))
        return str(random.randint(min_value, max_value))

    @staticmethod
    def get_random_str(min_len: int, max_len: int) -> str:
        length = random.randint(min_len, max_len)
        # characters = string.ascii_letters + string.digits + string.punctuation + string.whitespace
        characters = string.ascii_letters
        return ''.join(random.choice(characters) for _ in range(length))


class InputGenerator:

    def __init__(self, valid_dict):
        self.input_discriminator = InputType()
        self.vocabulary = InputVocabulary(valid_dict)
        self.random_generator = InputRandom()
        self.type = 0

    def get_single_input_value(self, input_html: str, invalid: bool = False) -> str:
        input_type = self.input_discriminator.get_input_type_by_html(input_html)
        if input_type in self.input_discriminator.random_types:
            res = self.random_generator.get_random_input(input_type)
        else:
            res = self.vocabulary.valid_dict[input_type]
        if type(res) == list:
            res = random.choice(res)
        return res
