eng_ru = {
    'a': 'а',
    'b': 'в',
    'e': 'е',
    'k': 'к',
    'm': 'м',
    'h': 'н',
    'o': 'о',
    'p': 'р',
    'c': 'с',
    't': 'т',
    'y': 'у',
    'x': 'х'
}

ru_eng = {
    'а': 'a',
    'в': 'b',
    'е': 'e',
    'к': 'k',
    'м': 'm',
    'н': 'h',
    'о': 'o',
    'р': 'p',
    'с': 'c',
    'т': 't',
    'у': 'y',
    'х': 'x'
}


def translate(text, dictionary):
    text = text.lower()
    for key in dictionary.keys():
        text = text.replace(key, dictionary[key])
    return text
