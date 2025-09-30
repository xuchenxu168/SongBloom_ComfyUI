from . import chinese,  english # , japanese 暂时干掉看看
from .symbols import *
import yaml
language_module_map = {"zh": chinese, "en": english} #, "ja": japanese

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

import re

# def split_text(text):
#     chinese_pattern = r'[\u4e00-\u9fa5][\u4e00-\u9fa5\ \,\.\!\?\，\。]+'
#     english_pattern = r'[a-zA-Z][a-zA-Z\'\ \,\.\!\?]+'
    
#     chinese_text = re.findall(chinese_pattern, text)
#     print(chinese_text)
#     english_text = re.findall(english_pattern, text)
    
#     return chinese_text, english_text

def split_text(text):
    pattern = re.compile("|".join(re.escape(p) for p in chinese.rep_map.keys()))
    text = pattern.sub(lambda x: chinese.rep_map[x.group()], text)

    result = []
    lang = []
    buffer = ""
    chinese_pattern = r'[\u4e00-\u9fa5]'
    special_pattern = r'[\,\.\!\?\…\-]'
    # TODO check 一下
    for char in text:
        if re.match(special_pattern, char):
            if buffer:
                if not re.match(chinese_pattern, buffer[0]):
                    result.append(buffer)
                    lang.append('en')
                else:
                    result.append(buffer)
                    lang.append("zh")
            result.append(char)
            lang.append('sp')
            buffer = ""

        
        elif re.match(chinese_pattern, char):
            if buffer and not re.match(chinese_pattern, buffer[-1]):
                result.append(buffer)
                buffer = ""
                lang.append('en')
            buffer += char
        else:
            if buffer and re.match(chinese_pattern, buffer[-1]):
                result.append(buffer)
                buffer = ""
                lang.append("zh")
            buffer += char

    if buffer:
        result.append(buffer)
        lang.append("zh" if re.match(chinese_pattern, buffer[-1]) else 'en')

    return result, lang

def mixed_language_to_phoneme(text):
    segments, lang = split_text(text)
    # print(segments, lang)
    result = [language_to_phoneme(s, l) for s, l in zip(segments, lang)]
    phones, word2ph = [], []
    for p, w, n in result:
        phones += p
        if w is None:
            w = []
        word2ph += w
    return phones, word2ph


def language_to_phoneme(text, language):
    if language == 'sp':
        return [text], None, text
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        try:
            phones = language_module.g2p(norm_text)
        except:
            phones = [norm_text]
        word2ph = None

    # for ph in phones:
    #     assert ph in symbols, ph
    return phones, word2ph, norm_text

def gen_vocabs():
    yaml.dump(symbols, open('./vocab.yaml', 'w'))

class G2P_Mix():
    def __call__(self, text):
        phones, word2ph = mixed_language_to_phoneme(text)
        return ' '.join(phones)