import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from pinyin.pinyin import G2P_PinYin
from cn_zh_g2p import G2P_Mix, symbols

key2processor = {
    'pinyin': G2P_PinYin(), 
    'phoneme': G2P_Mix(),
}

valid_struct_type = ['[chorus]', '[verse]', '[bridge]']
start_struct_type = ['[intro]', '[start]']
end_struct_type = ['[outro]', '[end]']
conn_struct_type = ['[inst]', '[solo]', '[break]']

LABELS = {
    '[intro]': 0,
    '[outro]': 1,
    '[bridge]': 2,
    '[inst]': 3,
    '[verse]': 4,
    '[chorus]': 5,
    '[silence]': 6,
}

NUMBERS = {
    '0': ['零', 'zero'],
    '1': ['一', 'one'],
    '2': ['二', 'two'],
    '3': ['三', 'three'],
    '4': ['四', 'four'],
    '5': ['五', 'five'],
    '6': ['六', 'six'],
    '7': ['七', 'seven'],
    '8': ['八', 'eight'],
    '9': ['九', 'nine']
}

def detect_structure(structure):
    valid_start = ['start', 'intro']
    valid_end = ['outro', 'end']
    valid_instru = ['solo', 'inst', 'break']
    valid_bridge = ['bridge']

    if structure in ['verse', 'chorus', 'silence']:
        return structure
    
    if structure in valid_start:
        return 'intro'
    if structure in valid_end:
        return 'outro'
    if structure in valid_instru:
        return 'inst'
    if structure in valid_bridge:
        return 'bridge'

def merge_structure(start_time, end_time, structure, lyric):
    cnt = 1
    while cnt < len(start_time):
        if structure[cnt] == structure[cnt-1]:
            end_time[cnt-1] = end_time[cnt]
            if structure[cnt] not in ["verse", "chorus", "bridge"]:
                del start_time[cnt]
                del end_time[cnt]
                del structure[cnt]
                del lyric[cnt]
            else:
                cnt += 1
        else:
            cnt += 1
    
    return start_time, end_time, structure, lyric


def is_struct_legal(struct, text):
    if struct in valid_struct_type and text != "":
        return True
    elif struct not in valid_struct_type and text == "":
        return True
    return False
