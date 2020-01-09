DB_FNAME = '../../hmt.db'
LEX_DNAME = '../../data/lexica/'

FEATURES = [
    # 'intensity_min',
    'intensity_mean', 
    'intensity_max', 
    # 'intensity_std', 
    # 'pitch_min', 
    'pitch_mean', 
    'pitch_max', 
    # 'pitch_std', 
    'jitter', 
    'shimmer', 
    'nhr',
    'rate_syl', 
    # 'rate_vcd'
]

# number of most frequent words to consider
MF_COUNT = 25

GENDER_PAIRS = ['f', 'm', 'fm']
GENDERS = ['f', 'm']
ROLES = ['d', 'f']

IDS = {
    'ID_TYPES_SPL': 'ID_TYPES_SPL',
    'ID_TYPES_MF': 'ID_TYPES_MF',
    'ID_POS_SPL': 'ID_POS_SPL',
    'ID_NEG_STD': 'ID_NEG_STD',
    'ID_LEM_SPL': 'ID_LEM_SPL'
}


