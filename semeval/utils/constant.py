
"""
Define constants for semeval-10 task.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
NO_RELATION_ID = 0

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'TITLE': 3, 'CAUSE_OF_DEATH': 4, 'CRIMINAL_CHARGE': 5, 'ORGANIZATION': 6, 'DATE': 7, 'MISC': 8, 'DURATION': 9, 'PERSON': 10, 'NUMBER': 11, 'STATE_OR_PROVINCE': 12, 'MONEY': 13, 'IDEOLOGY': 14, 'TIME': 15, 'COUNTRY': 16, 'RELIGION': 17, 'LOCATION': 18, 'ORDINAL': 19, 'NATIONALITY': 20}

OBJ_NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'TITLE': 3, 'CAUSE_OF_DEATH': 4, 'CRIMINAL_CHARGE': 5, 'DATE': 6, 'SET': 7, 'DURATION': 8, 'MONEY': 9, 'RELIGION': 10, 'COUNTRY': 11, 'IDEOLOGY': 12, 'NATIONALITY': 13, 'LOCATION': 14, 'MISC': 15, 'TIME': 16, 'ORDINAL': 17}

NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'TITLE': 3, 'ORGANIZATION': 4, 'CITY': 5, 'COUNTRY': 6, 'DATE': 7, 'PERSON': 8, 'CAUSE_OF_DEATH': 9, 'NUMBER': 10, 'STATE_OR_PROVINCE': 11, 'MISC': 12, 'NATIONALITY': 13, 'LOCATION': 14, 'CRIMINAL_CHARGE': 15, 'ORDINAL': 16, 'TIME': 17, 'SET': 18, 'DURATION': 19, 'PERCENT': 20, 'IDEOLOGY': 21, 'RELIGION': 22, 'MONEY': 23, 'URL': 24}


POS_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'DT': 2, 'NN': 3, 'IN': 4, 'VBN': 5, 'RB': 6, 'VBZ': 7, 'PRP$': 8, 'JJS': 9, 'NNS': 10, '.': 11, 'TO': 12, 'VB': 13, 'JJ': 14, 'NNP': 15, 'VBG': 16, 'WDT': 17, 'POS': 18, 'VBP': 19, 'VBD': 20, 'CC': 21, 'CD': 22, 'HYPH': 23, 'PRP': 24, ',': 25, 'WP': 26, '``': 27, "''": 28, 'RP': 29, 'RBR': 30, '-LRB-': 31, '-RRB-': 32, ':': 33, 'UH': 34, 'MD': 35, 'NNPS': 36, 'WRB': 37, 'EX': 38, 'FW': 39, 'RBS': 40, 'JJR': 41, 'SYM': 42, 'WP$': 43, '$': 44, 'PDT': 45, 'NFP': 46, 'ADD': 47, 'GW': 48, 'AFX': 49, 'LS': 50}

DEPREL_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'det': 2, 'nsubj': 3, 'prep': 4, 'pcomp': 5, 'advmod': 6, 'root': 7, 'poss': 8, 'amod': 9, 'dobj': 10, 'pobj': 11, 'nn': 12, 'punct': 13, 'nsubjpass': 14, 'auxpass': 15, 'cc': 16, 'conj': 17, 'cop': 18, 'possessive': 19, 'rcmod': 20, 'mark': 21, 'ccomp': 22, 'aux': 23, 'vmod': 24, 'appos': 25, 'npadvmod': 26, 'num': 27, 'acomp': 28, 'dep': 29, 'prt': 30, 'tmod': 31, 'neg': 32, 'xcomp': 33, 'advcl': 34, 'mwe': 35, 'quantmod': 36, 'expl': 37, 'parataxis': 38, 'iobj': 39, 'csubjpass': 40, 'preconj': 41, 'number': 42, 'predet': 43, 'csubj': 44, 'discourse': 45}

NEGATIVE_LABEL = 'Other'

# LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4, 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9, 'Entity-Destination-rev': 10, 'Cause-Effect-rev': 11, 'Member-Collection-rev': 12, 'Entity-Origin-rev': 13, 'Message-Topic-rev': 14, 'Component-Whole-rev': 15, 'Instrument-Agency-rev': 16, 'Product-Producer-rev': 17, 'Content-Container-rev': 18}
LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4, 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9}
# LABEL_TO_ID = { 'Other': 0, 'Component-Whole(e2,e1)': 1, 'Instrument-Agency(e2,e1)': 2, 'Member-Collection(e1,e2)': 3, 'Cause-Effect(e2,e1)': 4, 'Entity-Destination(e1,e2)': 5, 'Content-Container(e1,e2)': 6, 'Message-Topic(e1,e2)': 7, 'Product-Producer(e2,e1)': 8, 'Member-Collection(e2,e1)': 9, 'Entity-Origin(e1,e2)': 10, 'Cause-Effect(e1,e2)': 11, 'Component-Whole(e1,e2)': 12, 'Message-Topic(e2,e1)': 13, 'Product-Producer(e1,e2)': 14, 'Entity-Origin(e2,e1)': 15, 'Instrument-Agency(e1,e2)': 16, 'Content-Container(e2,e1)': 17, 'Entity-Destination(e2,e1)': 18}

INFINITY_NUMBER = 1e12

















