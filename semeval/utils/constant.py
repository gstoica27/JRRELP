
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
SUBJ_NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'TITLE': 3, 'CAUSE_OF_DEATH': 4, 'CRIMINAL_CHARGE': 5, 'ORGANIZATION': 6, 'DATE': 7, 'MISC': 8, 'DURATION': 9, 'PERSON': 10, 'NUMBER': 11, 'STATE_OR_PROVINCE': 12, 'MONEY': 13, 'LOCATION': 14, 'TIME': 15, 'IDEOLOGY': 16, 'COUNTRY': 17, 'RELIGION': 18, 'ORDINAL': 19, 'NATIONALITY': 20}

OBJ_NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'TITLE': 3, 'CAUSE_OF_DEATH': 4, 'CRIMINAL_CHARGE': 5, 'DATE': 6, 'SET': 7, 'DURATION': 8, 'ORDINAL': 9, 'MISC': 10, 'TIME': 11, 'MONEY': 12, 'RELIGION': 13, 'COUNTRY': 14, 'IDEOLOGY': 15, 'NATIONALITY': 16, 'LOCATION': 17}

NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'TITLE': 3, 'ORGANIZATION': 4, 'CITY': 5, 'COUNTRY': 6, 'DATE': 7, 'PERSON': 8, 'CAUSE_OF_DEATH': 9, 'NUMBER': 10, 'STATE_OR_PROVINCE': 11, 'MISC': 12, 'NATIONALITY': 13, 'LOCATION': 14, 'CRIMINAL_CHARGE': 15, 'DURATION': 16, 'ORDINAL': 17, 'TIME': 18, 'SET': 19, 'PERCENT': 20, 'IDEOLOGY': 21, 'RELIGION': 22, 'MONEY': 23, 'URL': 24}


POS_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'DT': 2, 'NN': 3, 'IN': 4, 'VBN': 5, 'RB': 6, 'VBZ': 7, 'PRP$': 8, 'JJS': 9, 'NNS': 10, '.': 11, 'TO': 12, 'VB': 13, 'JJ': 14, 'NNP': 15, 'VBG': 16, 'WDT': 17, 'POS': 18, 'VBP': 19, 'VBD': 20, 'CC': 21, 'CD': 22, 'HYPH': 23, 'PRP': 24, ',': 25, 'WP': 26, '``': 27, "''": 28, 'RP': 29, 'RBR': 30, '-LRB-': 31, '-RRB-': 32, ':': 33, 'UH': 34, 'MD': 35, 'NNPS': 36, 'WRB': 37, 'EX': 38, 'FW': 39, 'RBS': 40, 'JJR': 41, 'SYM': 42, 'WP$': 43, '$': 44, 'PDT': 45, 'NFP': 46, 'ADD': 47, 'GW': 48, 'AFX': 49, 'LS': 50}

DEPREL_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'det': 2, 'ROOT': 3, 'mark': 4, 'nsubj': 5, 'advmod': 6, 'dep': 7, 'nmod:poss': 8, 'amod': 9, 'obj': 10, 'case': 11, 'nmod': 12, 'compound': 13, 'punct': 14, 'xcomp': 15, 'obl': 16, 'cop': 17, 'acl:relcl': 18, 'ccomp': 19, 'aux': 20, 'nsubj:pass': 21, 'aux:pass': 22, 'cc': 23, 'conj': 24, 'nummod': 25, 'acl': 26, 'appos': 27, 'obl:npmod': 28, 'compound:prt': 29, 'obl:tmod': 30, 'fixed': 31, 'advcl': 32, 'discourse': 33, 'csubj': 34, 'expl': 35, 'parataxis': 36, 'iobj': 37, 'cc:preconj': 38, 'csubj:pass': 39, 'det:predet': 40, 'orphan': 41, '': 42, 'goeswith': 43}

NEGATIVE_LABEL = 'Other'

# LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4, 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9, 'Entity-Destination-rev': 10, 'Cause-Effect-rev': 11, 'Member-Collection-rev': 12, 'Entity-Origin-rev': 13, 'Message-Topic-rev': 14, 'Component-Whole-rev': 15, 'Instrument-Agency-rev': 16, 'Product-Producer-rev': 17, 'Content-Container-rev': 18}
# LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4, 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9}
LABEL_TO_ID = { 'Other': 0, 'Component-Whole(e2,e1)': 1, 'Instrument-Agency(e2,e1)': 2, 'Member-Collection(e1,e2)': 3, 'Cause-Effect(e2,e1)': 4, 'Entity-Destination(e1,e2)': 5, 'Content-Container(e1,e2)': 6, 'Message-Topic(e1,e2)': 7, 'Product-Producer(e2,e1)': 8, 'Member-Collection(e2,e1)': 9, 'Entity-Origin(e1,e2)': 10, 'Cause-Effect(e1,e2)': 11, 'Component-Whole(e1,e2)': 12, 'Message-Topic(e2,e1)': 13, 'Product-Producer(e1,e2)': 14, 'Entity-Origin(e2,e1)': 15, 'Instrument-Agency(e1,e2)': 16, 'Content-Container(e2,e1)': 17, 'Entity-Destination(e2,e1)': 18}

INFINITY_NUMBER = 1e12

















