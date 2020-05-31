# import os
# import json
# import numpy as np
#
# def add_types(data):
#     for d in data:
#         relation = d['relation']
#         if relation.lower() == 'other':
#             subj_type, obj_type = '*'
#         else:
#             subj_type, obj_type = relation.split('-')
#         d['subj_type'] = subj_type
#         d['obj_type'] = obj_type
#         ner_list = []
#         for idx, token in d['token']:
#             if d['subj_start'] < idx and idx < d['subj_end']:
#
#
#
# if __name__ == '__main__':
#     data_dir = '/Users/georgestoica/Desktop/icloud_desktop/Research/gcn-over-pruned-trees/semeval/dataset/semeval/aggcn_semeval'
#     train_file = os.path.join(data_dir, 'train_sampled.json')
#     dev_file = os.path.join(data_dir, 'dev.json')
#     test_file = os.path.join(data_dir, 'test.json')

