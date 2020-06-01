from stanfordcorenlp import StanfordCoreNLP
import os
import json
import numpy as np
import stanza


def parse_raw_text(text_file):
    parsed_text = {}
    with open(text_file, 'r') as handle:
        raw_text = handle.readlines()
        for i in range(0, len(raw_text), 4):
            sentence_id, sentence = raw_text[i].split('\t')
            filtered_sentence = sentence.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "").strip().strip('"')
            if sentence_id in parsed_text:
                raise ValueError('sentence ids must be unique!')
            parsed_text[sentence_id] = filtered_sentence
    return parsed_text

def extract_nlp_components(text_file, stanza_nlp, core_nlp):
    components = []
    with open(text_file, 'r') as handle:
        raw_text = handle.readlines()
        for counter, i in enumerate(range(0, len(raw_text), 4)):
            sentence_id, sentence = raw_text[i].split('\t')
            sentence = sentence.strip().strip('"')
            tokens = core_nlp.word_tokenize(sentence)
            # SUBJECT/OBJECT SPANS
            subject_start = tokens.index('<e1>')
            subject_end = tokens.index('</e1>') - 2
            object_start = tokens.index('<e2>') - 2
            object_end = tokens.index('</e2>') - 4
            sentence = sentence.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
            parsed_tokens = core_nlp.word_tokenize(sentence)
            # NER CALCULATION
            _, token_ners = zip(*core_nlp.ner(sentence))
            token_ners = list(token_ners)
            subject_ners = np.unique(np.array(token_ners)[subject_start: subject_end + 1])
            object_ners = np.unique(np.array(token_ners)[object_start:object_end + 1])
            subject_ner = compute_entity_ner(subject_ners)
            object_ner = compute_entity_ner(object_ners)
            # POS CALCULATION
            _, token_pos = zip(*core_nlp.pos_tag(sentence))
            token_pos = list(token_pos)
            # DEPENDENCY TREE
            # stanza_tokens = stanza_nlp(sentence).sentences[0].words
            # token_deprel, token_head = get_deprel_and_head(stanza_tokens)
            token_deprel, token_head = extract_dependencies(sentence, core_nlp=core_nlp)
            # RELATION
            relation = raw_text[i+1].strip()
            # QUALITY CHECK
            assert (len(parsed_tokens) == len(token_ners))
            assert (len(parsed_tokens) == len(token_pos))
            assert (len(parsed_tokens) == len(token_deprel))
            assert (len(parsed_tokens) == len(token_head))
            assert (subject_start >= 0)
            assert (subject_end < len(parsed_tokens))
            assert (object_start >= 0)
            assert (object_end < len(parsed_tokens))

            sample = {
                'id': str(int(sentence_id) - 1),
                'token': parsed_tokens,
                'subj_start': subject_start,
                'subj_end': subject_end,
                'obj_start': object_start,
                'obj_end': object_end,
                'subj_type': subject_ner,
                'obj_type': object_ner,
                'stanford_pos': token_pos,
                'stanford_ner': token_ners,
                'stanford_deprel': token_deprel,
                'stanford_head': token_head,
                'relation': relation
            }
            components.append(sample)
            if counter > 0 and counter % int(len(raw_text) / 40) == 0:
                prop_finished = counter / (len(raw_text) / 40)
                print(f'Finished {prop_finished}')

        return components

def extract_dependencies(sentence, core_nlp):
    dependencies = core_nlp.dependency_parse(sentence)
    deprels = [''] * len(dependencies)
    heads = [0] * len(dependencies)
    for node in dependencies:
        deprel, head, idx = node
        deprels[idx-1] = deprel
        heads[idx-1] = head
    return deprels, heads

def get_deprel_and_head(stanza_tokens):
    deprel = []
    head = []
    for token in stanza_tokens:
        deprel.append(token.deprel)
        head.append(token.head)
    return deprel, head

def load_data(data_file):
    with open(data_file, 'r') as handle:
        return json.load(handle)

def compute_entity_ner(ners):
    entity_ner = None
    for candidate_ner in ners:
        if entity_ner is None:
            entity_ner = candidate_ner
        elif candidate_ner != 'O' and entity_ner == 'O':
            entity_ner = candidate_ner
    return entity_ner

def augment_data(data, nlp):
    new_data = []
    skip_data = []
    for idx, sample in enumerate(data):
        # sample id is zero indexed but sentences are 1 indexed
        sample_id = str(int(sample['id']) + 1)
        # sentence = sentences[sample_id]
        tokens = sample['token']
        sentence = ' '.join(tokens)
        token_ners = nlp.ner(sentence)
        # Add NER
        ners = []
        for token, ner in token_ners:
            if ner.isupper():
                ners.append(ner)
            else:
                ners.append('O')
        # assert (len(ners) == len(tokens))
        if len(ners) != len(tokens):
            ners = match_ner_to_tokens(base_tokens=tokens,
                                       extracted_tokens=nlp.word_tokenize(sentence),
                                       ners=ners)
        sample['stanford_ner'] = ners
        ss, se = sample['subj_start'], sample['subj_end']
        os, oe = sample['obj_start'], sample['obj_end']

        subject_ners = np.unique(np.array(ners)[ss: se+1])
        object_ners = np.unique(np.array(ners)[os:oe+1])

        subject_ner = compute_entity_ner(subject_ners)
        object_ner = compute_entity_ner(object_ners)
        sample['subj_type'] = subject_ner
        sample['obj_type'] = object_ner
        if len(ners) > len(tokens):
            skip_data.append(sample)
        else:
            new_data.append(sample)
        if idx % int(len(data) * .1) == 0:
            prop_done = idx / len(data)
            print('Proportion completed: {}'.format(prop_done))
    return new_data


def match_ner_to_tokens(base_tokens, extracted_tokens, ners):
    matched_ners = []
    extracted_idx = 0
    for base_token in base_tokens:
        extracted_token = extracted_tokens[extracted_idx]
        ner = ners[extracted_idx]
        if base_token == extracted_token:
            extracted_idx += 1
        else:
            spread_ners = set()
            while extracted_token != base_token:
                spread_ners.add(ners[extracted_idx])
                extracted_idx += 1
                extracted_token += extracted_tokens[extracted_idx]
            extracted_idx += 1
            ner = None
            for spread_ner in spread_ners:
                if ner is None:
                    ner = spread_ner
                elif ner == 'O' and spread_ner != 'O':
                    ner = spread_ner
        matched_ners.append(ner)
    return matched_ners

if __name__ == '__main__':
    # stanza.download('en')
    # stanza_nlp = stanza.Pipeline('en')
    core_nlp = StanfordCoreNLP(r'/Users/georgestoica/Desktop/stanford-corenlp-4.0.0') #stanford-corenlp-4.0.0 | stanford-corenlp-full-2016-10-31

    # data_dir = '/Users/georgestoica/Desktop/icloud_desktop/Research/gcn-over-pruned-trees/semeval/dataset/semeval'
    data_dir = '/Users/georgestoica/Desktop/icloud_desktop/Research/gcn-over-pruned-trees/semeval/dataset/semeval/aggcn_semeval'
    train_file = os.path.join(data_dir, 'train.json')
    test_file = os.path.join(data_dir, 'test.json')
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    train_data = augment_data(train_data, core_nlp)
    test_data = augment_data(test_data, core_nlp)
    # mismatch_sentences = []
    # for idx, d in enumerate(train_data):
    #     tokens = d['token']
    #     sentence = stanza_nlp(' '.join(tokens)).sentences[0]
    #     if len(tokens) != len(sentence.words):
    #         mismatch_sentences.append(d)
    # raw_semevel_dir = '/Users/georgestoica/Desktop/SemEval2010_task8_all_data'
    # train_file = os.path.join(raw_semevel_dir, 'SemEval2010_task8_training', 'TRAIN_FILE.TXT')
    # test_file = os.path.join(raw_semevel_dir, 'SemEval2010_task8_testing_keys', 'TEST_FILE_FULL.TXT')
    # train_sentences = parse_raw_text(train_file)
    # test_sentences = parse_raw_text(test_file)

    # train_data = extract_nlp_components(train_file, stanza_nlp=stanza_nlp, core_nlp=core_nlp)
    train_save_file = os.path.join(data_dir, 'train_sampled.json')
    json.dump(train_data, open(train_save_file, 'w'))
    # test_data = extract_nlp_components(test_file, stanza_nlp=stanza_nlp, core_nlp=core_nlp)
    test_save_file = os.path.join(data_dir, 'test_new.json')
    json.dump(test_data, open(test_save_file, 'w'))

