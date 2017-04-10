import argparse
import json
import os
import nltk
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from squad.utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    target_dir = "data/squad"
    glove_dir = os.path.join(home, "data", "glove")
    # e.g.:
    # background_path = "/efs/data/dlfa/questions/intermediate/processed/intermediate_4_questions_with_lucene_background/intermediate_4_dev_background.tsv"
    # question_path = "/efs/data/dlfa/questions/intermediate/processed/intermediate_4_dev/question_and_answer/questions.tsv"
    # out_name = "intermediate_4_dev"
    parser.add_argument("background_path", type=str)
    parser.add_argument("question_path", type=str)
    parser.add_argument("out_name", type=str)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all-v1.1.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train-v1.1.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "dev-v1.1.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data['data'].extend(dev_data['data'])
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    prepro_each(args, args.background_path, args.question_path, out_name=args.out_name)

    # if args.mode == 'full':
    #     prepro_each(args, 'train', out_name='train')
    #     prepro_each(args, 'dev', out_name='dev')
    #     prepro_each(args, 'dev', out_name='test')
    # elif args.mode == 'all':
    #     create_all(args)
    #     prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
    #     prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
    #     prepro_each(args, 'all', out_name='train')
    # elif args.mode == 'single':
    #     assert len(args.single_path) > 0
    #     prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    # else:
    #     prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
    #     prepro_each(args, 'train', args.train_ratio, 1.0, out_name='dev')
    #     prepro_each(args, 'dev', out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def prepro_each(args, background_path, questions_path, out_name):
    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

    if not args.split:
        sent_tokenize = lambda para: [para]
    else:
        sent_tokenize = nltk.sent_tokenize

    # Read the questions and background tsvs
    # raw_file_data is a list of tuples, where the tuple is formatted as:
    # (string question, string passage, list of options, index of correct answer)
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    raw_file_data = []
    with open(questions_path, "r") as questions_file, open(background_path, "r") as background_file:
        for question_line, background_line in zip(questions_file, background_file):
            _, question_string, options_string, label_string = question_line.split("\t")
            options_list = options_string.split("###")
            label_int = int(label_string)
            # background_split("\t")[1:] splits on tab, and removes the index at the start of the line.
            # Then we join the list of strings by spaces.
            passage_string = ' '.join(background_line.split("\t")[1:])
            raw_file_data.append((question_string, passage_string, options_list, label_int))

    tokenized_questions, tokenized_question_characters = [], []
    tokenized_passages, tokenized_passage_characters = [], []
    tokenized_options, tokenized_option_characters = [], []
    labels = []

    for file_line in tqdm(raw_file_data):
        question_string, passage_string, options_list, label_int = file_line
        # Turns a string passage into a list of sentences, where each sentence is a list of words.
        word_tokenized_passage = [word_tokenize(sentence) for sentence in sent_tokenize(passage_string)]
        word_tokenized_passage = [process_tokens(token) for token in word_tokenized_passage]

        # Further breaks up the word_tokenized_passage by simply replacing each word with a list of characters
        character_tokenized_passage = [[list(word) for word in sentence] for sentence in word_tokenized_passage]

        # Turns a string question into a list of words
        word_tokenized_question = word_tokenize(question_string)

        # Futher breaks up the word_tokenized_question by simply replacing each word with a list of characters
        character_tokenized_question = [list(word) for word in word_tokenized_question]

        # Turns a list of options into a list of options, where each option is a list of words
        word_tokenized_options = [word_tokenize(option) for option in options_list]

        # Futher breaks up the word_tokenized_options by simply replacing each word witha list of characters
        character_tokenized_options = [[list(word) for word in option] for option in word_tokenized_options]

        # update the counters with the frequency of each word and character in the passage, options, and question
        for sentence in word_tokenized_passage:
            for word in sentence:
                word_counter[word] += 1
                lower_word_counter[word.lower()] += 1
                for character in word:
                    char_counter[character] += 1

        for option in word_tokenized_options:
            for word in option:
                word_counter[word] += 1
                lower_word_counter[word.lower()] += 1
                for character in word:
                    char_counter[character] += 1

        for word in word_tokenized_question:
            word_counter[word] += 1
            lower_word_counter[word.lower()] += 1
            for character in word:
                char_counter[character] += 1

        tokenized_questions.append(word_tokenized_question)
        tokenized_question_characters.append(character_tokenized_question)

        tokenized_passages.append(word_tokenized_passage)
        tokenized_passage_characters.append(character_tokenized_passage)

        tokenized_options.append(word_tokenized_options)
        tokenized_option_characters.append(character_tokenized_options)

        labels.append(label_int)

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'tokenized_questions': tokenized_questions,
            'tokenized_question_characters': tokenized_question_characters,
            'tokenized_options': tokenized_options,
            'tokenized_option_characters': tokenized_option_characters,
            'tokenized_passages': tokenized_passages,
            'tokenized_passage_characters': tokenized_passage_characters,
            'labels': labels}

    shared = {'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}

    print("saving ...")
    save(args, data, shared, out_name)



if __name__ == "__main__":
    main()
