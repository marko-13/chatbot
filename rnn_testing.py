# imports
from bot_rnn import RNNModel
from main import index_dataset, get_bot, read_paraphrazed_dataset, save_object
from rnn_dataset import Dataset


# Questions for testing accuracy
wanted_questions = [("Is fixed annuity safe?", [15453]),
                    ("Can I opt out of health insurance", [17884]),
                    ("How to become a medicare expert", [22917, 22918]),
                    ("Can disability insurance be deducted", [25381]),
                    ("Is scuba diving covered by life insurance", [num for num in range(20565, 20569)]),
                    ("is there life insurance for elderly", [25379, 25380]),
                    ("blanket life insurance", [9783]),
                    ("What is jumbo life insurance", [num for num in range(25650, 25654)]),
                    ("can i get a life insurance policy from my parent", [11963, 11964, 11965, 11966, 15786, 17280, 17281, 27635, 27636, 27637, 2465, 2466, 2467, 7215, 7216, 7399, 7400]),
                    ("cashing out a 401k", [49, 50, 1708, 5147, 5428, 7788, 7789, 12410]),
                    ("disability insurance and pregnancy", [746, 2516, 2517, 27196, 27197, 26405, 26314, 26315, 26069, 25980, 25981, 20765, 20435, 19122, 19123, 19124, 19125, 18062, 18063, 17447, 17448, 17004, 17005, 17006, 10633, 10634, 1253]),
                    ]



def test_rnn_glove_100d(model_wrapper):
    #nesto


def test_rnn_glove_200d(model_wrapper, high_recall):
    #nesto
    # model = model_wrapper.get_model()
    for question in wanted_questions:
        ret = model_wrapper.process_input()

    


def test_rnn_bert_default(model_wrapper):
    #nesto


def test_rnn_bert_custom(model_wrapper):
    #nesto

if __name__ == '__main__':

    if True:
        dict_orig = index_dataset()

        loader_bot = get_bot(dict_orig, 'word2vec', False)

        dict_paraphrazed = read_paraphrazed_dataset('paraphrazed_final_100.csv')

        dataset = Dataset(dict_orig, dict_paraphrazed, 5)
        save_object(dataset, f'./objects/rnn_dataset_object.pickle')

        dict_q_paraq = read_paraphrazed_dataset('paraphrazed_final_100.csv')
        # print(dict_q_paraq)
        brojac = 0
        list_of_pairs = []
        pair_y = []
        # print(len(dataset.mixed_pairs_train))
        for i in range(0, len(dataset.mixed_pairs_train)):
            next_pair = dataset.get_next_pair()
            # print(next_pair)
            pair = []
            # print(next_pair[0][0])
            # print(next_pair[0][1])
            pair.append(dict_q_paraq[next_pair[0][0]])
            pair.append(dict_orig[int(next_pair[0][1])][0])
            # print(dict_orig[next_pair[0][1]])
            pair_y.append((next_pair[1]))
            list_of_pairs.append(pair)

    high_recall_questions = []
    for question_array in wanted_questions:
        high_recall_questions.append(loader_bot.process_input(question_array[0]))

    model_bert_default = RNNModel(list_of_pairs, pair_y, bert=True)
    model_bert_custom = RNNModel(list_of_pairs, pair_y, hybrid=True)
    model_glove_200d = RNNModel(list_of_pairs, pair_y)

    # test_rnn_glove_100d()
    test_rnn_glove_200d(model_glove_200d, high_recall_questions)
    test_rnn_bert_default(model_bert_default, high_recall_questions)
    test_rnn_bert_custom(model_bert_custom, high_recall_questions)