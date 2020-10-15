import json
import logging
import os

import random

logger = logging.getLogger(__name__)


def generate_fix_test_data(full_file, path):
    """
    生成固定的测试集数据。

    该数据仅用于基本的模型可用性测试。随机抽取20%数据。

    :param raw_data_dir: 原始的数据集文件
    """

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    random.shuffle(full_data)
    n_test = int(len(full_data) * 0.2)
    test_data = full_data[:n_test]
    train_data = full_data[n_test:]

    with open(os.path.join(path, "train.json"), 'w', encoding='utf8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(os.path.join(path, "test.json"), 'w', encoding='utf8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    # json.dump(
    #     test_answers_dict,
    #     open(test_label_file, "w", encoding="utf8"),
    #     indent=2,
    #     ensure_ascii=False,
    #     sort_keys=True,
    # # )


def generate_Kfold_data(full_file, path):
    """
    生成固定的测试集数据。

    该数据仅用于基本的模型可用性测试。随机抽取20%数据。

    :param raw_data_dir: 原始的数据集文件
    """

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    random.shuffle(full_data)
    n_test = int(len(full_data) * 0.2)
    all_train_data = []
    all_test_data = []
    # 1
    test_data = full_data[:n_test]
    train_data = full_data[n_test:]
    all_train_data.append(train_data)
    all_test_data.append(test_data)
    with open(os.path.join(path, "train_1.json"), 'w', encoding='utf8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(os.path.join(path, "test_1.json"), 'w', encoding='utf8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    # 2
    test_data = full_data[n_test:n_test * 2]
    train_data = full_data[:n_test] + full_data[n_test * 2:]
    all_train_data.append(train_data)
    all_test_data.append(test_data)
    with open(os.path.join(path, "train_2.json"), 'w', encoding='utf8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(os.path.join(path, "test_2.json"), 'w', encoding='utf8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    # 3
    test_data = full_data[n_test * 2:n_test * 3]
    train_data = full_data[:n_test * 2] + full_data[n_test * 3:]
    all_train_data.append(train_data)
    all_test_data.append(test_data)

    with open(os.path.join(path, "train_3.json"), 'w', encoding='utf8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(os.path.join(path, "test_3.json"), 'w', encoding='utf8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    # 4
    test_data = full_data[n_test * 3:n_test * 4]
    train_data = full_data[:n_test * 3] + full_data[n_test * 4:]
    all_train_data.append(train_data)
    all_test_data.append(test_data)

    with open(os.path.join(path, "train_4.json"), 'w', encoding='utf8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(os.path.join(path, "test_4.json"), 'w', encoding='utf8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    # 5
    test_data = full_data[n_test * 4:n_test * 5]
    train_data = full_data[:n_test * 4] + full_data[n_test * 5:]
    all_train_data.append(train_data)
    all_test_data.append(test_data)

    with open(os.path.join(path, "train_5.json"), 'w', encoding='utf8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(os.path.join(path, "test_5.json"), 'w', encoding='utf8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    # with open(os.path.join(path, "alltrain.json"), 'w', encoding='utf8') as f:
    #     json.dump(all_train_data, f, indent=4, ensure_ascii=False)
    #
    # with open(os.path.join(path, "alltest.json"), 'w', encoding='utf8') as f:
    #     json.dump(all_test_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    generate_Kfold_data("./train.json", "./input")
