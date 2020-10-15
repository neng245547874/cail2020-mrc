import json


def convert(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    mrc_data = {"data": []}
    for data in full_data:
        mrc = {"paragraphs": []}
        mrc['paragraphs'].append({'context': ''.join(data['context'][0][1])})
        mrc['paragraphs'][0]['qas'] = [
            {"answers": [], 'is_impossible': 'true', 'question': data['question'], 'id': data['_id']}]
        # if data['answer'] == 'unknown':
        #     mrc['paragraphs'][0]['qas'] = [
        #         {"answers": [], 'is_impossible': 'true', 'question': data['question'], 'id': data['_id']}]
        # elif data['answer'] == 'yes':
        #     mrc['paragraphs'][0]['qas'] = [
        #         {"answers": [{'text': 'YES', 'answer_start': -1}], 'is_impossible': 'false',
        #          'question': data['question'],
        #          'id': data['_id']}]
        # elif data['answer'] == 'no':
        #     mrc['paragraphs'][0]['qas'] = [
        #         {"answers": [{'text': 'NO', 'answer_start': -1}], 'is_impossible': 'false',
        #          'question': data['question'],
        #          'id': data['_id']}]
        # else:
        #     mrc['paragraphs'][0]['qas'] = [{"answers": [
        #         {'text': data['answer'], 'answer_start': mrc['paragraphs'][0]['context'].find(data['answer'])}],
        #         'is_impossible': 'false', 'question': data['question'], 'id': data['_id']}]
        mrc_data['data'].append(mrc)

    json.dump(
        mrc_data,
        open(output_path, "w", encoding="utf8"),
        indent=2,
        ensure_ascii=False,
        sort_keys=False,
    )
