import argparse
import os
import csv
import json
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

images_gt = {}


def url_to_category(url):
    result = {}
    split_url = url.split('/')
    image_id = split_url[-1].replace('.png', '')
    result['category'] = 'implausible' if split_url[-3] == 'implausible' else 'plausible'
    if result['category'] == 'implausible':
        result['sub_category'] = split_url[-2]

    return image_id, result


def parse_category(raw_category: str):
    if raw_category.startswith('plausible'):
        return {'category': 'plausible'}

    result = {'category': 'implausible'}
    sub_category = raw_category.replace('implausible_', '')
    if sub_category:
        result['sub_category'] = sub_category

    return result


def get_user_answer(raw_answer):
    is_bc = True
    if len(raw_answer) > 2:
        is_bc = False

    c = next(x[0] for x in raw_answer.items() if x[1] is True)
    return is_bc, parse_category(c)


def get_category_key(is_bc, category_dict):
    main = category_dict['category']
    if main == 'plausible' or is_bc:
        return main

    return category_dict['sub_category']


def compare(is_bc, user_answer, gt):
    key1 = get_category_key(is_bc, user_answer)
    key2 = get_category_key(is_bc, gt)
    return key1 == key2


def parse_hit_results(user_answers):
    result = {'validation': True}
    is_bc, validation_answer = get_user_answer(user_answers['image_test'])
    if not compare(is_bc, validation_answer, parse_category(user_answers['image_test_result'])):
        result['validation'] = False
        return result

    result['questions'] = {}
    for question_id in range(1, 6):
        url_key = f'image_{question_id}_url'
        user_answer_key = f'image{question_id}'
        user_answer = user_answers[user_answer_key]
        url = user_answers[url_key]
        image_id, actual_category = url_to_category(url)
        if image_id not in images_gt:
            images_gt[image_id] = actual_category

        is_bc, user_category = get_user_answer(user_answer)
        result['questions'][question_id] = {
            'is_bc': is_bc,
            'image_id': image_id,
            'actual': actual_category,
            'actual_key': get_category_key(is_bc, actual_category),
            'predicted': user_category,
            'predicted_key': get_category_key(is_bc, user_category),
            'correct': compare(is_bc, user_category, actual_category)
        }

    return result


def filter_user_responses(csv_file):
    accepted_questions_bc = []
    accepted_questions_mcc = []
    accepted_count = 0
    rejected_count = 0

    for row in csv_file:
        answers = json.loads(row['Answer.taskAnswers'])[0]
        user_response = parse_hit_results(answers)
        if user_response['validation']:
            accepted_count += 1
            for question_id in user_response['questions']:
                if user_response['questions'][question_id]['is_bc']:
                    accepted_questions_bc.append(user_response['questions'][question_id])
                else:
                    accepted_questions_mcc.append(user_response['questions'][question_id])
        else:
            rejected_count += 1

    print(f'Out of {rejected_count + accepted_count} responses, {accepted_count} accepted.')
    return accepted_questions_bc, accepted_questions_mcc


def count_majority_vote(questions_by_image):
    votes = {}
    for image in questions_by_image:
        answers = questions_by_image[image]
        if len(answers) < 3:
            continue

        counted_answers = Counter([x['predicted_key'] for x in answers])
        top_2_results = counted_answers.most_common(2)
        if len(top_2_results) > 1 and top_2_results[0][1] == top_2_results[1][1]:
            votes[image] = 'undecided'
        else:
            votes[image] = top_2_results[0][0]

    return votes


def aggregate_by_image(responses):
    responses_by_image = {}
    for response in responses:
        image_id = response['image_id']
        if image_id not in responses_by_image:
            responses_by_image[image_id] = []

        responses_by_image[image_id].append(response)

    return responses_by_image


parser = argparse.ArgumentParser(description='Get a summary of the results submitted to the HITs')
parser.add_argument('--csv_dir', '-c', help='Path to the directory containing CSV files')
args = parser.parse_args()

csv_dir = args.csv_dir

rows = []
for filename in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, filename)
    if filename.lower().endswith('.csv'):
        input_file = csv.DictReader(open(file_path))
        rows.extend(input_file)

accepted_bc, accepted_mcc = filter_user_responses(rows)
accepted_by_image_bc = aggregate_by_image(accepted_bc)
accepted_by_image_mcc = aggregate_by_image(accepted_mcc)
majority_vote_bc = count_majority_vote(accepted_by_image_bc)
majority_vote_mcc = count_majority_vote(accepted_by_image_mcc)

labels = []
preds = []
labels_to_idx = {'plausible': 0, 'implausible': 1, 'undecided': 2}
titles = ['plausible', 'implausible', 'undecided']

for image_id in majority_vote_bc:
    labels.append(labels_to_idx[get_category_key(True, images_gt[image_id])])
    preds.append(labels_to_idx[majority_vote_bc[image_id]])

bc_cf_matrix = confusion_matrix(labels, preds)
d = ConfusionMatrixDisplay.from_predictions(labels, preds)
plt.show()

labels = []
preds = []
labels_to_idx = {
    'plausible': 0, 'co-occurrence_location': 1, 'co-occurrence_rotation': 2, 'gravity': 3, 'intersection': 4,
    'size': 5, 'pose': 6, 'undecided': 7
}
for image_id in majority_vote_mcc:
    labels.append(labels_to_idx[get_category_key(False, images_gt[image_id])])
    preds.append(labels_to_idx[majority_vote_mcc[image_id]])

mcc_cf_matrix = confusion_matrix(labels, preds)
plot_confusion_matrix(bc_cf_matrix, target_names=titles)