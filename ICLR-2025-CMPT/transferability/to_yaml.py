import os
import csv
import yaml


def frozen_prompt_or_projection():
    for mode in ('ProjectionTransfer', 'FrozenPromptTransfer'):

        with open('Logs/csv/{}-New.csv'.format(mode)) as f:

            rows = []
            for row in csv.reader(f, skipinitialspace=True):
                rows.append(row)

        del row

        final_dict = {}
        for idx in range(1, len(rows[0])):

            tgt_task = rows[0][idx]

            final_dict['roberta_to_{}'.format(tgt_task.lower())] = {}

            for row in rows[1:]:
                final_dict['roberta_to_{}'.format(tgt_task.lower())][row[0]] = row[idx].split('±')[0]

        with open('transferability/ground_truths/{}.yaml'.format(mode), 'w') as f:
            yaml.dump(data=final_dict, stream=f, allow_unicode=True)


# def attention_projection():
#     tgt_tasks = ('caltech101', 'cifar', 'dtd', 'oxford_flowers102', 'oxford_iiit_pet', 'sun397', 'svhn',
#                  'patch_camelyon', 'resisc45', 'eurosat', 'diabetic_retinopathy',
#                  'dmlab', 'kitti', 'smallnorb_azi', 'smallnorb_ele', 'dsprites_loc', 'dsprites_ori', 'clevr_dist',
#                  'clevr_count')
#
#     final_dict = {}
#     for tgt_task in tgt_tasks:
#
#         if not os.path.exists('Logs/csv/AT-{}.csv'.format(tgt_task)):
#             continue
#
#         final_dict['roberta_to_{}'.format(tgt_task.lower())] = {}
#
#         with open('Logs/csv/AT-{}.csv'.format(tgt_task)) as f:
#             rows = []
#             for row in csv.reader(f, skipinitialspace=True):
#                 rows.append(row)
#
#         for row in rows[1:]:
#             key = row[0]
#             value = row[1:]
#
#             value = [x.split('±')[0] for x in value]
#
#             final_dict['roberta_to_{}'.format(tgt_task.lower())][key] = value
#
#     with open('transferability/ground_truths/AttentionTransfer.yaml', 'w') as f:
#         yaml.dump(data=final_dict, stream=f, default_flow_style=None, allow_unicode=True)


if __name__ == '__main__':

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    for file_name in os.listdir('transferability/ground_truths/'):
        if file_name.endswith('yaml'):
            os.remove(os.path.join('transferability/ground_truths/', file_name))

    frozen_prompt_or_projection()
    # attention_projection()
