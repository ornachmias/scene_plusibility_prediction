import pandas as pd

csv_path = '../data/evaluation/model_preds_dist_v2.csv'
orig_csv = pd.read_csv(csv_path)

for model_name in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']:
    columns = ['category']
    categories_cols = [f'{model_name}_mcc_{x}' for x in list(range(7))]
    columns.extend(categories_cols)
    df = orig_csv[columns]
    df[categories_cols] = df[categories_cols].astype(float).round(3)
    df.rename(columns={f'{model_name}_mcc_{x}': f'C{x}' for x in list(range(7))}, inplace=True)
    df = df.fillna('-')
    df.to_csv(f'{model_name}_mcc.csv', float_format='%.3f', index=False)


columns = ['category']
for model_name in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']:
    categories_cols = [f'{model_name}_bc_{x}' for x in [0, 1]]
    columns.extend(categories_cols)

df = orig_csv[columns]
df[categories_cols] = df[categories_cols].astype(float).round(3)
new_col_names = {f'{x}_bc_0': f'{x}_Plausible' for x in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']}
new_col_names.update({f'{x}_bc_1': f'{x}_Implausible' for x in ['ViT', 'ResNet', 'CRTNet', 'CoAtNet']})
df.rename(columns=new_col_names, inplace=True)
df = df.fillna('-')
df.to_csv(f'bc.csv', float_format='%.3f', index=False)


vit_cf = {"table": {"1": 1389, "0": 1851}, "chair": {"0": 3548, "1": 4556}, "silverware": {"1": 3958, "0": 1464}, "bottle": {"1": 215, "0": 81}, "notebook": {"0": 165, "1": 73}, "monitor": {"0": 453, "1": 143}, "book": {"1": 227, "0": 175}, "desktop": {"1": 190, "0": 588}, "mouse": {"1": 34, "0": 108}, "lamp": {"1": 188, "0": 214}, "keyboard": {"1": 65, "0": 191}, "plate": {"1": 1462, "0": 1066}, "cup": {"1": 960, "0": 626}, "walldecoration": {"1": 183, "0": 149}, "closet": {"1": 296, "0": 210}, "bed": {"1": 90, "0": 70}, "speakers": {"0": 19, "1": 15}, "control": {"0": 17, "1": 27}, "plant": {"1": 109, "0": 21}, "bag": {"0": 102, "1": 102}, "box": {"0": 165, "1": 29}, "pillow": {"1": 40, "0": 138}, "shoes": {"0": 29, "1": 65}, "laptop": {"0": 23, "1": 31}, "couch": {"1": 107, "0": 81}, "mousepad": {"1": 5, "0": 7}, "toy": {"0": 17, "1": 9}}
resnet_cf = {"chair": {"0": 4088, "1": 4016}, "cup": {"0": 672, "1": 914}, "closet": {"0": 253, "1": 253}, "lamp": {"0": 216, "1": 186}, "walldecoration": {"0": 175, "1": 157}, "table": {"0": 2013, "1": 1227}, "pillow": {"0": 94, "1": 84}, "desktop": {"0": 547, "1": 231}, "couch": {"0": 121, "1": 67}, "monitor": {"0": 420, "1": 176}, "bed": {"0": 90, "1": 70}, "silverware": {"1": 3612, "0": 1810}, "plate": {"1": 1440, "0": 1088}, "bottle": {"1": 256, "0": 40}, "book": {"0": 200, "1": 202}, "shoes": {"1": 46, "0": 48}, "notebook": {"1": 93, "0": 145}, "plant": {"1": 88, "0": 42}, "bag": {"1": 68, "0": 136}, "keyboard": {"1": 81, "0": 175}, "mouse": {"0": 103, "1": 39}, "control": {"1": 15, "0": 29}, "box": {"0": 160, "1": 34}, "laptop": {"1": 18, "0": 36}, "speakers": {"0": 27, "1": 7}, "mousepad": {"1": 8, "0": 4}, "toy": {"0": 21, "1": 5}}
crtnet_cf = {"table": {"0": 1742, "1": 1498}, "notebook": {"0": 98, "1": 140}, "plate": {"1": 1317, "0": 1211}, "cup": {"1": 885, "0": 701}, "chair": {"0": 4009, "1": 4095}, "monitor": {"0": 318, "1": 278}, "keyboard": {"0": 126, "1": 130}, "silverware": {"0": 2286, "1": 3136}, "bag": {"1": 82, "0": 122}, "walldecoration": {"1": 192, "0": 140}, "plant": {"1": 91, "0": 39}, "desktop": {"1": 371, "0": 407}, "lamp": {"1": 225, "0": 177}, "couch": {"1": 92, "0": 96}, "book": {"0": 169, "1": 233}, "bed": {"0": 73, "1": 87}, "closet": {"0": 229, "1": 277}, "bottle": {"1": 183, "0": 113}, "mouse": {"1": 77, "0": 65}, "laptop": {"1": 27, "0": 27}, "pillow": {"1": 100, "0": 78}, "toy": {"1": 15, "0": 11}, "control": {"1": 24, "0": 20}, "box": {"1": 45, "0": 149}, "shoes": {"0": 40, "1": 54}, "speakers": {"0": 18, "1": 16}, "mousepad": {"1": 7, "0": 5}}
coatnet_cf = {"plate": {"1": 1062, "0": 1466}, "silverware": {"1": 3119, "0": 2303}, "table": {"1": 1453, "0": 1787}, "cup": {"0": 766, "1": 820}, "chair": {"0": 4183, "1": 3921}, "book": {"1": 255, "0": 147}, "keyboard": {"1": 123, "0": 133}, "monitor": {"0": 317, "1": 279}, "lamp": {"0": 152, "1": 250}, "bag": {"0": 145, "1": 59}, "bottle": {"1": 180, "0": 116}, "control": {"0": 31, "1": 13}, "desktop": {"0": 389, "1": 389}, "closet": {"1": 332, "0": 174}, "walldecoration": {"0": 87, "1": 245}, "bed": {"1": 88, "0": 72}, "plant": {"1": 73, "0": 57}, "notebook": {"1": 144, "0": 94}, "pillow": {"1": 117, "0": 61}, "couch": {"1": 89, "0": 99}, "box": {"1": 94, "0": 100}, "shoes": {"0": 40, "1": 54}, "mouse": {"0": 80, "1": 62}, "toy": {"1": 14, "0": 12}, "laptop": {"1": 47, "0": 7}, "mousepad": {"1": 11, "0": 1}, "speakers": {"1": 15, "0": 19}}

data = {}
for k in vit_cf.keys():
    for model_name, cf in [('ViT', vit_cf), ('ResNet', resnet_cf), ('CRTNet', crtnet_cf), ('CoAtNet', coatnet_cf)]:
        if k not in data:
            data[k] = {}

        s = cf[k]['0'] + cf[k]['1']
        data[k][f'{model_name}_Plausible'] = cf[k]['0'] / s
        data[k][f'{model_name}_Implausible'] = cf[k]['1'] / s

rows = []
for k in data.keys():
    row = {'Object Type': k}
    row.update(data[k])
    rows.append(row)

cf_df = pd.DataFrame(rows)
cf_df.to_csv('cf_categories_v2.csv', index=False, float_format='%.3f')
