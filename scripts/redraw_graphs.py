import os

import matplotlib
from matplotlib import pyplot as plt

results = {
    'types': {
        'ViT': {'gravity': 0.7199504337050805, 'intersection': 0.5907877169559412, 'co-occurrence_location': 0.6974789915966386, 'co-occurrence_rotation': 0.6453333333333333, 'size': 0.6305254378648875, 'pose': 0.529291553133515}, 'ResNet': {'gravity': 0.7881040892193308, 'intersection': 0.6675567423230975, 'co-occurrence_location': 0.7424969987995198, 'co-occurrence_rotation': 0.7453333333333333, 'size': 0.7518765638031694, 'pose': 0.6341961852861036}, 'CRTNet': {'gravity': 0.8115577889447236, 'intersection': 0.6915887850467289, 'co-occurrence_location': 0.7635054021608644, 'co-occurrence_rotation': 0.7946666666666666, 'size': 0.7693911592994161, 'pose': 0.7166212534059946}, 'CoAtNet': {'gravity': 0.7385377942998761, 'intersection': 0.6188251001335113, 'co-occurrence_location': 0.7016806722689075, 'co-occurrence_rotation': 0.714, 'size': 0.7147623019182652, 'pose': 0.6008174386920981}
    },
    'sizes': {'ViT': {'small': 0.47193347193347196, 'medium': 0.6146010186757216, 'large': 0.6563245823389021}, 'ResNet': {'small': 0.49376299376299376, 'medium': 0.6935483870967742, 'large': 0.7410501193317423}, 'CRTNet': {'small': 0.6361746361746362, 'medium': 0.7767402376910016, 'large': 0.7768496420047732}, 'CoAtNet': {'small': 0.5114345114345115, 'medium': 0.6273344651952462, 'large': 0.6575178997613366}},
    'amount': {'ViT': {1: 0.5813333333333334, 2: 0.6254512635379061, 3: 0.6527068437180796, 4: 0.6784855769230769, 5: 0.6883614088820827}, 'ResNet': {1: 0.6466666666666666, 2: 0.7138989169675091, 3: 0.7395301327885597, 4: 0.7854567307692307, 5: 0.7963246554364471}, 'CRTNet': {1: 0.7206178643384822, 2: 0.7558664259927798, 3: 0.7834525025536262, 4: 0.7824519230769231, 5: 0.8078101071975498}, 'CoAtNet': {1: 0.6026666666666667, 2: 0.6814079422382672, 3: 0.7104187946884576, 4: 0.7536057692307693, 5: 0.7572741194486983}}
}

bars_colors = ['royalblue', 'seagreen', 'darkorchid', 'palevioletred']


def generate_graph(ax, metrics, categories, output_name, labels):
    for i, model in enumerate(metrics):
        x = []
        y = []
        colors = []
        for j, category in enumerate(categories):
            y.append(metrics[model][category])
            x.append(j + (-0.2 + (0.2 * i)))
            colors.append(bars_colors[i])

        ax.bar(x, y, width=0.2, color=colors, label=model, align='center', edgecolor = "black", linewidth=0.8)

    ax.legend(loc='lower right')
    ax.set_xticks([x for x in range(len(categories))])
    ax.set_xticklabels(labels, rotation=40, ha='right')



font = {'family': 'normal',
        'size': 16}
matplotlib.rc('font', **font)
plt.rc('legend', fontsize=14)

# fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(18, 4))
#
# implausibility_types = ['gravity', 'intersection', 'co-occurrence_location', 'co-occurrence_rotation', 'size', 'pose']
# labels = ['Gravity', 'Intersection', 'Location', 'Rotation', 'Size', 'Pose']
# generate_graph(ax[0], results['types'], implausibility_types, 'types_bc.png', labels)
#
# sizes = ['small', 'medium', 'large']
# labels = ['Small', 'Medium', 'Large']
# generate_graph(ax[1], results['sizes'], sizes, 'sizes_bc.png', labels)
#
# amount = [1, 2, 3, 4, 5]
# generate_graph(ax[2], results['amount'], amount, 'amount_bc.png', amount)
# plt.tight_layout()
# plt.subplots_adjust(left=0.034, bottom=0.267, right=0.988, top=0.945, wspace=0.255, hspace=0.186)
# # plt.subplot_tool()
# plt.show()
# plt.savefig(os.path.join('../', 'all.png'))
results = {'ViT': {6: 0.22070844686648503, 3: 0.5449591280653951, 0: 0.4673024523160763, 1: 0.36239782016348776, 5: 0.37193460490463215, 2: 0.3555858310626703, 4: 0.18119891008174388}, 'ResNet': {0: 0.5817438692098093, 1: 0.5408719346049047, 6: 0.3692098092643052, 3: 0.7561307901907357, 2: 0.6335149863760218, 4: 0.4223433242506812, 5: 0.611716621253406}, 'CRTNet': {3: 0.7752043596730245, 4: 0.4550408719346049, 5: 0.6321525885558583, 1: 0.6021798365122616, 6: 0.5613079019073569, 2: 0.771117166212534, 0: 0.6553133514986376}, 'CoAtNet': {4: 0.2220708446866485, 5: 0.40735694822888285, 1: 0.3446866485013624, 0: 0.3773841961852861, 3: 0.6171662125340599, 6: 0.3637602179836512, 2: 0.47956403269754766}}
labels = ['Plausible', 'Location', 'Rotation', 'Gravity', 'Intersection', 'Size', 'Pose']
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 6))
for i, model in enumerate(results):
    x = []
    y = []
    colors = []
    for j, label in enumerate(labels):
        y.append(results[model][j])
        x.append(j + (-0.2 + (0.2 * i)))
        colors.append(bars_colors[i])

    ax.bar(x, y, width=0.2, color=colors, label=model, align='center', edgecolor = "black", linewidth=0.8)

ax.legend(loc='lower left')
ax.set_xticks([x for x in range(len(labels))])
ax.set_xticklabels(labels, rotation=40, ha='right')
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('../', 'mcc_v2.png'))
