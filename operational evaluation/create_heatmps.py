import numpy as np
import matplotlib
import matplotlib.pyplot as plt

augs = ["RandomFlip", "RandomCrop", "CutMix", "MixUp"]

fa_opixray_data = np.array([[88.6, 88.7, 88.7, 88.7],
                    [87.7, 88.3, 88.6, 88.7],
                    [88.3, 88.3, 88.3, 88.3],
                    [88.1, 88.4, 88.4, 88.8]])
fa_opixray_black_cells =[[1,0]]

cr_opixray_data = np.array([[86.1, 86.7, 85.5, 86.2],
                    [86.1, 86.1, 86.1, 86.4],
                    [86.4, 86.7, 85.4, 86.0],
                    [85.4, 86.3, 85.8, 86.7]])
cr_opixray_black_cells = [[3, 0], [0,2], [2,2]]

fa_sixray_data = np.array([[84.4, 84.1, 84.1, 84.9],
                    [83.9, 84.6, 83.7, 84.1],
                    [83.2, 83.6, 84.0, 84.1],
                    [83.6, 84.2, 83.5, 84.1]])
fa_sixray_black_cells = [[2, 0]]

cr_sixray_data = np.array([[83.3, 82.8, 83.6, 83.1],
                    [82.6, 83.3, 82.8, 83.3],
                    [82.9, 83.5, 83.0, 82.6],
                    [83.1, 82.9, 83.2, 83.1]])
cr_sixray_black_cells = [[1,0], [2,3]]

data = cr_sixray_data
black_cells = cr_sixray_black_cells

matplotlib.rcParams.update({'font.size': 15})
fig, ax = plt.subplots()
im = ax.imshow(data, cmap="Greens")


# We want to show all ticks...
ax.set_xticks(np.arange(len(augs)))
ax.set_yticks(np.arange(len(augs)))
# ... and label them with the respective list entries
ax.set_xticklabels(augs)
ax.set_yticklabels(augs)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(augs)):
    for j in range(len(augs)):
        text = ax.text(j, i, data[i, j],
                       ha="center", va="center", color="w")



for cell in black_cells:
    text = ax.text(cell[1], cell[0], data[cell[0], cell[1]], ha="center", va="center", color="black")

#ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()