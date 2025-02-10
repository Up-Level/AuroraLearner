import csv
import sys
import time
import numpy as np

sys.path.append("../auroralearner")
import png

def plot_image(x: np.ndarray, ys: np.ndarray, y_size = 32):
    plot = np.zeros((x.shape[0], y_size), dtype=int)
    colour_increment = 255 // (ys.shape[0])
    colour = colour_increment
    for y in ys:
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_vals = (x - x_min) / x_range * (x.shape[0] - 1)
        y_vals = (y - y_min) / y_range * (y_size - 1)

        for i in range(x_vals.shape[0]):
            plot[int(x_vals[i]), int(y_vals[i])] = colour
        colour += colour_increment
    
    return plot

labels = ['Date_UTC', 'SML', 'SMU', 'GSE_Bx', 'GSE_By', 'GSE_Bz', 'GSE_Vx', 'GSE_Vy', 'GSE_Vz', 'CLOCK_ANGLE_GSE']
data = {}
for label in labels:
    data[label] = list()

with open("polar/20241211-14-51-supermag.csv", mode="r") as file:
    csvFile = csv.reader(file)
    # Consume first row of file with labels
    next(csvFile)

    prevLine = np.zeros_like(labels, dtype=np.float32)
    for line in csvFile:
        for j, label in enumerate(labels):
            if j == 0:
                # Convert date to unix epoch
                data[label].append(time.mktime(time.strptime(line[j], "%Y-%m-%d %H:%M:%S")))
            else:
                value = float(line[j])

                # If the value is invalid (=999999) then use the previous valid value
                data[label].append(value if value - 999998 < 0 else prevLine[j])
                prevLine[j] = data[label][-1]

for label in labels:
    data[label] = np.array(data[label])

graph = plot_image(data["Date_UTC"][:128], np.array([data["SML"], data["SMU"], data["GSE_Bx"], data["GSE_By"], data["GSE_Bz"]])[:, :128], y_size=32)
"""graph_str = ""
for x in range(graph.shape[0]):
    for y in range(graph.shape[1]):
        if graph[x, y] == 0:
            graph_str += " "
        else:
            graph_str += str(graph[x, y])
        
    graph_str += "\n"""

writer = png.Writer(graph.shape[0], graph.shape[1], greyscale=True)
with open(f"polar/graph.png", "wb") as file:
        writer.write(file, graph.T.tolist())