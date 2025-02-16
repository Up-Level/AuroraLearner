import csv
import sys
import os
import time
import numpy as np
from PIL import Image
from PIL.Image import Resampling

TRAIN_FRAC = 0.8
IMAGE_SIZE = 120
PLOT_HEIGHT = 16

def plot_image(x: np.ndarray, ys: np.ndarray, y_size = 32):
    plot = np.zeros((x.shape[0], y_size), dtype=np.uint8)
    colour_increment = 255 // (ys.shape[0])
    colour = colour_increment

    for y in ys:
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_vals = ((x - x_min) / (x_range + 1) * (x.shape[0] - 1)).astype(int)
        y_vals = ((y - y_min) / (y_range + 1) * (y_size - 1)).astype(int)

        prev_value = y_vals[0]
        for i in range(x_vals.shape[0]):
            if prev_value != y_vals[i]:
                for y in range(y_vals[i], prev_value, np.sign(prev_value - y_vals[i])):
                    plot[x_vals[i], y] = colour
            else:
                plot[x_vals[i], y_vals[i]] = colour

            prev_value = y_vals[i]
        colour += colour_increment
    
    return plot

def process_indices():
    files = ["1996.csv", "1997.csv", "1998.csv", "1999.csv", "2000.csv"]
    labels = ['Date_UTC', 'SML', 'SMU', 'GSE_Bx', 'GSE_By', 'GSE_Bz', 'GSE_Vx', 'GSE_Vy', 'GSE_Vz', 'CLOCK_ANGLE_GSE']
    data = {}
    for label in labels:
        data[label] = list()

    for file in files:
        with open(f"polar/indices/{file}", mode="r") as openFile:
            csvFile = csv.reader(openFile)
            # Consume first row of file, which only contains data labels
            next(csvFile)

            prevLine = np.zeros_like(labels, dtype=np.float32)
            for line in csvFile:
                for j, label in enumerate(labels):
                    # First index is the date
                    if j == 0:
                        # Convert date to unix epoch
                        data[label].append(time.mktime(time.strptime(line[j], "%Y-%m-%d %H:%M:%S")))
                    else:
                        value = float(line[j])

                        # If the value is invalid (=999999) then use the last-known valid value
                        data[label].append(value if value - 999998 < 0 else prevLine[j])
                        prevLine[j] = data[label][-1]

    for label in labels:
        data[label] = np.array(data[label])

    return data

def process_images():
    sequence_files = os.listdir("polar/images_raw")
    sequences = list()
    timestamps = list()

    for sequence_file in sequence_files:
        image_files = os.listdir(f"polar/images_raw/{sequence_file}")
        sequence = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        seq_timestamps = np.zeros((len(image_files)))

        for i, image_file in enumerate(image_files):
            image = Image.open(f"polar/images_raw/{sequence_file}/{image_file}")

            # Raw images have been upscaled by 2x for some reason. This resizes them to their correct resolution
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Resampling.NEAREST)
            # Take just the R channel as the image is greyscale
            sequence[i] = np.array(image.getdata(0)).reshape(IMAGE_SIZE, IMAGE_SIZE)
            image.close()

            seq_timestamps[i] = time.mktime(time.strptime(image_file, "%Y%m%d_%H%M%S_a.gif"))
        
        sequences.append(sequence)
        timestamps.append(seq_timestamps)
    
    return np.array(sequences, dtype=object), np.array(timestamps, dtype=object)

def main():
    if len(sys.argv) < 2:
        print("1 argument required. Valid arguments are 'indices' or 'images'.")
        return

    match sys.argv[1]:
        case "indices":
            indices = process_indices()
            np.savez_compressed("polar/indices.npz", **indices)
        case "images":
            indices: dict = np.load("polar/indices.npz")
            sequences, timestamps = process_images()

            try:
                size_plot = int(sys.argv[2])
            except:
                size_plot = IMAGE_SIZE - 1

            dataset = list()

            for i in range(sequences.shape[0]):
                dataset.append(np.zeros((sequences[i].shape[0], IMAGE_SIZE + PLOT_HEIGHT, IMAGE_SIZE + PLOT_HEIGHT), np.uint8))
                for j in range(sequences[i].shape[0]):
                    # Finds the values in indices which are closest to the timestamp of the image
                    closest_index = np.argmin(np.abs(timestamps[i][j] - indices["Date_UTC"])) + 1
                    start_index = max(closest_index - size_plot, 0)

                    graph = plot_image(indices["Date_UTC"][start_index:closest_index],
                                       np.array([indices["SML"]])[:, start_index:closest_index],
                                       y_size=PLOT_HEIGHT)
                    graph = np.pad(graph, [(0, IMAGE_SIZE + PLOT_HEIGHT - (closest_index-start_index)), (0, 0)])

                    sequence = np.pad(sequences[i][j], [(0, PLOT_HEIGHT), (0, 0)])

                    dataset[i][j] = np.concatenate([sequence, graph], axis=1).T
                    Image.fromarray(dataset[i][j], "L").save(f"polar/images/{i}_{j}.png")
                
                print(f"{i + 1}/{sequences.shape[0]} - {(i + 1) * 100 // sequences.shape[0]}%")
            
            dataset = np.array(dataset, dtype=object)
            train_index = int(dataset.shape[0] * TRAIN_FRAC)
            np.random.shuffle(dataset)
            np.savez_compressed("polar/dataset-train.npz", dataset[:train_index])
            np.savez_compressed("polar/dataset-test.npz",  dataset[ train_index:dataset.shape[0]])
        case _:
            print("Invalid argument. Valid arguments are 'indices' or 'images'.")

if __name__ == "__main__":
    main()