import os
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

def findSN(file,plot = False):

    csv.field_size_limit(sys.maxsize)

    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for i in row:
                data.append(int(i))
    data_ = np.array(data).reshape((1024, 1024, 4))

    # change all [1,1,1,1] pixels near circle radius=512 to [1,0,0,0]
    epsilon = 100 * 512
    for i in range(data_.shape[0]):
        for j in range(data_.shape[1]):
            r2 = (i - 512) ** 2 + (j - 512) ** 2
            if (512) ** 2 - epsilon <= r2 <= (512) ** 2 + epsilon:
                if (data_[i, j] == [1, 1, 1, 1]).all():
                    data_[i, j] = [1, 0, 0, 0]

    # mask and coordinates
    mask = np.all(data_ == 1, axis=-1)
    mask_boundary = np.all(data_ == [1, 0, 0, 0], axis=-1)
    ys, xs = np.where(mask)
    ybs, xbs = np.where(mask_boundary)

    # ==================================================
    # 2. Cluster nearby red points
    # ==================================================
    if len(xs) == 0:
        print(file + " has no tip points.")
        labels = np.array([])
        num_clusters = 0
    else:
        points = np.column_stack((xs, ys))
        eps = 30
        clustering = DBSCAN(eps=eps, min_samples=1).fit(points)
        labels = clustering.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0) # -1 for noise

    if plot:
        # ==================================================
        # 3. Plot two subplots side by side
        # ==================================================
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # -----------------------
        # (a) Original map
        # -----------------------
        ax = axes[0]
        ax.imshow(data_[..., :3] * 512, interpolation='nearest')
        for (x, y) in zip(xs, ys):
            circ = plt.Circle((x, y), radius=10, color='red', fill=False, linewidth=0.8)
            ax.add_patch(circ)
        for (x, y) in zip(xbs, ybs):
            circ = plt.Circle((x, y), radius=10, color='blue', fill=False, linewidth=0.8)
            ax.add_patch(circ)
        center_circle = plt.Circle((512, 512), radius=512,
                                color='cyan', fill=False, linewidth=1.2, linestyle='--')
        ax.add_patch(center_circle)
        ax.invert_yaxis()
        ax.set_title("Original Map")

        # -----------------------
        # (b) Grouped (DBSCAN)
        # -----------------------
        ax = axes[1]
        ax.imshow(data_[..., :3] * 512, interpolation='nearest')
        cmap = plt.cm.get_cmap('hsv', num_clusters + 1)
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=30, color=cmap(label), label=f'Group {label}', alpha=0.8)
        ax.legend(fontsize=8, loc='upper right')
        ax.invert_yaxis()
        ax.set_title("Grouped Clusters (DBSCAN)")

        plt.tight_layout()
        plt.show()
    return num_clusters



def load_voltage(file): # return array
    # load voltage as 1024*1024*4 array
    array = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for value in row[2:]:
                array.append((float(value)))
    array = np.array(array).reshape((1024,1024,4))
    # only retain values with circle with r = 512
    mask = np.zeros((1024,1024), dtype=bool)
    for i in range(1024):
        for j in range(1024):
            r2 = (i - 512) ** 2 + (j - 512) ** 2
            if r2 <= (512) ** 2:
                mask[i, j] = True
    array[~mask] = np.nan
    return array

def watch_and_process(folder='.'):
    processed = set()
    print(f"Watching folder: {os.path.abspath(folder)}")

    while True:
        for filename in os.listdir(folder):
            if filename.startswith('fcolor_230ms') and filename.endswith('.csv'):
                filepath = os.path.join(folder, filename)

                if filepath not in processed:
                    try:
                        print(f"Detected new file: {filename}")
                        new_arr = load_voltage(filepath)[:,:,0]

                        # --- delete old file and rewrite ---
                        os.remove(filepath)
                        np.savez_compressed(filepath.replace('.csv', '.npz'), arr=new_arr)
                        print(f"Processed and replaced: {filename}")

                        arr = np.load(filepath.replace('.csv', '.npz'))['arr']
                        print(np.array_equal(arr, new_arr,equal_nan=True))  # verify
                        processed.add(filepath)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

        time.sleep(2)  # check every 2 seconds

if __name__ == '__main__':
    watch_and_process("C:\\Users\\xan37\\Downloads\\data_1")