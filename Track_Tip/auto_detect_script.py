import os
import time
import threading
import csv
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
stack = []
download_path = os.path.join(os.environ["USERPROFILE"], "Downloads")

from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def combine_imgs(img1_path, img2_path, output_path):
    """Combine two images side-by-side while preserving original sizes."""
    # Open both images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Compute combined dimensions
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)

    # Create blank canvas (black background)
    combined = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))

    # Vertically center images if heights differ
    y1 = (max_height - img1.height) // 2
    y2 = (max_height - img2.height) // 2

    # Paste both images side by side
    combined.paste(img1, (0, y1))
    combined.paste(img2, (img1.width, y2))

    # Save result
    combined.save(output_path)


def plot_voltage(file,ys,xs): # return array
    # load voltage as 512*512*4 array
    array = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for value in row[2:]:
                array.append((float(value)))
    array = np.array(array).reshape((512,512,4))
    # only retain values with circle with r = 256
    mask = np.zeros((512,512), dtype=bool)
    for i in range(512):
        for j in range(512):
            r2 = (i - 256) ** 2 + (j - 256) ** 2
            if r2 <= (256) ** 2:
                mask[i, j] = True
    array[~mask] = np.nan
    

    
    plt.figure(figsize=(5.12, 5.12))
    plt.imshow(array[:,:,0])
    # plot ys xs onto image
    ax = plt.gca()
    for (x, y) in zip(xs, ys):
        circ = plt.Circle((x, y), radius=10, edgecolor='red', fill=False, linewidth=0.8)
        ax.add_patch(circ)
    plt.savefig(file.replace(".csv",".png"))
    plt.close()
    return

def findSN(file,plot = True):

    #csv.field_size_limit(sys.maxsize)

    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        idx = 0
        for row in reader:
            if idx == 0:
                FTE = float(row[0])
            else:  
                for i in row:
                    data.append(int(i))
            idx += 1
    data_ = np.array(data).reshape((512, 512, 4))

    # change all [1,1,1,1] pixels near circle radius=512 to [1,0,0,0]
    epsilon = 100 * 10
    for i in range(data_.shape[0]):
        for j in range(data_.shape[1]):
            r2 = (i - 256) ** 2 + (j - 256) ** 2
            if (256) ** 2 - epsilon <= r2 <= (256) ** 2 + epsilon:
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
        plt.figure(figsize=(5.12, 5.12))

        # (b) Grouped (DBSCAN)
        ax = plt.gca()
        ax.imshow(data_[..., :3] * 512, interpolation='nearest')

        cmap = plt.cm.get_cmap('hsv', num_clusters + 1)
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       s=30, color=cmap(label), label=f'Group {label}', alpha=0.8)

        ax.legend(fontsize=8, loc='upper right')
        ax.set_title(f"Grouped Clusters (DBSCAN), FTE = {FTE:.3f}")
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(file.replace(".csv", ".png"))
        plt.close()

        
    return num_clusters,FTE,ys,xs

def match_pattern(filename: str) -> bool:
    """Check if filename contains all required substrings in correct order."""
    key_parts = ["tip_omega_", "_IC", "_mask", "ms" , ".csv"]
    idx = 0
    for part in key_parts:
        next_pos = filename.find(part, idx)
        if next_pos == -1:
            return False
        idx = next_pos + len(part)
    return True


def watch_folder(interval=2):
    """Continuously scan current folder for new matching files."""
    seen = set()
    # initilize a csv to store results
    csv_file = "results_summary.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "Num_Clusters", "FTE"])    
    print("Watching download folder for files matching pattern...")
    download_path = os.path.join(os.environ["USERPROFILE"], "Downloads")
    while True:
        for fname in os.listdir(download_path):
            if fname not in seen and match_pattern(fname):
                seen.add(fname)
                stack.append(os.path.join(download_path, fname))
                print(f"[Watcher] Detected: {fname} -> pushed to stack")
        time.sleep(interval)


def handle_stack(interval=1):
    """Continuously process files from the stack (LIFO pop)."""
    while True:
        if stack:
            fname = stack.pop()
            fcolor_fname = fname.replace("tip", "fcolor")
            fcolor_fname = fcolor_fname.replace("248ms","249ms")  # for 250 ms
            num_clusters,FTE,ys,xs = findSN(fname)
            plot_voltage(fcolor_fname,ys,xs)
            combine_imgs(fcolor_fname.replace(".csv",".png"),fname.replace(".csv",".png"),fname.replace(".csv","_combined.png"))
            # del the fname file
            os.remove(fcolor_fname)
            os.remove(fname)
            os.remove(fname.replace(".csv",".png"))
            os.remove(fcolor_fname.replace(".csv",".png"))
            # append results to csv
            with open("results_summary.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([fname, num_clusters,FTE])
            
        else:
            time.sleep(interval)


if __name__ == "__main__":
    # run both watcher and handler in parallel threads
    t1 = threading.Thread(target=watch_folder, daemon=True)
    t2 = threading.Thread(target=handle_stack, daemon=True)

    t1.start()
    t2.start()

    # keep main thread alive
    while True:
        time.sleep(10)
