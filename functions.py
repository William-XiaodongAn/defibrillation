import csv
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from scipy import stats

def linear_interpolation(y0,y1,x0,x1,x):
    
    return (y0 * (x1-x) + y1 * (x-x0)) / (x1-x0)

def load_voltage(path_to_csv):
    # ---- load voltage data from csv file ----
    data = []
    end_of_first_dt = False
    path = path_to_csv
    #path = r"F://result/tau_pw_350_tau_d_0.406_spatial_20e4_32e4_8_8_APD_step_4.csv"

    number_of_points = 0
    end_of_first_dt = False

    data = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:
            for i in range(len(row)):
                for j in row[i]:
                    if j == ';':
                        row[i] = row[i].replace(';', '')
                        if end_of_first_dt == False:
                            end_of_first_dt = True
                            number_of_points = i
                        
                if row[i] != '' and i%4 != 0:
                    data.append(float(row[i]))    
    max_value = max(data)
    min_value = min(data)
    # if we input whole texture
    if (number_of_points * 3 / 4).is_integer():
        number_of_points = int( number_of_points * 3 / 4)
    else:
        print('error! non-integar number of points')
    # if we input whole texture

    if (len(data)/number_of_points).is_integer():
        rows, cols=number_of_points,int(len(data)/number_of_points)
    else:
        print('error! non-integar time steps',len(data),number_of_points) 
        
    voltage = np.zeros([rows,cols])

    for i in range(rows):
        for j in range(cols):

            voltage[i][j] = data[j * rows +  i]

    return voltage

def get_APD_DI_BCL(arr,dt, APD_percentage = 0.3): # arr is 2D array, each row is a time series of one pixel
    APD = []
    for i in range(len(arr)):
        APD.append([])

        APD_value = 0
        start_apd = False
        for j in range(len(arr[0])):


            if start_apd == False and arr[i][j] > APD_percentage:
                start_apd = True
                if j == 0:
                    APD_start_error = 0
                else:
                    APD_start_error = linear_interpolation(1,0,arr[i][j],arr[i][j-1],APD_percentage)
                    APD_start_error = 1 - APD_start_error
                
            if start_apd == True and arr[i][j] <= APD_percentage: 
                start_apd = False
                APD_end_error = linear_interpolation(0,-1,arr[i][j],arr[i][j-1],APD_percentage)
                
                APD_value += (APD_start_error + APD_end_error) * dt                
                APD[i].append(APD_value)
                APD_value = 0
                APD_start_error = -100
                APD_end_error = -100
            if start_apd == True:
                APD_value += dt

    # the first DI is either not-complete or missing, so do not use it in restitutional curve

    DI = []            
    for i in range(len(arr)):
        DI.append([])

        DI_value = 0
        start_di = False
        for j in range(len(arr[0])):


            if start_di == False and arr[i][j] <= APD_percentage:
                start_di = True
                DI_start_error = linear_interpolation(0,-1,arr[i][j],arr[i][j-1],APD_percentage)
                
            if start_di == True and arr[i][j] > APD_percentage: 
                start_di = False
                DI_end_error = linear_interpolation(1,0,arr[i][j],arr[i][j-1],APD_percentage)
                DI_end_error = 1 - DI_end_error
                DI_value += (DI_start_error + DI_end_error) * dt                 
                DI[i].append(DI_value)
                DI_value = 0
                DI_start_error = -100
                DI_end_error = -100 
            if start_di == True:
                DI_value += dt

        if arr[i][-1] > APD_percentage:
            try:
                del DI[i][-1]
            except IndexError:
                pass

    X1 = []
    X2 = [] 
    BCL = []
    for i in range(len(arr)):
        if arr[i][0] > APD_percentage:
            for j in range(2,len(APD[i])):
                X1.append(DI[i][j-1])
                X2.append(APD[i][j])
                BCL.append(DI[i][j-1] + APD[i][j])
        else:
            for j in range(2,len(APD[i])):
                X1.append(DI[i][j])
                X2.append(APD[i][j])    
                BCL.append(DI[i][j] + APD[i][j])        
    return X2,X1,BCL # APD, DI ,BCL

def read_output_dat_d2(path):
    """
    Reads files like:
      #center= 2496
      #dim= 1
      <x> <y>
      ...
      #dim= 2
      <x> <y>
      ...
    Returns: dict[int, (np.ndarray x, np.ndarray y)]
    """
    dims = {}
    cur = None
    xs, ys = [], []

    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#dim="):
                # flush previous block
                if cur is not None and xs:
                    dims[cur] = (np.asarray(xs, float), np.asarray(ys, float))
                # start new block
                cur = int(s.split("=")[1])
                xs, ys = [], []
                continue
            if s.startswith("#center="):
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    xs.append(x); ys.append(y)
                except ValueError:
                    pass

    # flush last block
    if cur is not None and xs:
        dims[cur] = (np.asarray(xs, float), np.asarray(ys, float))
    return dims
def read_output_dat_lya(file_path):
    output = {}
    x_data = []
    y_data = []
    current_epsilon = None
    current_dim = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue # Skip empty lines

            # Check if the line is a header
            if line.startswith('#'):
                # Reset data lists for the new block
                x_data = []
                y_data = []
                
                # Parse the new epsilon value from the header line
                pattern = r"epsilon=\s*(?P<epsilon>[0-9.eE+-]+)\s*dim=\s*(?P<dim>\d+)"

                # --- Perform a single search ---
                match = re.search(pattern, line)
                if match:       
                    current_epsilon = float(match.group("epsilon"))
                    current_dim = int(match.group("dim"))
            else:
                # If it's a data line, parse it
                parts = line.split()
                x_data.append(int(parts[0]))
                y_data.append(float(parts[1]))
                output[(current_epsilon, current_dim)] = (np.array(x_data), np.array(y_data))
    return output


def plot_lya(voltage,x1 = 0,x2 = 10,save_fig = False, fig_nam = "D2_plot.png"):
    '''
    voltage, x1, x2,fig_num: voltage signal, x1 and x2 are the range to calculate plateau average
    calculate D2 plateau average from library of (logx,yy) per dimension and update the horizontal line of average plateau
    '''
    with open("output.dat", "w") as f:
        for val in voltage:
            f.write(f"{val}\n")

    proc = subprocess.Popen(
        ["lyap_k", "output.dat", "-M3","-m3"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in proc.stdout:
        print(line, end="")   # prints each line as lyap_k produces it    
    # --- usage ---
    output_dict = read_output_dat_lya("output.dat.lyap")   # <- put your path here

    plt.figure()
    max_epsilon = max(epsilon for epsilon,d in output_dict)
    x, y = output_dict[(max_epsilon,3)]
    xx = x
    yy   = y
    mask = (xx >= x1) & (xx <= x2)
    
    plt.plot(x, y, label=f"epsilon {max_epsilon}, dim 3")  
    
    x_points = [xx[mask][0], xx[mask][-1]]
    y_points = [yy[mask][0], yy[mask][-1]]
    # calculate average slope between x1 and x2 and plot it
    slope = stats.linregress(xx[mask], yy[mask]).slope
    # plot the slope line
    plt.plot(x_points, y_points, 'k--', label=f"Slope ≈ {slope:.3f}")
    return slope

def cal_D2(voltage,x1 = -6,x2 = -2,save_fig = False, fig_nam = "D2_plot.png"):
    '''
    voltage, x1, x2,fig_num: voltage signal, x1 and x2 are the range to calculate plateau average
    calculate D2 plateau average from library of (logx,yy) per dimension and update the horizontal line of average plateau
    '''
    with open("output.dat", "w") as f:
        for val in voltage:
            f.write(f"{val}\n")

    proc = subprocess.Popen(
        ["d2", "output.dat", "-M1,6", "-t50", "-N0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in proc.stdout:
        print(line, end="")   # prints each line as lyap_k produces it    
    # --- usage ---
    dims = read_output_dat_d2("output.dat.d2")   # <- put your path here

    plt.figure()
    plateaus = []   # collect per-dim plateau values
    output = {}    
    for d in sorted(dims):
        if d < 3: 
            continue
        x, y = dims[d]
        m = x > 0  # only positive for log
        logx = np.log(x[m])
        yy   = y[m]
        mask = (logx >= x1) & (logx <= x2)
        plateau = yy[mask].mean()
        plateaus.append(plateau)
        output[d] = (logx, yy)
        
        plt.plot(logx, yy, label=f"dim {d}")    

    plateau_average = np.mean(plateaus)
    
    plt.hlines(plateau_average, -7.2, 0, linestyles="--", colors="k")
    plt.title(f"D$_2$ vs log($\epsilon$) (D$_2$ plateau ≈ {plateau_average:.3f})")
    plt.xlabel("log(x)")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_nam, dpi=300, bbox_inches='tight')
    plt.show()
    return output,plateau_average # library of (logx,yy) per dimension, with key = dim and figure number


def delay_embed(x, m=3, tau=1):
    """
    Time-delay embed a 1D array x into R^m with delay tau.
    Returns an array of shape (N - (m-1)*tau, m).
    """
    x = np.asarray(x).reshape(-1)
    N = x.size - (m - 1) * tau
    if N <= 0:
        raise ValueError("Series too short for given m and tau.")
    # columns: [x(t), x(t+tau), ..., x(t+(m-1)tau)]
    return np.column_stack([x[i:i+N] for i in range(0, m*tau, tau)])

def plot_phase_from_1d(x, m=3, tau=1, zscore=True, two_d=True):
    """
    Build embedding and draw phase portrait(s).
    Returns:
        fig3d, fig2d (if two_d=True)
    """
    X = delay_embed(x, m=m, tau=tau)

    # 3D trajectory
    fig3d = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig3d.add_subplot(projection='3d')
    ax.plot(X[:,0], X[:,1], X[:,2], linewidth=0.6, alpha=0.9)
    ax.set_xlabel('x(t)')
    ax.set_ylabel(f'x(t+{tau})')
    ax.set_zlabel(f'x(t+{2*tau})')
    ax.set_title(f'Time-delay embedding (m={m}, τ={tau})')
    ax.grid(False)

    fig2d = None
    if two_d:
        fig2d, ax2 = plt.subplots(figsize=(4, 4), dpi=200)
        ax2.plot(X[:,0], X[:,1], linewidth=0.5, alpha=0.35)
        ax2.set_xlabel('x(t)')
        ax2.set_ylabel(f'x(t+{tau})')
        ax2.set_aspect('equal', 'box')
        ax2.set_title('2D projection')
        ax2.axis('off')

    plt.show(block=False)  # non-blocking show
    return fig3d, fig2d
