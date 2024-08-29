import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm.notebook import tqdm
def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel.
    Args:
        size (int): The size of the kernel (size x size).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        np.array: 2D Gaussian kernel.
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel

def apply_psf_to_grid(cell, sigma):
    """Apply a Gaussian PSF to the cell grid.
    Args:
        cell (np.array): The cell grid.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        np.array: Updated cell grid.
    """
    size = 2 * int(3 * sigma) + 1  # Ensure the kernel size is large enough
    kernel = gaussian_kernel(size, sigma)
    kernel_sum = np.sum(kernel)
    blurred_cell = gaussian_filter(cell, sigma=sigma)
    return blurred_cell


def simple_binding2(cell_x = None, cell_y = None, pixel_size = None, O = None, K = None, Ns = None, operator_pixel = None, num_trials = None, truncate = False, average_trials=False, use_binom=False, use_psf=False, sig= 0.0847, num_sites = 2):
    
    '''This function simulates a cell with deterministic binding. It takes in the following parameters:
    cell_x: Cell size in microns
    cell_y: Cell size in microns
    pixel_size: Pixel size in microns
    n_pixels_x: Number of pixels in the x direction
    n_pixels_y: Number of pixels in the y direction
    n_pixels: Number of pixels in the cell
    O: Number of operators or binding sites
    K: Dissociation constant in units of proteins per pixel
    Ns: Number of proteins P in the cell
    operator_pixel: Location of the operator in the cell in pixel coordinates
    It returns the following results:
    means: The mean number of proteins in the cell
    vars: The variance of the number of proteins in the cell
    cell_grids: The number of proteins in each pixel of the cell
    bound_fraction: The fraction of proteins that are bound
    bound_molecules: The number of bound molecules
    '''

    # Cell Parameters
    if cell_x == None:
        cell_x = 1 # Cell size in microns
    if cell_y == None:
        cell_y = 2  # Cell size in microns
    if pixel_size == None:    
        pixel_size = 0.044 # Pixel size in microns
    if num_trials == None:
        num_trials = 5 # Number of trials to run
    n_pixels_x = round(cell_x/pixel_size) # Number of pixels in the x direction
    n_pixels_y = round(cell_y/pixel_size) # Number of pixels in the y direction
    n_pixels = n_pixels_x * n_pixels_y # Number of pixels in the cell

    # Reaction parameters
    if O == None:
        O = 20 # Number of operators or binding sites
    if K == None:
        K = 0.2 # Dissociation constant in units of proteins per pixel
    if Ns == None:
        Ns =  range(0, 1001, 1) # Number of proteins P in the cell

    # Location of binding sites. Generate random coordinates based on the num_sites parameter.
    operator_pixels_x = np.random.randint(0, n_pixels_x, (num_sites))
    operator_pixels_y = np.random.randint(0, n_pixels_y, (num_sites))


    # Initialize arrays to store the results. All the arrays should have an index N and then another sub index for the trial number.This is because we will run multiple trials for each N.
    
    means = np.zeros((len(Ns), num_trials))
    vars = np.zeros((len(Ns), num_trials))
    cell_grids = np.zeros(((len(Ns), num_trials, n_pixels_x, n_pixels_y)))
    bound_fractions = np.zeros((len(Ns), num_trials))
    bound_molecules_arr = np.zeros((len(Ns), num_trials))
    means_psf = np.zeros((len(Ns), num_trials))
    vars_psf = np.zeros((len(Ns), num_trials))
    cell_grids_psf = np.zeros(((len(Ns), num_trials, n_pixels_x, n_pixels_y)))
    bound_fractions_psf = np.zeros((len(Ns), num_trials))
    bound_molecules_arr_psf = np.zeros((len(Ns), num_trials))

    for N in Ns:
        for trial in range(num_trials):
            # Begin by initializing the cell grid at each trial
            cell = np.zeros((n_pixels_x, n_pixels_y))
            
            # Calculate the concentration of proteins in the cell
            P = N / n_pixels # in units of proteins per pixel
            # Calculate the bound fraction
            bound_fraction = P / (P + K) 
            # Calculate the number of bound molecules by multiplying the bound fraction by the number of operators, since each operator can bind one protein and the bound fraction tells us the number of bound operators based on the concentration of proteins in the cell.
            if truncate == False and use_binom == True:    
                max_trials = min(N, O)
                bound_molecules = np.random.binomial(max_trials, bound_fraction)
            elif truncate == True:
                bound_molecules = round(bound_fraction * O)
            elif truncate == False and use_binom == False:
                bound_molecules = bound_fraction * O
            # Calculate the number of free molecules.
            if bound_molecules < N and N != 0:
                free_molecules = N - bound_molecules
            else:
                free_molecules = 0
            if free_molecules > 0:
                free_molecules = round(free_molecules)
                x_coords = np.random.randint(0, n_pixels_x, (free_molecules))
                y_coords = np.random.randint(0, n_pixels_y, (free_molecules))
                np.add.at(cell, (x_coords, y_coords), 1)
            else:
                pass
                # Distribute the bound molecules to the operator
            # We have an if statement because the bound fraction can be greater than 1, which means that there are more bound molecules than binding sites. In this case, we just set the number of bound molecules to the number of operators.
            if bound_molecules < N:
                np.add.at(cell, (operator_pixels_x, operator_pixels_y), bound_molecules/num_sites)
            elif bound_molecules > O:
                np.add.at(cell, (operator_pixels_x, operator_pixels_y), O/num_sites)
            if use_psf == True:
                # Apply PSF to grid using gaussian_filter
                cell_psf = apply_psf_to_grid(cell, sigma=sig/pixel_size) # Adjust sigma as needed, 84.7 nm is the standard deviation of the PSF from a paper, ruoyou got it from aisha.
                '''
                if N == 900 and trial == 1:
                    print(f'Sigma in pixels: {sig/pixel_size}')
                    print(f'Sigma in microns: {sig}')
                '''
                means_psf[N, trial] = np.mean(cell_psf)
                vars_psf[N, trial] = np.var(cell_psf)
                cell_grids_psf[N, trial] = cell_psf
                bound_fractions_psf[N, trial] = bound_fraction
                bound_molecules_arr_psf[N, trial] = bound_molecules
            # Calculate and store the cell grid and other results
            mean = np.mean(cell)
            var = np.var(cell)
            means[N, trial] = mean
            vars[N, trial] = var
            cell_grids[N, trial] = cell
            bound_fractions[N, trial] = bound_fraction
            bound_molecules_arr[N, trial] = bound_molecules
    # return a dictionary with the resultant arrays
    if average_trials == True:
        means = np.mean(means, axis=1)
        vars = np.mean(vars, axis=1)
        cell_grids = np.mean(cell_grids, axis=1)
        bound_fractions = np.mean(bound_fractions, axis=1)
        bound_molecules_arr = np.mean(bound_molecules_arr, axis=1)
        means_psf = np.mean(means_psf, axis=1)
        vars_psf = np.mean(vars_psf, axis=1)
        cell_grids_psf = np.mean(cell_grids_psf, axis=1)
        bound_fractions_psf = np.mean(bound_fractions_psf, axis=1)
        bound_molecules_arr_psf = np.mean(bound_molecules_arr_psf, axis=1)
    return {'means': means, 
            'vars': vars, 
            'cell_grids': cell_grids, 
            'bound_fractions': bound_fractions,
            'bound_molecules': bound_molecules_arr, 
            'operator_pixel': operator_pixel, 
            'means_psf': means_psf, 
            'vars_psf': vars_psf, 
            'cell_grids_psf': cell_grids_psf, 
            'bound_fractions_psf': bound_fractions_psf, 
            'bound_molecules_psf': bound_molecules_arr_psf}

#example call
test2 = simple_binding2(O= 40, use_binom = True, K = 0.01, average_trials = True, num_trials=10, use_psf = True)
print(test2['means'])