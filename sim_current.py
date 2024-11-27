import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import concurrent.futures

def linear_fit(x,y):
    p,cov = np.polyfit(x,y,1,cov=True)
    slope, intercept = p
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    return slope, intercept, slope_err, intercept_err

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
    blurred_cell = gaussian_filter(cell, sigma)
    return blurred_cell

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



def get_I_spot(k,sigma_x, sigma_y,sigma_z=0 ,pixel_size=0.044, D=3, pixel_depth=0):
    if D == 2:
        I_spot = 4*np.pi*k*(sigma_x*sigma_y)/(pixel_size**2)
    elif D == 3:
        I_spot = 8*(np.pi**(3/2))*k*(sigma_x*sigma_y*sigma_z)/(pixel_depth*pixel_size**2)
    return I_spot

def simple_binding2(cell_x = None, cell_y = None, pixel_size = None, O = None, K = None, Ns = None, operator_pixel = None, num_trials = 5, truncate = False, average_trials=True, use_binom=True, use_psf=True, sig= 0.0847,I_spot=1, multiple_sites = True,I_spot2 = False, cooperativity = 1):
    
    '''This function simulates a cell with deterministic binding. It takes in the following parameters:

    cell_x (float): The size of the cell in the x direction in microns.
    cell_y (float): The size of the cell in the y direction in microns.
    pixel_size (float): The size of each pixel in microns.
    O (int): The number of operators or binding sites in the cell.
    K (float): The dissociation constant in units of proteins per pixel.
    Ns (list): A list of the number of proteins P in the cell.
    operator_pixel (tuple): The location of the operator in the cell in pixel coordinates.
    num_trials (int): The number of trials to run.
    truncate (bool): Whether to truncate the number of bound molecules to the number of operators.
    average_trials (bool): Whether to average the results over all trials.
    use_binom (bool): Whether to use the binomial distribution to calculate the number of bound molecules.
    use_psf (bool): Whether to apply the point spread function (PSF) to the cell grid.
    sig (float): The standard deviation of the PSF in microns.
    I_spot (float): The intensity of the spot in the cell grid. In units of photons per pixel
    cooperativity (int): The cooperativity of the binding. If cooperativity is greater than 1, the Hill equation is used to calculate the bound fraction. If cooperativity is 1, the bound fraction is calculated as P/(P+K), where P is the concentration of proteins in the cell and K is the dissociation constant.

    It returns the following results in a dictionary, where each key corresponds to an array of results:

    means (np.array): The mean intensity of the cell grid for each N and trial.
    vars (np.array): The variance of the cell grid for each N and trial.
    cell_grids (np.array): The cell grid for each N and trial.
    bound_fractions (np.array): The bound fraction for each N and trial.
    bound_molecules_arr (np.array): The number of bound molecules for each N and trial.
    means_psf (np.array): The mean intensity of the cell grid with PSF for each N and trial.
    vars_psf (np.array): The variance of the cell grid with PSF for each N and trial.
    cell_grids_psf (np.array): The cell grid with PSF for each N and trial.
    bound_fractions_psf (np.array): The bound fraction with PSF for each N and trial.
    bound_molecules_arr_psf (np.array): The number of bound molecules with PSF for each N and trial.

    '''

    # Cell Parameters
    if cell_x == None:
        cell_x = 2 # Cell size in microns
    if cell_y == None:
        cell_y = 1  # Cell size in microns
    if pixel_size == None:    
        pixel_size = 0.044 # Pixel size in microns
    if num_trials == None:
        num_trials = 10 # Number of trials to run
    n_pixels_x = round(cell_x/pixel_size) # Number of pixels in the x direction
    n_pixels_y = round(cell_y/pixel_size) # Number of pixels in the y direction
    n_pixels = n_pixels_x * n_pixels_y # Number of pixels in the cell

    # Reaction parameters
    if O == None:
        O = 20 # Number of operators or binding sites
    if K == None:
        K = 0.2 # Dissociation constant in units of proteins per pixel
    if Ns == None:
        Ns =  range(0, 101, 1) # Number of proteins P in the cell

    if multiple_sites == True:
        operator_pixel1 = (n_pixels_x//5, n_pixels_y//5) # Location of the operator in the cell in pixel coordinates
        operator_pixel2 = (n_pixels_x//2,n_pixels_y//2) # Location of the operator in the cell in pixel coordinates
        operator_pixel3 = (n_pixels_x//2, n_pixels_y//10) # Location of the operator in the cell in pixel coordinates
    elif multiple_sites == False:
        operator_pixel = (n_pixels_x//2, n_pixels_y//2)

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
    
    for N in tqdm(Ns):
        for trial in (range(num_trials)):
            # Begin by initializing the cell grid at each trial
            cell = np.zeros((n_pixels_x, n_pixels_y))
            
            # Calculate the concentration of proteins in the cell
            P = N / n_pixels # in units of proteins per pixel
            # Calculate the bound fraction
            bound_fraction = P**cooperativity / (P**cooperativity + K**cooperativity) 
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
                if I_spot2 == True:
                    np.add.at(cell, (x_coords, y_coords), I_spot)
                else:
                    np.add.at(cell, (x_coords, y_coords), 1)
            else:
                pass
                # Distribute the bound molecules to the operator
            # We have an if statement because the bound fraction can be greater than 1, which means that there are more bound molecules than operators. In this case, we just set the number of bound molecules to the number of operators.
            if bound_molecules < N:
                if multiple_sites == False:
                    cell[operator_pixel] += bound_molecules
                elif multiple_sites == True:
                    cell[operator_pixel1] += bound_molecules/3
                    cell[operator_pixel2] += bound_molecules/3
                    cell[operator_pixel3] += bound_molecules/3
            elif bound_molecules > O:
                if multiple_sites == False:
                    cell[operator_pixel] += O
                elif multiple_sites == True:
                    cell[operator_pixel1] += O/3
                    cell[operator_pixel2] += O/3
                    cell[operator_pixel3] += O/3
            if use_psf == True and I_spot2 == False:
                # Apply PSF to grid using gaussian_filter
                cell_psf = I_spot*apply_psf_to_grid(cell, sigma=sig/pixel_size,) # Adjust sigma as needed, 84.7 nm is the standard deviation of the PSF from a paper, ruoyou got it from aisha.
                # We multiple above by I_spot, it is effectively the Amplitude of the PSF.
            
                means_psf[N, trial] = np.mean(cell_psf)
                vars_psf[N, trial] = np.var(cell_psf)
                cell_grids_psf[N, trial] = cell_psf
                bound_fractions_psf[N, trial] = bound_fraction
                bound_molecules_arr_psf[N, trial] = bound_molecules
            elif use_psf == True and I_spot2 == True:
                # Apply PSF to grid using gaussian_filter
                cell_psf = apply_psf_to_grid(cell, sigma=sig/pixel_size,)
                # We multiple above by I_spot, it is effectively the Amplitude of the PSF.

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


    if average_trials == False:
        # Since averaging compiles all the results from the trials to one array, 
        # we should do that here as well for consistency.
        # Instead of averaging though, we will concatenate all the results into one array.
        means = np.concatenate(means, axis=1)
        vars = np.concatenate(vars, axis=1)
        cell_grids = np.concatenate(cell_grids, axis=1)
        bound_fractions = np.concatenate(bound_fractions, axis=1)
        bound_molecules_arr = np.concatenate(bound_molecules_arr, axis=1)
        
        means_psf = np.concatenate(means_psf, axis=1)
        vars_psf = np.concatenate(vars_psf, axis=1)
        cell_grids_psf = np.concatenate(cell_grids_psf, axis=1)
        bound_fractions_psf = np.concatenate(bound_fractions_psf, axis=1)
        bound_molecules_arr_psf = np.concatenate(bound_molecules_arr_psf, axis=1)



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

if __name__ == '__main__':
    results = simple_binding2()
    print(results.keys())
