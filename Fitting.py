def fit_data(datasets, plot=False, fit_report=False, plot_title=None, save_image=False, save_html=False):
    """
    Fit multiple datasets either linearly or non-linearly, and optionally plot and report the fit results.

    Parameters:
    -----------
    datasets : list of tuples
        Each tuple contains the following elements:
        - x_data: Array or list of x-values.
        - y_data: Array or list of y-values.
        - linear_fit: Boolean flag to indicate whether to perform a linear fit (True) or non-linear fit (False).
        - O_init: Initial guess for the parameter O (used in non-linear fitting).
        - K_init: Initial guess for the parameter K (used in non-linear fitting).
        - mask_up: Upper bound of the x-values to be used in the fit.
        - mask_down: Lower bound of the x-values to be used in the fit.
        - data_name: String representing the name of the dataset (used for labeling in the plot).
        - marker_color: String representing the color of the data points in the plot.
        - fit_color: String representing the color of the fit line in the plot.
        
    plot : bool, optional
        If True, the function will plot the data and the fit results using Plotly (default is False).
    
    fit_report : bool, optional
        If True, the function will print a report of the fit parameters for non-linear fitting (default is False).
    
    plot_title : str, optional
        The title of the plot (default is None).
    
    save_image : bool, optional
        If True, the function will save the plot as a PNG image with the plot title as name(default is False).

    save_html : bool, optional
        If True, the function will save the plot as an HTML file with the plot title as name(default is False).
    
    Returns:
    --------
    results : list
        A list containing the results for each dataset. Each element in the list is a tuple containing:
        - For linear fits: (y_fit, y_fit_err, slope, intercept, slope_err, intercept_err)
        - For non-linear fits: (y_fit, y_fit_err, O, K, O_err, K_err)
    """
    
    # Initialize the Plotly figure if plotting is enabled
    fig = go.Figure() if plot else None

    # Initialize a list to store the results for each dataset
    results = []
    
    # marker styles list 
    marker_style = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right', 'pentagon', 'hexagon', 'octagon', 'star', 'hexagram', 'star-triangle-up', 'star-triangle-down', 'star-square', 'star-diamond', 'diamond-tall', 'diamond-wide', 'hourglass', 'bowtie', 'circle-cross', 'circle-x', 'square-cross', 'square-x', 'diamond-cross', 'diamond-x', 'cross-thin', 'x-thin', 'asterisk', 'hash', 'y-up', 'y-down', 'y-left', 'y-right', 'line-ew', 'line-ns', 'line-ne', 'line-nw', 'arrow-up', 'arrow-down', 'arrow-left', 'arrow-right', 'triangle-ne', 'triangle-nw', 'triangle-se', 'triangle-sw', 'triangle', 'triangle-ew', 'triangle-ns', 'triangle-ne-open', 'triangle-se-open', 'triangle-sw-open', 'triangle-nw-open', 'pentagon-open', 'hexagon-open', 'octagon-open', 'star-open', 'hexagram-open', 'star-triangle-up-open', 'star-triangle-down-open', 'star-square-open', 'star-diamond-open', 'diamond-tall-open', 'diamond-wide-open', 'hourglass-open', 'bowtie-open', 'circle-cross-open', 'circle-x-open', 'square-cross-open', 'square-x-open', 'diamond-cross-open', 'diamond-x-open', 'cross-thin-open', 'x-thin-open', 'asterisk-open', 'hash-open', 'y-up-open', 'y-down-open', 'y-left-open', 'y-right-open', 'line-ew-open', 'line-ns-open', 'line-ne-open', 'line-nw-open', 'arrow-up-open', 'arrow-down-open', 'arrow-left-open', 'arrow-right-open', 'triangle-ne-open', 'triangle-nw-open', 'triangle-se-open', 'triangle-sw-open', 'triangle-open', 'triangle-ew-open', 'triangle-ns-open']
    # Iterate over each dataset in the input list
    for i, (x_data, y_data, linear_fit, O_init, K_init, mask_up, mask_down, data_name, marker_color, fit_color) in enumerate(datasets):
        # Select marker style from the list
        marker_style_choice = marker_style[i % len(marker_style)]
        # Perform a linear fit if linear_fit is True
        if linear_fit:
            # Apply the mask to limit the x-data to the specified range (mask_down, mask_up)
            mask = (x_data > mask_down) & (x_data < mask_up)
            x_data = x_data[mask]
            y_data = y_data[mask]
            
            # Perform a linear fit using numpy.polyfit and calculate the covariance matrix
            p, cov = np.polyfit(x_data, y_data, 1, cov=True)
            slope, intercept = p
            slope_err = np.sqrt(cov[0, 0])
            intercept_err = np.sqrt(cov[1, 1])
            
            # Calculate the fitted y-values and the associated errors
            y_fit = slope * x_data + intercept
            y_fit_err = np.sqrt((x_data * slope_err)**2 + intercept_err**2)
            
            # Plot the data and the linear fit if plotting is enabled
            if plot:
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers',marker=dict(symbol=marker_style_choice) ,name=f"{data_name}"))
                fig.add_trace(go.Scatter(x=x_data, y=y_fit, mode='lines', line=dict(color=fit_color),
                                         error_y=dict(type='data', array=y_fit_err, visible=True),
                                         name=f"y = ({slope:.3f} ± {slope_err:.3f})x + ({intercept:.3f} ± {intercept_err:.3f})"))
            
            # Append the results for this dataset to the results list
            results.append((y_fit, y_fit_err, slope, intercept, slope_err, intercept_err))
        
        # Perform a non-linear fit if linear_fit is False
        else:
            # Perform the non-linear fit using curve_fit with the initial guesses for O and K
            popt, pcov = curve_fit(get_variance, x_data, y_data, p0=[O_init, K_init], method='lm', maxfev=1000000)
            O, K = popt
            O_err, K_err = np.sqrt(np.diag(pcov))
            
            # Calculate the fitted y-values and the associated errors
            y_fit = get_variance(x_data, O, K)
            y_fit_err = np.sqrt(y_data)
            
            # Plot the data and the non-linear fit if plotting is enabled
            if plot:
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=f"{data_name}"))
                fig.add_trace(go.Scatter(x=x_data, y=y_fit, mode='markers', marker=dict(color=fit_color),
                                         name=f"O = {O:.3f} ± {O_err:.3f}, K = {K:.3f} ± {K_err:.3f}",
                                         error_y=dict(type='data', array=y_fit_err, visible=True)))
            
            # Print a fit report if fit_report is enabled
            if fit_report:
                print(f"O = {O:.3f} ± {O_err:.3f}, K = {K:.3f} ± {K_err:.3f}, chi2 = {np.sum((y_data - y_fit)**2)}, r2 = {sklearn.metrics.r2_score(y_data, y_fit)}")
            
            # Append the results for this dataset to the results list
            results.append((y_fit, y_fit_err, O, K, O_err, K_err))
    
    # Finalize the plot if plotting is enabled
    if plot:
        fig.update_layout({'title': f"{plot_title}"}, xaxis_title='Mean', yaxis_title='Pixel Variance')
        if save_image:
            fig.write_image(f"{plot_title}.png", scale=5, width=1000, height=500)
        if save_html:
            fig.write_html(f"{plot_title}.html")
        fig.show()
    
    # Return the results for all datasets
    return results
def fit_metrics(observed, fitted, errors=None):
    """
    Calculate both chi-squared and R-squared values for the fit, aligning arrays based on the shortest length.

    Parameters:
    -----------
    observed : array-like
        The observed data values.
    fitted : array-like
        The fitted data values.
    errors : array-like, optional
        The uncertainties or errors associated with the observed data values.
        If not provided, chi-squared is not calculated.

    Returns:
    --------
    chi2 : float or None
        The chi-squared value. Returns None if errors are not provided.
    r2 : float
        The R-squared value.
    """
    # Convert input arrays to numpy arrays
    observed = np.asarray(observed)
    fitted = np.asarray(fitted)
    
    # Determine the common length (shortest length) to align the arrays
    min_length = min(len(observed), len(fitted))
    
    if min_length == 0:
        raise ValueError("At least one of the arrays is empty.")
    
    # Truncate arrays to the common length
    aligned_observed = observed[:min_length]
    aligned_fitted = fitted[:min_length]
    
    # Calculate R-squared
    r2 = r2_score(aligned_observed, aligned_fitted)
    
    # Calculate chi-squared if errors are provided
    if errors is not None:
        errors = np.asarray(errors)
        
        if len(errors) != len(observed):
            raise ValueError("Errors array must have the same number of data points as the observed array.")
        
        # Truncate errors to the common length
        aligned_errors = errors[:min_length]
        
        # Calculate chi-squared
        chi2 = np.sum(((aligned_observed - aligned_fitted) / aligned_errors) ** 2)
        return chi2, r2
    
    return None, r2
def get_variance(x, O, K, exponent=2, n_pix=2585664):
    x_bound = (x**exponent)*O/((x**exponent) + (K**exponent))
    return (x + ((x_bound**2)/n_pix))*10.2022