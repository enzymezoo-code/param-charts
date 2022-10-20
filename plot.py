!pip install tinydb

from types import SimpleNamespace
import torch

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import os
import tinydb
import itertools
import pandas as pd
import tempfile
from PIL import Image
import torchvision.transforms.functional as TF
import uuid
import numpy as np


################################################################################
# Define functions for reading from the database and displaying images in a grid
################################################################################
        
def get_images(display_x, display_y, 
                database_filepath,
                static_match_parameters=None, 
                ignore_these_vars=None,
                print_close_matches=False,
                save_plot=False,
                test_output_path=None):
    """
    display_x: tuple (parameter_name_str, parameter_value_list)
               Variable to appear on the horizontal axis
    display_y: tuple (parameter_name_str, parameter_value_list)
               Variable to appear on the vertical axis
    static_match_parameters: dict {parameter_name_str : parameter_value}
               Variables must match these values
               If a variable is omitted from static_match_parameters, it still 
               ensures all images have all matching parameters unless the param 
               is in ignore_these_vars
    ignore_these_vars: list of strings
               Variable names to ignore whether they all match in this plot
               It's recommended that only variables that are completely 
               irrelevant go here
    print_close_matches: bool Prints the files that almost match
    save_plot: Save the plot to a file and enter the plot information to the database
    test_output_path: Directory to save the plot image in test_output_path/plots/*.png
    """
    if ignore_these_vars is None:
        ignore_these_vars = []

    if static_match_parameters is None:
        static_match_parameters = {}

    if test_output_path is None:
        test_output_path = os.path.dirname(database_filepath)

    ###
    # Look for images in the database file that fit the parameters
    ###
    entry_list = [] # list of tuples of (file_name, parameter_dict)
    print("Started Reading JSON database")
    if not os.path.exists(database_filepath):
        raise Exception(f'JSON Database file does not exist: {database_filepath}')

    whole_db = tinydb.TinyDB(database_filepath, sort_keys=True, indent=2)
    db = whole_db.table('images', cache_size=None)

    x_name = display_x[0]
    y_name = display_y[0]
    x_vals = display_x[1]
    y_vals = display_y[1]

    if print_close_matches:
        print([_[x_name]for _ in db.all()])
        print([_[y_name]for _ in db.all()])

    requirements = static_match_parameters.copy()

    display_img_list = []
    for y_val in y_vals:
        for x_val in x_vals:
            requirements[x_name] = x_val
            requirements[y_name] = y_val
            matches = db.search(tinydb.Query().fragment(requirements))
            if len(matches) == 1:
                stored_image = matches[0]
            elif len(matches) > 1:
                print(f"Multiple images found for {x_name}: {x_val}, {y_name}: {y_val}")
                # Get the parameter names that differ between examples (for easy reading)
                # (We would like to use set(), but we can't use set() with all data types)
                mismatch_keys = []
                none_keys = []
                all_keys = list(set([k for match in matches for k in match.keys()]))
                # Add None to keys that are missing in different entries (reduces probability of error on lookup)
                for key in all_keys:
                    for i,match in enumerate(matches):
                        if key not in match.keys():
                            matches[i][key] = None
                # Which parameters don't match in any of the x/y matches
                for match in matches[1:]:
                    for key in all_keys:
                        if match[key] != matches[0][key]:
                            mismatch_keys += [key]
                mismatch_keys = list(set(mismatch_keys)) # remove duplicates
                mismatch_keys = [k for k in mismatch_keys if k not in ignore_these_vars] # remove variables we're ignoring
                for i_match, match in enumerate(matches):
                    mismatch_dict = {k:match[k] for k in mismatch_keys}
                    none_dict = {k:(match[k] if k in match.keys() else None) for k in none_keys}
                    # print(f"({i_match}): {mismatch_dict} {none_dict}   all parameters: {match}")
                    print(f"({i_match}): {mismatch_dict} {none_dict}")
                i_image_select = input("Please enter the image number to use:\n")
                print(f'You entered {i_image_select}')
                stored_image = matches[int(i_image_select)]
            else:
                print(f"No images found for {x_name}: {x_val}, {y_name}: {y_val}")
                if print_close_matches:
                    matches = db.search(tinydb.Query().fragment({x_name: x_val, y_name: y_val}))
                    print(f"Images in the database that match {x_name} and {y_name}: ")
                    for match in matches:
                        mismatch = {}
                        for key,value in requirements.items():
                            if key in match.keys() and match[key] != value:
                                mismatch[key] = match[key]
                            elif key not in match.keys():
                                mismatch[key] = None
                        
                        print(f"{match}")
                        print(f"  Non-matching parameters:   {mismatch}")
                stored_image = None

            # print(f"Match: {stored_image}")
            print(f"Match found")
            display_img_list += [stored_image]

            # Ensure all other vars match for this plot, except ignore_these_vars
            images_in_list = [_ for _ in display_img_list if _ is not None]
            if len(images_in_list) == 1: # first image in the graph
                for key,value in images_in_list[0].items():
                    if key not in ignore_these_vars:
                        requirements[key] = value

    if all([_ is None for _ in display_img_list]):
        raise Exception(f"No images found for {display_x}, {display_y}, {static_match_parameters}")

    whole_db.close()

    return display_img_list, requirements

###
# Make display grid
###
def plot_grid(img_list, display_x, display_y, database_filepath, plot_resize=None, save_plot=False, test_output_path=None, requirements=None):
    #TODO Plot should pad or resize images when using different image resolutions or aspect ratios

    """
    plot_resize: float Percent to resize the image grid in the output plot.
    requirements: dictionary of variables that were used to make these images
    """
    if plot_resize is None:
        plot_resize = 1.0

    if test_output_path is None:
        test_output_path = os.path.dirname(database_filepath)

    # img_list: list of image file names to plot (elements can be None)
    #                   len(img_list) = len(display_x) * len(display_y)
    x_name = display_x[0]
    y_name = display_y[0]
    x_vals = display_x[1]
    y_vals = display_y[1]

    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    display_tensor_list = []
    for fname in img_list:
        if fname is not None:
            display_tensor_list += [transform_to_tensor(Image.open(fname))]
        else: # image with these values was not found
            display_tensor_list += [None]
    # for parameter_dict in img_list:
    #     if parameter_dict is not None:
    #         img_file_name = parameter_dict["img_file_name"]
    #         display_tensor_list += [transform_to_tensor(Image.open(img_file_name))]
    #     else: # image with these values was not found
    #         display_tensor_list += [None]

    # Set the None tensors to a blank tensor of the appropriate shape
    tensor_shape = max(dt.shape for dt in display_tensor_list if dt is not None)
    for i_tens,tens in enumerate(display_tensor_list):
        if display_tensor_list[i_tens] is None:
            display_tensor_list[i_tens] = torch.zeros(tensor_shape)

    # Build grid image
    ncolumns = len(x_vals)
    nrows = len(y_vals)
    whole_img_grid = TF.to_pil_image(utils.make_grid(display_tensor_list, ncolumns, normalize=False).cpu())

    # Save grid image
    hexstring = uuid.uuid4().hex # for unique file names
    os.makedirs(os.path.join(test_output_path,'grid'), exist_ok=True)
    grid_image_filename = os.path.join(test_output_path,'grid',f'test_{hexstring}.png')
    whole_img_grid.save(grid_image_filename)
    
    ###
    # Add grid to axis
    ###
    plt.style.use('dark_background')

    grid_size = whole_img_grid.size
    dpi = plt.rcParams['figure.dpi']  # pixels per inch
    actual_size = [_/dpi for _ in grid_size]
    plt.figure(dpi=dpi)

    # Add labels
    ticks = np.arange(grid_size[0]/ncolumns/2, grid_size[0], grid_size[0]/ncolumns)
    labels = x_vals
    plt.xticks(ticks, labels, fontsize=70*plot_resize)
    plt.xlabel(x_name, fontsize=90*plot_resize)

    ticks = np.arange(grid_size[1]/nrows/2, grid_size[1], grid_size[1]/nrows)
    labels = y_vals
    plt.yticks(ticks, labels, fontsize=70*plot_resize)
    plt.ylabel(y_name, fontsize=90*plot_resize)

    # Plot images at the actual size
    def set_ax_size(axsize, ax=None):
        # Sets the axis size regardless of figure size
        # axsize: (width, height) in pixels
        # Credit to ImportanceOfBeingErnest on StackOverflow
        w,h = axsize
        if not ax:
            ax = plt.gca()
        # positions of axis edges
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)

    if plot_resize == 1.0:
        axsize = actual_size
    else:
        axsize = [s*plot_resize for s in actual_size]
    set_ax_size(axsize)

    plt.imshow(whole_img_grid)

    ###
    # Save plot to file
    ###
    if save_plot:
        plot_dir = os.path.join(test_output_path,'plots')
        os.makedirs(plot_dir, exist_ok=True)
        # Save Figure
        plot_file_path = f'{plot_dir}/plot_{hexstring}.png'
        plt.savefig(plot_file_path, bbox_inches='tight')
        
        plt.show()

        # Save parameters for figure in the database 
        vars_to_save = {
            "axes":{x_name:x_vals,y_name:y_vals},
            "static_match_parameters":static_match_parameters,
            "ignore_these_vars":ignore_these_vars,
            "plot_resize":plot_resize,
            "all_parameters":requirements,
            "plot_file_path":plot_file_path,
            "grid_image_filename":grid_image_filename
        }
        save_parameters(database_filepath, vars_to_save, 'plots')


# Function to save database entry
def save_parameters(database_filepath, vars_to_save, table_name):
    """
    database_filepath (str): Path to the database json file
    vars_to_save: dictionary of variables to save in the database
    table_name: TinyDB database table to save into
    """
    
    database_lockdir = f'{database_filepath}.lock'
    os.makedirs(database_lockdir, exist_ok=True)

    # Save settings
    # Writing to database is robust to multiple simultaneous writes
    timeout = time.time() + 60*5   # 5 minutes to timeout
    while True:
        if len(os.listdir(database_lockdir)) == 0: # database not locked
            # Lock the database so other running processes won't clobber things
            with tempfile.NamedTemporaryFile(dir=database_lockdir) as lockfile:
                # Write to database
                with tinydb.TinyDB(database_filepath, sort_keys=True, indent=2) as db:
                    images_table = db.table(table_name)
                    images_table.insert(vars_to_save)
            break
        else:
            time.sleep(5) # Wait 5 seconds
        if time.time() > timeout:
            raise Exception(f"Could not write to database. Database is locked. You may need to remove the lock file in {database_lockdir}")
    
    print(f"Saved settings to {database_filepath}")

################################################################################
################################################################################
# Looping functions
################################################################################

def get_save_args(args, omit=None):
    if not isinstance(args,dict):
        args_dict = args.__dict__
    else:
        args_dict = args
    #   save_vars = {}
    #   for key in variables_to_loop.keys():
    #     save_vars[key] = args_dict[key] # only save vars in variables_to_loop
    #   save_vars = {key:( val if not isinstance(val, pd.Series) else val.to_dict() )
    #                 for key,val in save_vars.items()}
    #   save_vars = {key:val for key,val in save_vars.items() if key != "text_prompts" and key != "image_prompts"}
    #   save_vars["text_prompts_str"] = " + ".join([s for prompt_lst in args_dict["text_prompts"].values() for s in prompt_lst])
    #   save_vars["image_prompts_str"] = " + ".join([s for prompt_lst in args_dict["image_prompts"].values() for s in prompt_lst])

    save_vars = args_dict.copy()
    # remove variable formats that can't be saved into a json file
    save_vars = {key:( val if not isinstance(val, pd.Series) else val.to_dict() )
                for key,val in save_vars.items()}

    # remove keys in the omit list
    if omit is not None:
        save_vars = {key:val for key,val in save_vars.items() if key not in omit}

    return save_vars
################################################################################

def loop_parameters(variables_to_loop, database_filepath, default_args, render_img_fn, img_fname_fn, skip_duplicates=True, ignore_these_vars=None):
    """
    variables_to_loop (dict): Dictionary where key is a named variables and value is a list of values
    database_filepath (str): path to the json database file
    default_args (SimpleNamespace): default args
    render_img_fn: function that takes takes an args namespace as input and generates an image
    img_fname_fn: function that takes takes an args namespace as input and gives the filename of the image that will be saved
    skip_duplicates: skip the image render if it is already in the database
    """

    # Parse variables_to_loop 
    param_possibilities = [[[key,v] for v in value] for key,value in variables_to_loop.items()]
    param_combos = list(itertools.product(*param_possibilities))
    print(f"Running {len(param_combos)} different combinations of parameters.")

    args = SimpleNamespace(**vars(default_args)) # Shallow copy of default namespace

    for i_param, these_params in enumerate(param_combos):
        # Define parameters we are looping through
        #   for param_set in these_params:
        #       globals()[param_set[0]] = param_set[1] # sets parameter as a global variable
        print(f'Parameter loop (total {len(param_combos)}): {i_param+1}')
        print('New Parameters:')
        for param_set in these_params:
            setattr(args, param_set[0], param_set[1])
            print(param_set[0], param_set[1])

        # Option to skip run if the parameter combination is already in the database
        if skip_duplicates:
            with tinydb.TinyDB(database_filepath, sort_keys=True, indent=2) as whole_db:
                db = whole_db.table('images', cache_size=None)
                vars_saved = get_save_args(args, omit=ignore_these_vars)
                # print("vars_saved",json.dumps(vars_saved, sort_keys=True, indent=2))
                matches = db.search(tinydb.Query().fragment(vars_saved))
                print("matches",matches)
            if len(matches) > 0: # We already ran these parameters, so don't run again.
                print("Run already completed. Skipping...")
                continue 

        render_img_fn(args)  

        vars_to_save = get_save_args(args)
        vars_to_save['img_file_name'] = img_fname_fn(args)
        save_parameters(database_filepath, vars_to_save, table_name='images')

############################################################

# Define the function that renders a single image
# for functions that take a dictionary instead of a namespace
# render_img_fn = lambda namespace_args: render_image_batch(vars(namespace_args)) 
# render_image_batch is a function that takes a namespace as input
render_img_fn = render_image_batch

database_filename = "deforum_database.json"
test_output_dir = os.path.join(output_path,'databases')

os.makedirs(f'{test_output_dir}', exist_ok=True)
database_filepath = os.path.join(test_output_dir, database_filename)


def get_fname(args):
    from helpers.prompt import sanitize
    index = 0
    if args.filename_format == "{timestring}_{index}_{prompt}.png":
        filename = f"{args.timestring}_{index:05}_{sanitize(args.prompt)[:160]}.png"
    else:
        filename = f"{args.timestring}_{index:05}_{args.seed}.png"
    filepath = os.path.join(args.outdir, filename)
    return filepath

img_fname_fn = get_fname


#
# Specific to deforum as is
#TODO for deforum, wrap the root, args, and animationargs in a function call
#
def modify_on_loop(args, root):
    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    args.seed_behavior = 'fixed'

    # Load clip model if using clip guidance
    if (args.clip_loss_scale > 0) or (args.aesthetics_loss_scale > 0):
        root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(device)
        if (args.aesthetics_loss_scale > 0):
            root.aesthetics_model = load_aesthetics_model(args, root)

    # Ensure only one image is generated for these settings
    if (args.use_init and os.path.isdir(args.init_image)) or len(prompts) > 1 or args.n_batch > 1:
        raise Exception("Arguments must allow only one image to be generated.\n" +
                        "Please ensure init_image is not a directory, number of prompts is 1, and n_batch is 1. Current default args:\n" +
                        f"use_init = {args.use_init}\n" + 
                        f"init_image = {args.init_image}\n" + 
                        f"len(prompts) = {len(prompts)}\n" + 
                        f"n_batch = {args.n_batch}\n" )
    # TODO Either make it impossible to use variables that result in more than one image, or account for this in the img_fname_fn
    print("Warning: Do not use variables that will result in more then one image begin generated. It will not record the correct filename.")

    return args, root

default_args, root = modify_on_loop(args, root)

# Define the function that renders a single image
# for functions that take a dictionary instead of a namespace
# render_img_fn = lambda namespace_args: render_image_batch(vars(namespace_args)) 
# render_image_batch is a function that takes a namespace as input
def render_img_fn(args):
    global root
    args, root_args = modify_on_loop(args, root) 
    render_image_batch(args, prompts, root)

#
####################################################################################
# Change these to loop and plot
####################################################################################
#

# What are we generating?
variables_to_loop = {
    'seed':[1003124998],
    'scale':[7],
    'sampler':['euler'],
    'steps':[50],
    'gradient_wrt':["x0_pred"], # ["x", "x0_pred"]
    'colormatch_loss_scale':[0,2000,20000],#[0,2000,5000,10000],
    'grad_threshold_type':["mean"], # ["dynamic", "static", "mean", "schedule"],
    'ignore_sat_scale':[1,2,3],
    'clamp_grad_threshold':[0.1],
    'clamp_start':[0.2],
    'clamp_stop':[0.01],
    'aesthetics_loss_scale':[0],
    'clip_name':["ViT-L/14"],#["ViT-B/32","ViT-B/16","ViT-L/14","ViT-L/14@336px"]
}

# What are we plotting?
# display_x = ('steps',[10,20,30,40,50,60,70])
# display_y = ('scale',[3,7,11,15,19,23])
# display_x = ('gradient_wrt',["x", "x0_pred"])
# display_x = ('grad_threshold_type',["dynamic", "static", "mean"])
# display_y = ('aesthetics_loss_scale',[0,10,25,50,100])
display_y = ('ignore_sat_scale',[1,2,3])
display_x = ('colormatch_loss_scale',[0,2000,20000])

####################################################################################
####################################################################################

# Extra settings
save_plot = True
test_output_path = output_path

static_match_parameters = {key:val[0] for key,val in variables_to_loop.items() if len(val)==1}
ignore_these_vars = ['timestring', 'img_file_name']

#####################
# Do the runs
#####################

loop_parameters(variables_to_loop, 
                database_filepath, 
                default_args, 
                render_img_fn=render_img_fn,
                img_fname_fn=img_fname_fn,
                skip_duplicates=True,
                ignore_these_vars=ignore_these_vars)


img_dict_list, requirements = get_images(display_x, display_y, 
                            database_filepath,
                            static_match_parameters=static_match_parameters, 
                            ignore_these_vars=ignore_these_vars,
                            print_close_matches=False,
                            save_plot=save_plot,
                            test_output_path=test_output_path)

img_fnames = [d["img_file_name"] if d is not None else None for d in img_dict_list]
plot_grid(img_fnames, display_x, display_y, database_filepath, plot_resize=None, save_plot=True, test_output_path=None, requirements=None)
