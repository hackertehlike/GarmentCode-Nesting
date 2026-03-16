# Running Nesting From GarmentCode

Run GarmentCode GUI with
```
python gui.py
```
as before. Create your garment, then click the NEST button on the left toolbar to launch the Nesting GUI.

> NOTE: The 3D view may not work correctly on this version. If you wish to use this feature, run GarmentCode as on its original branch or on  `garmentcode.ethz.ch` and download the pattern, then run the nesting module individually as described below.


# Running Nesting Without GarmentCode

Run 
```
python -m nesting.gui
```
to launch the Nesting GUI. By default it will be loaded with the pattern specified in `nesting/config.py`. You can upload an external pattern as a JSON file -- if you wish to use body and design parameters, update `default_pattern_path`, `default_design_param_path` and `default_body_param_path` in SystemConfig in the config file.

# Running the Nesting Algorithms
The algorithms may take a while to return a layout, and in this time the window will disconnect. Do **not** close the tab or click other buttons! The window will automatically refresh to display the final solution.