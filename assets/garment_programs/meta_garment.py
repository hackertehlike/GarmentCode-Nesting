from assets.garment_programs.tee import *
from assets.garment_programs.godet import *
from assets.garment_programs.bodice import *
from assets.garment_programs.pants import *
from assets.garment_programs.bands import *
from assets.garment_programs.skirt_paneled import *
from assets.garment_programs.skirt_levels import *
from assets.garment_programs.circle_skirt import *
from assets.garment_programs.sleeves import *

class TotalLengthError(BaseException):
    """Error indicating that the total length of a garment goes beyond 
    the floor length for a given person"""
    pass

class IncorrectElementConfiguration(BaseException):
    """Error indicating that given pattern is an empty garment"""
    pass

class MetaGarment(pyg.Component):
    """Meta garment component
        Depending on parameter values it can generate sewing patterns
    for various dresses and jumpsuit styles and fit them to the body
    measurements
    """
    def __init__(self, name, body, design) -> None:
        super().__init__(name)
        self.body = body
        self.design = design

        # Elements
        self.upper_name = design['meta']['upper']['v']
        self.lower_name = design['meta']['bottom']['v']
        self.belt_name = design['meta']['wb']['v']

        # Upper garment
        if self.upper_name: 
            upper = globals()[self.upper_name]
            self.subs = [upper(body, design)]

            # Set a label
            self.subs[-1].set_panel_label('body', overwrite=False)

        # Define Lower garment
        if self.lower_name:
            Lower_class = globals()[self.lower_name]
            # NOTE: full rise for fitted tops
            Lower = Lower_class(body, design, rise=1. if self.upper_name and 'Fitted' in self.upper_name else None)
        else: 
            Lower = None

        # Belt (or not)
        # TODO Adapt the rise of the lower garment to the width of the belt for correct matching
        if self.belt_name:
            Belt_class = globals()[self.belt_name]
            
            # Adjust rise to match the Lower garment if needed
            Belt = Belt_class(body, design, Lower.get_rise() if Lower else 1.)

            self.subs.append(Belt)

            # Place below the upper garment 
            if len(self.subs) > 1:
                self.subs[-1].place_by_interface(
                    self.subs[-1].interfaces['top'],
                    self.subs[-2].interfaces['bottom'], 
                    gap=5
                )

                self.stitching_rules.append(
                    (self.subs[-2].interfaces['bottom'],
                     self.subs[-1].interfaces['top']))
            
            # Add waist label
            self.subs[-1].interfaces['top'].edges.propagate_label('lower_interface', append=True)
            # Set panel segmentation labels
            self.subs[-1].set_panel_label('body', overwrite=False)

        # Attach Lower garment if present
        if self.lower_name:
            self.subs.append(Lower)
            # Place below the upper garment or self.wb
            if len(self.subs) > 1:
                self.subs[-1].place_by_interface(
                    self.subs[-1].interfaces['top'],
                    self.subs[-2].interfaces['bottom'], 
                    gap=5
                )
                self.stitching_rules.append(
                    (self.subs[-2].interfaces['bottom'],
                     self.subs[-1].interfaces['top']))
            
            # Add waist label
            if not self.belt_name:
                self.subs[-1].interfaces['top'].edges.propagate_label('lower_interface', append=True)
            # Set panel segmentation labels
            self.subs[-1].set_panel_label('leg', overwrite=False)


    def get_panel_by_name(self, panel_name):
        """Retrieve a panel by its name
        
        This method searches for a panel with the given name in the following order:
        1. Checks if any direct subcomponent is a Panel with the requested name
        2. Checks if any subcomponent has an attribute with the requested name that is a Panel
        3. Recursively searches in any subcomponent that is itself a Component
        4. Looks in the assembled pattern's panels dictionary
        
        Args:
            panel_name (str): The name of the panel to retrieve (e.g., 'skirt_front')
            
        Returns:
            pyg.Panel or None: The panel with the given name if found, None otherwise
            
        Example:
            ```python
            # Get the front skirt panel
            front_panel = garment.get_panel_by_name('skirt_front')
            
            # Get a specific panel
            panel = garment.get_panel_by_name('skirt_back')
            ```
        """
        # First, try to find the panel directly in subcomponents
        # This handles both panel objects stored directly as attributes and
        # panels within subcomponents
        for component in self._get_subcomponents():
            # Check if the component is a Panel with the requested name
            if hasattr(component, 'name') and component.name == panel_name:
                return component
            
            # Check if the component has an attribute with the requested name
            if hasattr(component, panel_name):
                panel = getattr(component, panel_name)
                if isinstance(panel, pyg.Panel):
                    return panel
            
            # If the component is itself a Component, recursively search in it
            if hasattr(component, 'get_panel_by_name'):
                panel = component.get_panel_by_name(panel_name)
                if panel:
                    return panel
        
        # If not found through direct attributes, try through the assembled pattern
        pattern = self.assembly().pattern
        if 'panels' in pattern and panel_name in pattern['panels']:
            return pattern['panels'][panel_name]
        
        return None
    
    def get_all_panel_names(self):
        """Get the names of all panels in this garment
        
        This method collects panel names from:
        1. Direct subcomponents that are Panels
        2. Attributes of subcomponents that are Panels
        3. Recursively from any subcomponent that is itself a Component
        4. The assembled pattern's panels dictionary
        
        Returns:
            list: A list of unique panel names in the garment
            
        Example:
            ```python
            # Get all panel names
            panel_names = garment.get_all_panel_names()
            print(f"This garment has {len(panel_names)} panels: {', '.join(panel_names)}")
            ```
        """
        names = []
        
        # Get panels from component attributes
        for component in self._get_subcomponents():
            if hasattr(component, 'name') and isinstance(component, pyg.Panel):
                names.append(component.name)
            
            # If the component is itself a Component, recursively get panel names
            if hasattr(component, 'get_all_panel_names'):
                names.extend(component.get_all_panel_names())
            
            # For components like SkirtCircle that store panels as attributes
            for attr_name in dir(component):
                if attr_name.startswith('_') or attr_name in ('name', 'interfaces'):
                    continue
                    
                attr = getattr(component, attr_name)
                if isinstance(attr, pyg.Panel) and hasattr(attr, 'name'):
                    names.append(attr.name)
        
        # Get panels from the assembled pattern
        pattern = self.assembly().pattern
        if 'panels' in pattern:
            names.extend(list(pattern['panels'].keys()))
        
        # Return unique names
        return list(set(names))
        
    def assert_total_length(self, tol=1):
        """Check the total length of components"""
        # Check that the total length of the components are less that body height
        length = self.length()
        floor = self.body['height'] - self.body['head_l']
        if length > floor + tol:
            raise TotalLengthError(f'{self.__class__.__name__}::{self.name}::ERROR:'
                                    f':Total length {length} exceeds the floor length {floor}')
        
    # TODO these checks don't require initialization of the pattern!
    def assert_non_empty(self, filter_belts=True):
        """Check that the garment is non-empty
            * filter_wb -- if set, then garments consisting only of waistbands are considered empty
        """
        if not self.upper_name and not self.lower_name:
            if filter_belts or not self.belt_name:
                raise IncorrectElementConfiguration()
            
    def assert_skirt_waistband(self):
        """Check if a generated heavy skirt is created with a waistband"""

        if self.lower_name and self.lower_name in ['SkirtCircle', 'AsymmSkirtCircle', 'SkirtManyPanels']:
            if not (self.belt_name or self.upper_name):
                raise IncorrectElementConfiguration()