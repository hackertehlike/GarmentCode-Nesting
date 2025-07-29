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
            self.subs[-1].interfaces['top'].edges.propagate_label('lower_interface')
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
                self.subs[-1].interfaces['top'].edges.propagate_label('lower_interface')
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
        # pattern = self.assembly().pattern
        # if 'panels' in pattern and panel_name in pattern['panels']:
        #     return pattern['panels'][panel_name]
        
        return None
    
    def get_all_panel_names(self):
        """Get the names of all panels in this garment
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

    def split_panel(self, panel_name, proportion=0.5):
        """Split a panel into two subpieces and replace it in the component hierarchy.
        
        This method takes a panel name, finds the panel, splits it using the panel's split method,
        and then replaces the original panel with the two new subpanels in the component structure.
        This ensures that the assembly() method will use the new subpanels instead of the original.
        
            ```
        """
        # find the panel
        panel = self.get_panel_by_name(panel_name)

        # debug the semantic labels
        #if hasattr(panel, 'edges'):
        #    print(f"[MetaGarment.split_panel] Starting split for panel {panel.name}")
        #    print([edge.semantic_labels for edge in panel.edges])

        print(f"Attempting to split panel {panel_name} with proportion {proportion}")
        if not panel:
            raise ValueError(f"Panel {panel_name} not found")
            
        # check if the panel has a split method
        if not hasattr(panel, 'split'):
            raise ValueError(f"Panel {panel_name} does not support splitting")
            
        # split the panel
        subpanel1, subpanel2 = panel.split(proportion)
        print(f"Split panel {panel_name} into {subpanel1.name} and {subpanel2.name}")
        
        # find the parent component containing this panel
        parent_component = self._find_panel_parent(panel_name)
        if not parent_component:
            raise ValueError(f"Could not find parent component for panel {panel_name}")
            
        # replace the original panel with the  subpanels
        replaced = self._replace_panel_with_subpanels(parent_component, panel, [subpanel1, subpanel2])
        
        if replaced:
            # return the names of the new panels
            return [subpanel1.name, subpanel2.name]
        
        return None
        
    def _get_all_subcomponents_recursive(self):
        """Traverse the component tree and yield all subcomponents."""
        components_to_visit = list(self._get_subcomponents())
        visited = set()

        while components_to_visit:
            component = components_to_visit.pop(0)
            if component in visited:
                continue
            
            visited.add(component)
            yield component

            if hasattr(component, '_get_subcomponents'):
                # Add children to the list to be visited
                components_to_visit.extend(component._get_subcomponents())
        
    def _find_panel_parent(self, panel_name):
        """Find the component that directly contains the panel with the given name.
        
        Args:
            panel_name (str): The name of the panel to find the parent for
            
        Returns:
            Component: The component that contains the panel, or None if not found
        """
        # First check this component's direct attributes
        if hasattr(self, panel_name):
            attr = getattr(self, panel_name)
            if isinstance(attr, pyg.Panel) and attr.name == panel_name:
                return self
                
        # Then check all subcomponents
        for component in self._get_all_subcomponents_recursive():
            # Check if the component is itself the panel we're looking for
            print(f"Checking component {component.name} for panel {panel_name}")
            if hasattr(component, 'name') and component.name in panel_name:
                return component
                    
        return None
        
    def _replace_panel_with_subpanels(self, parent_component, original_panel, subpanels):
        """Replace a panel with its subpanels in the parent component.
        
        Args:
            parent_component: The component containing the original panel
            original_panel: The panel to replace
            subpanels: List of new panels to replace the original with
        """
        replaced = False
        
        # Case 1: Panel is stored as an attribute with a name matching the panel's name
        if hasattr(parent_component, original_panel.name):
            attr = getattr(parent_component, original_panel.name)
            if attr is original_panel:
                # Store the first subpanel under the same attribute
                setattr(parent_component, original_panel.name, subpanels[0])
                # And add any additional subpanels to subs
                if hasattr(parent_component, 'subs'):
                    for i, subpanel in enumerate(subpanels):
                        if i > 0:
                            if subpanel not in parent_component.subs:
                                parent_component.subs.append(subpanel)
                replaced = True
                
        # Case 2: Panel is stored as an attribute with a different name
        if not replaced:
            for attr_name in dir(parent_component):
                if attr_name.startswith('_') or attr_name in ('name', 'interfaces'):
                    continue
                    
                attr = getattr(parent_component, attr_name)
                if attr is original_panel:
                    # Found it - replace with the first subpanel
                    setattr(parent_component, attr_name, subpanels[0])
                    # And add any additional subpanels to subs
                    if hasattr(parent_component, 'subs'):
                        for i, subpanel in enumerate(subpanels):
                            if i > 0:
                                if subpanel not in parent_component.subs:
                                    parent_component.subs.append(subpanel)
                    replaced = True
                    break
                    
        # Case 3: Panel is in the subs list
        if hasattr(parent_component, 'subs'):
            try:
                idx = parent_component.subs.index(original_panel)
                # Replace the original panel with the subpanels
                parent_component.subs.pop(idx)
                if not replaced:
                    for subpanel in reversed(subpanels): # insert in order
                        parent_component.subs.insert(idx, subpanel)
                replaced = True
            except ValueError:
                pass # original_panel not in subs list
                
        return replaced
            
        #     # Fallback: if we couldn't find where the panel is stored, 
        #     # just add all subpanels to the subs list.
        #     print(f"Warning: Could not determine how panel {original_panel.name} is stored in its parent. "
        #           f"Adding subpanels to the parent's subs list.")
        #     if not hasattr(parent_component, 'subs'):
        #         parent_component.subs = []
        #     for subpanel in subpanels:
        #         if subpanel not in parent_component.subs:
        #             parent_component.subs.append(subpanel)