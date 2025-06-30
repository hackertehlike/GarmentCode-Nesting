#!/usr/bin/env python
# filepath: /Users/aysegulbarlas/codestuff/GarmentCode/nesting/test_crossover_oxk.py
import sys, os
# add project root to sys.path for local package imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import time
import yaml
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import numpy as np

from nesting.layout import Layout, Container, Piece
from nesting.chromosome import Chromosome
import nesting.config as config


def create_test_pieces(num_pieces=10, num_parents=2):
    """Create test pieces with parent-child relationships."""
    pieces = []
    
    # Create parent pieces
    for i in range(num_parents):
        parent_id = f"parent_{i}"
        # Create a simple polygon for the piece
        points = []
        radius = random.uniform(10, 20)
        center_x = random.uniform(-100, 100)
        center_y = random.uniform(-100, 100)
        
        for angle in range(0, 360, 36):  # 10-sided polygon
            x = center_x + radius * np.cos(np.radians(angle))
            y = center_y + radius * np.sin(np.radians(angle))
            points.append([x, y])
        
        piece = Piece(points, id=parent_id)
        pieces.append(piece)
        
        # Create child pieces for this parent
        num_children = (num_pieces - num_parents) // num_parents
        for j in range(num_children):
            child_id = f"child_{i}_{j}"
            # Create a smaller polygon for the child piece
            child_points = []
            child_radius = radius * 0.5
            child_center_x = center_x + random.uniform(-radius/2, radius/2)
            child_center_y = center_y + random.uniform(-radius/2, radius/2)
            
            for angle in range(0, 360, 36):  # 10-sided polygon
                x = child_center_x + child_radius * np.cos(np.radians(angle))
                y = child_center_y + child_radius * np.sin(np.radians(angle))
                child_points.append([x, y])
            
            child = Piece(child_points, id=child_id)
            child.parent_id = parent_id
            pieces.append(child)
    
    return pieces


def test_crossover_oxk():
    """Test the crossover_oxk method."""
    # Enable verbose output to see what's happening
    config.VERBOSE = True
    
    # Create container
    container = Container(width=1000, height=1000)
    
    # Create test pieces for parent 1
    pieces1 = create_test_pieces(num_pieces=12, num_parents=3)
    layout1 = Layout({p.id: p for p in pieces1})
    parent1 = Chromosome(pieces1, container)
    
    # Create test pieces for parent 2 (different pieces but same IDs)
    pieces2 = create_test_pieces(num_pieces=12, num_parents=3)
    for i, piece in enumerate(pieces2):
        # Use the same IDs as parent1 for compatibility
        piece.id = pieces1[i].id
        if hasattr(pieces1[i], 'parent_id'):
            piece.parent_id = pieces1[i].parent_id
    
    layout2 = Layout({p.id: p for p in pieces2})
    parent2 = Chromosome(pieces2, container)
    
    # Set design params (required for crossover)
    parent1.design_params = {'test': 'value1'}
    parent2.design_params = {'test': 'value2'}
    parent1.body_params = {}
    parent2.body_params = {}
    
    # Perform crossover with k=2 (2 segments)
    print("\n===== Testing crossover_oxk with k=2 =====")
    try:
        # Mock the regeneration parts that would require actual garment code
        original_calculate_fitness = Chromosome.calculate_fitness
        
        def mock_calculate_fitness(self):
            self.fitness = random.random()
            
        Chromosome.calculate_fitness = mock_calculate_fitness
        
        # Patch the regeneration part since we're just testing the gene selection
        def mock_regenerate(self, pattern):
            return {p.id: p for p in self.genes}
            
        import types
        from unittest.mock import patch, MagicMock
        
        # Monkeypatch MetaGarment and PatternPathExtractor
        from unittest.mock import patch
        
        with patch('assets.garment_programs.meta_garment.MetaGarment') as MockMetaGarment, \
             patch('nesting.path_extractor.PatternPathExtractor') as MockPathExtractor:
             
            # Set up the mock assembly
            mock_pattern = MagicMock()
            mock_pattern.name = "test_pattern"
            mock_pattern.serialize = MagicMock()
            
            mock_meta_garment = MagicMock()
            mock_meta_garment.assembly.return_value = mock_pattern
            MockMetaGarment.return_value = mock_meta_garment
            
            # Set up the mock extractor
            mock_extractor = MagicMock()
            pieces_dict = {p.id: p for p in pieces1}
            mock_extractor.get_all_panel_pieces.return_value = pieces_dict
            MockPathExtractor.return_value = mock_extractor
            
            # Now run the crossover with circular walk
            child = parent1.crossover_oxk(parent2, k=2, circular_walk=True)
            
            # Print the results
            print("\n===== Results =====")
            print(f"Parent 1 genes: {[g.id for g in parent1.genes]}")
            print(f"Parent 2 genes: {[g.id for g in parent2.genes]}")
            print(f"Child genes: {[g.id for g in child.genes]}")
            
            # Check if siblings are kept together
            parent_to_children = {}
            for piece in parent1.genes:
                if piece.parent_id:
                    if piece.parent_id not in parent_to_children:
                        parent_to_children[piece.parent_id] = []
                    parent_to_children[piece.parent_id].append(piece.id)
            
            print("\n===== Checking Sibling Groups =====")
            for parent_id, children in parent_to_children.items():
                # Check which children are in the child chromosome
                child_gene_ids = [g.id for g in child.genes]
                children_in_child = [child_id for child_id in children if child_id in child_gene_ids]
                
                print(f"Parent {parent_id}:")
                print(f"  All children: {children}")
                print(f"  Children in child: {children_in_child}")
                
                # Check if some but not all children are in the child
                if 0 < len(children_in_child) < len(children):
                    print(f"  ERROR: Split sibling group detected! Some but not all siblings were copied.")
                elif len(children_in_child) == len(children):
                    print(f"  OK: All siblings kept together.")
                else:
                    print(f"  OK: No siblings from this group were copied.")
            
            # Check for parent-child coexistence
            print("\n===== Checking Parent-Child Coexistence =====")
            parent_child_coexist = False
            
            for piece in child.genes:
                if piece.parent_id and piece.parent_id in child_gene_ids:
                    parent_child_coexist = True
                    print(f"  ERROR: Child {piece.id} and its parent {piece.parent_id} both exist in the child chromosome!")
            
            if not parent_child_coexist:
                print("  OK: No parent-child coexistence detected.")
            
            # Check that parents and children don't coexist
            print("\n===== Checking Parent-Child Coexistence =====")
            child_gene_ids = [g.id for g in child.genes]
            
            # Get all parent IDs
            parent_ids = set(parent_to_children.keys())
            
            # Check each parent
            for parent_id in parent_ids:
                if parent_id in child_gene_ids:
                    # This parent exists in the child, check if any of its children also exist
                    children = parent_to_children[parent_id]
                    children_in_child = [child_id for child_id in children if child_id in child_gene_ids]
                    
                    if children_in_child:
                        print(f"  ERROR: Parent {parent_id} and its children {children_in_child} coexist in the child!")
                    else:
                        print(f"  OK: Parent {parent_id} exists but none of its children do.")
                else:
                    # Parent does not exist in the child
                    children = parent_to_children[parent_id]
                    children_in_child = [child_id for child_id in children if child_id in child_gene_ids]
                    
                    if children_in_child:
                        print(f"  OK: Parent {parent_id} does not exist but its children {children_in_child} do.")
                    else:
                        print(f"  OK: Neither parent {parent_id} nor its children exist in the child.")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original method
        Chromosome.calculate_fitness = original_calculate_fitness
        config.VERBOSE = False  # Reset verbose flag


def test_real_garment_crossover():
    """Test crossover_oxk with real garment piece sets."""
    print("\n===== Testing crossover_oxk with real garment pieces =====")
    
    # Enable verbose output
    config.VERBOSE = True
    
    # Create container
    container = Container(width=1000, height=1000)
    
    # Define the piece sets
    parent1_piece_ids = {
        'left_sleeve_b', 'sl_left_cuff_f', 'left_btorso', 'right_btorso', 
        'sl_left_cuff_b', 'left_hood', 'right_hood', 'left_sleeve_f', 
        'wb_back', 'skirt_back', 'wb_front', 'sl_right_cuff_f', 
        'right_sleeve_b', 'right_ftorso', 'skirt_front', 'right_sleeve_f', 
        'left_ftorso', 'sl_right_cuff_b'
    }
    
    parent2_piece_ids = {
        'left_sleeve_b', 'skirt_front_s2', 'sl_left_cuff_f', 'left_btorso', 
        'right_btorso', 'sl_left_cuff_b', 'left_hood', 'left_sleeve_f', 
        'wb_back', 'skirt_back', 'wb_front', 'sl_right_cuff_f', 
        'right_sleeve_b', 'right_ftorso', 'skirt_front_s1', 'right_sleeve_f', 
        'left_ftorso', 'sl_right_cuff_b', 'right_hood'
    }
    
    # Create simple piece objects
    def create_simple_piece(piece_id):
        # Create a simple polygon for the piece
        points = []
        radius = random.uniform(10, 20)
        center_x = random.uniform(-100, 100)
        center_y = random.uniform(-100, 100)
        
        for angle in range(0, 360, 36):  # 10-sided polygon
            x = center_x + radius * np.cos(np.radians(angle))
            y = center_y + radius * np.sin(np.radians(angle))
            points.append([x, y])
        
        piece = Piece(points, id=piece_id)
        
        # Set rotation randomly to test rotation inheritance
        piece.rotation = random.choice([0, 90, 180, 270])
        
        return piece
    
    # Create parent-child relationships for sleeves and cuffs
    parent_child_map = {
        'left_sleeve_f': ['sl_left_cuff_f'],
        'left_sleeve_b': ['sl_left_cuff_b'],
        'right_sleeve_f': ['sl_right_cuff_f'],
        'right_sleeve_b': ['sl_right_cuff_b'],
        'skirt_front': ['skirt_front_s1', 'skirt_front_s2']
    }
    
    # Create pieces for parent 1
    pieces1 = []
    for piece_id in parent1_piece_ids:
        piece = create_simple_piece(piece_id)
        
        # Set parent_id for child pieces
        for parent_id, children in parent_child_map.items():
            if piece_id in children:
                piece.parent_id = parent_id
        
        pieces1.append(piece)
    
    # Create pieces for parent 2
    pieces2 = []
    for piece_id in parent2_piece_ids:
        piece = create_simple_piece(piece_id)
        
        # Set parent_id for child pieces
        for parent_id, children in parent_child_map.items():
            if piece_id in children:
                piece.parent_id = parent_id
        
        pieces2.append(piece)
    
    # Create chromosomes
    parent1 = Chromosome(pieces1, container)
    parent2 = Chromosome(pieces2, container)
    
    # Set design params (required for crossover)
    parent1.design_params = {'test': 'value1'}
    parent2.design_params = {'test': 'value2'}
    parent1.body_params = {}
    parent2.body_params = {}
    
    # Monkeypatch for testing
    original_calculate_fitness = Chromosome.calculate_fitness
    
    def mock_calculate_fitness(self):
        self.fitness = random.random()
        
    Chromosome.calculate_fitness = mock_calculate_fitness
    
    try:
        # Patch the regeneration part since we're just testing the gene selection
        from unittest.mock import patch, MagicMock
        
        with patch('assets.garment_programs.meta_garment.MetaGarment') as MockMetaGarment, \
             patch('nesting.path_extractor.PatternPathExtractor') as MockPathExtractor:
             
            # Set up the mock assembly
            mock_pattern = MagicMock()
            mock_pattern.name = "test_pattern"
            mock_pattern.serialize = MagicMock()
            
            mock_meta_garment = MagicMock()
            mock_meta_garment.assembly.return_value = mock_pattern
            MockMetaGarment.return_value = mock_meta_garment
            
            # Set up the mock extractor
            mock_extractor = MagicMock()
            
            # Important: Return the pieces with their IDs preserved
            def setup_mock_extractor(pieces):
                pieces_dict = {p.id: p for p in pieces}
                MockPathExtractor.return_value.get_all_panel_pieces.return_value = pieces_dict
            
            # Set up the extractor to return the same pieces as in the child
            def patch_regeneration(parent1, parent2):
                def side_effect(spec):
                    mock_ex = MagicMock()
                    
                    # Get all piece IDs from the child
                    child_piece_ids = set()
                    for g in child.genes:
                        child_piece_ids.add(g.id)
                    
                    # Create a dictionary with pieces from both parents
                    all_pieces = {}
                    for p in parent1.genes + parent2.genes:
                        if p.id in child_piece_ids:
                            all_pieces[p.id] = copy.deepcopy(p)
                    
                    mock_ex.get_all_panel_pieces.return_value = all_pieces
                    return mock_ex
                
                MockPathExtractor.side_effect = side_effect
            
            # First use parent1's pieces for the extractor
            setup_mock_extractor(parent1.genes)
            
            # Now run the crossover
            print("\n----- Running crossover with k=2 -----")
            child = parent1.crossover_oxk(parent2, k=2, circular_walk=True)
            
            # Print the results with rotations
            print("\n===== Results =====")
            print(f"Parent 1 genes and rotations:")
            for g in parent1.genes:
                print(f"  {g.id}: rotation={g.rotation}")
                
            print(f"\nParent 2 genes and rotations:")
            for g in parent2.genes:
                print(f"  {g.id}: rotation={g.rotation}")
                
            print(f"\nChild genes and rotations:")
            for g in child.genes:
                print(f"  {g.id}: rotation={g.rotation}")
            
            # Check if the skirt_front/skirt_front_s1/skirt_front_s2 are handled correctly
            print("\n===== Checking Skirt Front Pieces =====")
            child_gene_ids = {g.id for g in child.genes}
            
            if 'skirt_front' in child_gene_ids and ('skirt_front_s1' in child_gene_ids or 'skirt_front_s2' in child_gene_ids):
                print("ERROR: Parent 'skirt_front' and its children coexist in the child!")
            elif 'skirt_front' in child_gene_ids:
                print("OK: Parent 'skirt_front' exists but no children.")
            elif 'skirt_front_s1' in child_gene_ids and 'skirt_front_s2' in child_gene_ids:
                print("OK: Both split children exist.")
            elif 'skirt_front_s1' in child_gene_ids or 'skirt_front_s2' in child_gene_ids:
                print("ERROR: Only one of the skirt front split pieces exists!")
            else:
                print("OK: Neither skirt front parent nor children exist.")
            
            # Check if siblings are kept together
            print("\n===== Checking Sleeve-Cuff Relationships =====")
            
            sleeve_cuff_pairs = [
                ('left_sleeve_f', 'sl_left_cuff_f'),
                ('left_sleeve_b', 'sl_left_cuff_b'),
                ('right_sleeve_f', 'sl_right_cuff_f'),
                ('right_sleeve_b', 'sl_right_cuff_b')
            ]
            
            for sleeve, cuff in sleeve_cuff_pairs:
                sleeve_in_child = sleeve in child_gene_ids
                cuff_in_child = cuff in child_gene_ids
                
                if sleeve_in_child and cuff_in_child:
                    print(f"ERROR: Both {sleeve} and its child {cuff} exist in the child!")
                elif sleeve_in_child:
                    print(f"OK: {sleeve} exists but not its child {cuff}.")
                elif cuff_in_child:
                    print(f"OK: Child {cuff} exists but not its parent {sleeve}.")
                else:
                    print(f"OK: Neither {sleeve} nor {cuff} exist.")
    
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original method
        Chromosome.calculate_fitness = original_calculate_fitness
        config.VERBOSE = False  # Reset verbose flag


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    test_crossover_oxk()
    test_real_garment_crossover()
