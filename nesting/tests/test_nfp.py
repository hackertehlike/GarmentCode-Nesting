from nfp import NFPGenerator
import pygarment as pyg
from pygarment.garmentcode.edge import EdgeSequence, Edge
import numpy as np

def create_test_panel(name, vertices):
    """Create a simple test panel from vertices"""
    panel = pyg.Panel(name)
    edges = EdgeSequence()
    
    # Create edges from vertices
    for i in range(len(vertices)):
        start = vertices[i]
        end = vertices[(i + 1) % len(vertices)]
        edge = Edge([start, end])
        edges.append(edge)
    
    panel.edges = edges
    return panel

def test_nfp_generator():
    # Create NFP generator
    nfp_gen = NFPGenerator()
    
    # Create two simple rectangular panels
    panel_a = create_test_panel("A", [
        [0, 0],
        [100, 0],
        [100, 50],
        [0, 50]
    ])
    
    panel_b = create_test_panel("B", [
        [0, 0],
        [50, 0],
        [50, 30],
        [0, 30]
    ])
    
    try:
        # Test panel to polygon conversion
        poly_a = nfp_gen.panel_to_polygon(panel_a)
        print(f"Panel A converted to polygon with {poly_a.size()} vertices")
        
        poly_b = nfp_gen.panel_to_polygon(panel_b)
        print(f"Panel B converted to polygon with {poly_b.size()} vertices")
        
        # Test NFP generation
        nfp = nfp_gen.generate_nfp(panel_a, panel_b)
        print(f"NFP generated with {nfp.size()} vertices")
        
        # Print NFP vertices
        print("\nNFP vertices:")
        for i in range(nfp.size()):
            point = nfp[i]
            print(f"  ({point.x()}, {point.y()})")
            
        return True
        
    except Exception as e:
        print(f"Error testing NFP generator: {e}")
        return False

if __name__ == "__main__":
    success = test_nfp_generator()
    print(f"\nTest {'passed' if success else 'failed'}")