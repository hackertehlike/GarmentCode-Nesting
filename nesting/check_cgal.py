import pkgutil
import sys
import CGAL

def check_cgal_modules():
    import CGAL

    print("✅ CGAL is available.")
    print("CGAL module type:", type(CGAL))
    print("CGAL dir contents:")
    print(dir(CGAL.CGAL_Kernel))


if __name__ == "__main__":
    check_cgal_modules()