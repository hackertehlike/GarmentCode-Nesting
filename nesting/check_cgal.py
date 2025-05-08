import importlib
import pkgutil
import CGAL

def check_cgal_modules() -> None:
    print("✅  CGAL is available.")
    print("CGAL module type:", type(CGAL))

    # list sub‑modules that actually exist on disk
    print("Discovered CGAL extension modules:")
    for mod in pkgutil.iter_modules(CGAL.__path__, "CGAL."):
        print(" •", mod.name)

    # try to import Kernel (handle the case where it is not present)
    try:
        kernel = importlib.import_module("CGAL.CGAL_Kernel")
        print("\nCGAL_Kernel symbols:", dir(kernel)[:20], "…")
    except ModuleNotFoundError:
        print("\n⚠️  CGAL_Kernel not found – it was not built/installed")

if __name__ == "__main__":
    check_cgal_modules()
