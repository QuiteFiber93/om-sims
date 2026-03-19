from urllib import request
from pathlib import Path
import zipfile
from tqdm import tqdm

def download_data(url, dest, desc=None):
    desc = desc or dest.name

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as bar:
        def hook(block_num, block_size, total_size):
            if total_size > 0:
                bar.total = total_size
            bar.update(block_size)

        request.urlretrieve(url, dest, reporthook=hook)

if __name__ == '__main__':        
    BASE = Path(__file__).resolve().parent.parent
    DATASET = BASE / "datasets"
    SPICE_DIR = DATASET / "spice"
    GRAVTY_DIR = DATASET / "gravity"

    # Tries to create directory for SPICE kernels
    # Will also create requisite parent directories
    SPICE_DIR.mkdir(parents = True, exist_ok = True)

    FILES = {
        'lsk/naif0012.tls' : "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
        "pck/pck00011.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
        "pck/earth_latest_high_prec.bpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
        "spk/de442s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442s.bsp",
        "spk/gm_de440.tpc"  :   "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
        "pck/earth_fixed.tf"    :   "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_fixed.tf",
        "spk/stations/earthstns_fx_201023.bsp"  :   "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/stations/earthstns_fx_201023.bsp",
        "fk/stations/earth_topo_201023.tf"  :   "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/stations/earth_topo_201023.tf"
        
    }

    for path, url in FILES.items():
        dest = SPICE_DIR / path
        dest.parent.mkdir(parents = True, exist_ok = True)
        
        if dest.exists():
            print(f'[INFO]\t{path}: already exists in spice directory.')
            continue
        
        else:
            print(f"[INFO]\t{path}: does not exist. Retrieving {path} from {url}.")
            download_data(url, dest, desc=path)
            
    METAKERNEL_PATH = SPICE_DIR / 'metakernel.txt'

    print('[INFO]\tCreating meta-kernel containing relevant SPICE kernels.')
    with METAKERNEL_PATH.open(mode = 'w') as METAKERNEL_FILE:
        # Kernel must have a \begindata line
        METAKERNEL_FILE.write("\\begindata\n")
        # Path values to make writing / changing kernel simpler
        METAKERNEL_FILE.write(f"PATH_VALUES = ('{SPICE_DIR}')\n")
        # Path symbols to use in KERNELS_TO_LOAD
        METAKERNEL_FILE.write(f"PATH_SYMBOLS = ('KERNELS')\n")
        # Listing off kernels to laod
        METAKERNEL_FILE.write(f"KERNELS_TO_LOAD = (")
        for path in FILES.keys():
            METAKERNEL_FILE.write(f"'$KERNELS\{Path(path)}', \n")
        METAKERNEL_FILE.write(f")")

    # Attempts to create gravity file directory
    GRAVTY_DIR.mkdir(parents = True, exist_ok = True)

    # Downloading EGM2008 coefficients
    ZIP_URL = "https://earth-info.nga.mil/php/download.php?file=egm-08spherical"
    ZIP_PATH = GRAVTY_DIR / 'emg2008.zip'
    EARTH_GRAVITY_FILE = 'EGM2008_to2190_TideFree'
    EARTH_GRAVITY_PATH = GRAVTY_DIR / EARTH_GRAVITY_FILE
    EARTH_GRAVITY_DEST_NAME = "egm2008.txt"

    # Checks to see if a download is made
    if not (GRAVTY_DIR / EARTH_GRAVITY_DEST_NAME).exists():
        if not ZIP_PATH.exists():
            print('[INFO]\tRetrieving EGM2008 Spherical Harmonics (ZIP)')
            download_data(ZIP_URL, ZIP_PATH, desc="EGM2008 ZIP")
        
            
        print('[INFO]\tExtracting EGM2008 Coefficients')
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extract(EARTH_GRAVITY_FILE, GRAVTY_DIR)
            
        print('[INFO]\tRemoving zip file')
        ZIP_PATH.unlink()
        
        EARTH_GRAVITY_PATH.rename(GRAVTY_DIR / EARTH_GRAVITY_DEST_NAME)
        
    else:
        print('[INFO]\tEarth Gravity file already exists.')
        
    TEXTURE_DIR = DATASET / "textures"
    TEXTURE_DIR.mkdir(parents = True, exist_ok = True)
    TEXTURE_FILES = {
        "earth/bluemarble-8km.jpg" : "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-base/july/world.200407.3x5400x2700.jpg"
    }
    for rel_path, url in TEXTURE_FILES.items():
        
        dest = TEXTURE_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        if dest.exists():
            continue
        else:
            print(f"[INFO]\tTexture/{rel_path}: does not exist. Retrieving from {url}")
            download_data(url, dest, desc = f"texture/{rel_path}")