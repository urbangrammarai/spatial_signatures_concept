"""
Utilities for the data_processing module
"""

import os
import subprocess
import requests
import rasterio
import geopandas
from datetime import datetime
from shapely.geometry import box
from pathlib import Path
from download import download
from .terracotta import optimize_rasters

BASE_URL = (
    "http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
    "GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/"
    "GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/"
)

BANDS = [None, "R", "G", "B", "I"]

def get_filesize(url, human_unit=True):
    """
    Return the file size of a remote (http...) or local file
    ...
    
    Arguments
    ---------
    url : str
         Path to the file. If `url` starts by `http`, it'll use 
         `requests`, else it will use `os.path.getsize`
    human_unit : Boolean
                 [Optional. Default=True] If True, convert bytes 
                 to human-readable units as appropriate
    
    Returns
    -------
    size : int
          File size
    unit : str
          Unit in which size is expressed
    """
    unit = "bytes"
    if url[:4] == "http":
        f = requests.head(url).headers
        try:
            s = int(f["Content-length"])
        except:
            return None
    else:
        s = os.path.getsize(url)
    if human_unit is True:
        s, unit = format_bytes(s)
    return s, unit


def format_bytes(size):
    """
    Convert bytes to human-readable units
    
    Taken from: https://stackoverflow.com/a/49361727
    ...
    
    Arguments
    ---------
    size : int
          File size
    
    Returns
    -------
    fsize : int
           Formatted file size
    unit : str
          Unit in which size is expressed
    """
    # 2**10 = 1024
    power = 2 ** 10
    n = 0
    power_labels = {0: "", 1: "kilo", 2: "mega", 3: "giga", 4: "tera"}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n] + "bytes"


def parse_tile_meta(tile, base_url=BASE_URL, verbose=True):
    """
    Create a GeoDataFrame with metadata about a given UTM tile
    ...
    
    Arguments
    ---------
    tile : str
          UTM code for the desired tile
    base_url : str
          [Optional. Default=BASE_URL] Base URL for all the tiles.
          By default it is set to V1
    verbose : Boolean
             [Optional. Default=True] If True print progress
    
    Returns
    -------
    tile_meta : GeoDataFrame
               Table with metadata about scenes in `tile`. Current info
               includes:
                   - URL
                   - EPSG
                   - UTMtile
                   - minX
                   - minY
                   - maxX
                   - maxY
                   - size
                   - size_unit
                   - size_bytes
                   - geometry
    """
    if tile[-1] == "/":
        tile = tile.strip("/")
    if verbose:
        print(
            f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Working on tile {tile}"
        )
    tile_meta = []
    vrt_p = f"{base_url}{tile}/{tile}_UTM.vrt"
    vrt = rasterio.open(vrt_p)
    gtiffs = [i.strip("/vsicurl/") for i in vrt.files if i[-4:] == ".tif"]
    if len(gtiffs) == 0:
        try:
            vrt_p_aux = f"{base_url}{tile}/{tile}.vrt"
            vrt = rasterio.open(vrt_p_aux)
            gtiffs = [i.strip("/vsicurl/") for i in vrt.files if i[-4:] == ".tif"]
            if verbose:
                print(
                    f"\t{vrt_p.split('/')[-1]} contains no GeoTIFF, attempting {vrt_p_aux.split('/')[-1]}"
                )
        except:
            if verbose:
                print(f"\t{vrt_p_aux} failed, skipping tile")
            return None
    if verbose:
        print(
            f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | {len(gtiffs)} files to process"
        )
    for gtiff in gtiffs:
        if verbose:
            print(
                f"\t{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Working on file {gtiff.split('/')[-1]}"
            )
        try:
            r = rasterio.open(gtiff)
            minX = r.bounds.left
            minY = r.bounds.bottom
            maxX = r.bounds.right
            maxY = r.bounds.top
            geom = box(*r.bounds)
        except:
            minX = minY = maxX = maxY = geom = None
        try:
            size, size_unit = get_filesize(gtiff, human_unit=False)
            hsize, hsize_unit = format_bytes(size)
        except:
            size = hsize = hsize_unit = None
        row = {
            "URL": gtiff,
            "EPSG": r.crs.to_epsg(),
            "UTMtile": tile,
            "minX": minX,
            "minY": minY,
            "maxX": maxX,
            "maxY": maxY,
            "size": hsize,
            "size_unit": hsize_unit,
            "size_bytes": size,
            "geometry": geom,
        }
        tile_meta.append(row)
    tile_meta = geopandas.GeoDataFrame.from_records(tile_meta)
    tile_meta.crs = f"EPSG:{row['EPSG']}"
    return tile_meta


def process_scene(
    row,
    t_crs="EPSG:2770",
    dryrun=False,
    verbose=True,
    check_if_available=True,
    progressbar=True,
    remove_intermediate=True,
):
    """
    Download a GeoTIFF and process (reproject+optimise)
    ...
    
    Arguments
    ---------
    row : pandas.Series/dict
         Key/value object with, at least, the following content:
             - `dst_path`: path of target file
             - `UTMtile`: UTM tile ID of the GeoTIFF
             - `URL`: remote location of the GeoTIFF
    dryrun : Boolean
            [Optional. Default=False] If True, it does not execute download, reprojection and removal or intermediate file
    verbose : Boolean
            [Optional. Default=True] If True, print informative messages about progress
    check_if_available : Boolean
                        [Optional. Default=True] If True, check if `dst_path` exists and exit if so
    progressbar : Boolean
                 [Optional. Default=True] If True, print dynamic progress bar for download
    remove_intermediate : Boolean
                          [Optional. Default=True] If True, remove intermediate files created
    
    Returns
    -------
    None
    """
    osgb_path = row["dst_path"].replace(".tif", "_osgb.tif")
    optd_path = osgb_path.replace(".tif", "_optimised.tif")
    if verbose:
        print(
            (
                f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Working "
                f"on Tile {row['UTMtile']} - File: {row['dst_path'].split('/')[-1]}"
            )
        )
    # Download
    if dryrun:
        print(f"Downloading {row['URL']}")
    else:
        _ = download(
            row["URL"],
            row["dst_path"],
            verbose=verbose,
            progressbar=progressbar,
        )
    # Reproject
    run = True
    if check_if_available and Path(osgb_path).is_file():
        print("\tReprojected file available locally, skipping reprojection...")
        run = False
    if run:
        cmd = f"rio warp {row['dst_path']} " f"{osgb_path} " f"--threads 16 --dst-crs {t_crs}"
        if dryrun:
            print(cmd)
        else:
            if verbose:
                print(f"\t{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | {cmd}")
            output = subprocess.call(cmd, shell=True)
        if remove_intermediate:
            cmd = f"rm {row['dst_path']}"
            if dryrun:
                print(cmd)
            else:
                if verbose:
                    print(f"\t{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | {cmd}")
                output = subprocess.call(cmd, shell=True)
    # Optimise
    run = True
    if check_if_available and Path(optd_path).is_file():
        print("\tOptimised file available locally, skipping optimisation...")
        run = False
    if run:
        if dryrun:
            print(f"Optimise {osgb_path}")
        else:
            if verbose:
                print("\tSplit-opt.")
            # Reproject to web mercator
            wm_path = osgb_path.replace('osgb.tif', 'wm.tif')
            cmd = f"gdalwarp -t_srs 'EPSG:3857' {osgb_path} {wm_path}"
            output = subprocess.call(cmd, shell=True)
            for b in range(1, rasterio.open(osgb_path).count+1):
                tgt_p = osgb_path.replace('_osgb.tif', f"_wm_{BANDS[b]}.tif")
                print(f"\t\t{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Optimising {tgt_p.split('/')[-1]}")

                # Pull out band
                cmd = f"gdal_translate -b {b} {wm_path} {tgt_p}"
                output = subprocess.call(cmd, shell=True)
                # Optimise
                folder = "/".join(tgt_p.split("/")[:-1])
                _ = optimize_rasters([tgt_p],
                                     folder,
                                     overwrite=True,
                                     quiet=True
                                    )
            cmd = f"rm {wm_path}"
            output = subprocess.call(cmd, shell=True)    
            return None


if __name__ == "__main__":

    url = "http://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/30U/S2_percentile_UTM_209-0000000000-0000000000.tif"
    s = get_filesize(url)
