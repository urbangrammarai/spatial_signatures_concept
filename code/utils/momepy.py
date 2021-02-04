import collections
import math
import operator

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import pygeos
import shapely
import networkx as nx
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from scipy.spatial import Voronoi
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize

# TODO: this should not be needed with shapely 2.0
from geopandas._vectorized import _pygeos_to_shapely


def extend_lines(gdf, tolerance, target=None, barrier=None, extension=0):
    """ Function taken from momepy master. Refer to momepy for future use.
    
    Extends lines from gdf to istelf or target within a set tolerance

    Extends unjoined ends of LineString segments to join with other segments or
    target. If ``target`` is passed, extend lines to target. Otherwise extend
    lines to itself.

    If ``barrier`` is passed, each extended line is checked for intersection
    with ``barrier``. If they intersect, extended line is not returned. This
    can be useful if you don't want to extend street network segments through
    buildings.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing LineString geometry
    tolerance : float
        tolerance in snapping (by how much could be each segment
        extended).
    target : GeoDataFrame, GeoSeries
        target geometry to which ``gdf`` gets extended. Has to be 
        (Multi)LineString geometry.
    barrier : GeoDataFrame, GeoSeries
        extended line is not used if it intersects barrier
    extension : float
        by how much to extend line beyond the snapped geometry. Useful
        when creating enclosures to avoid floating point imprecision.
    
    Returns
    -------
    GeoDataFrame
        GeoDataFrame of with extended geometry
    
    See also
    --------
    momepy.close_gaps
    momepy.remove_false_nodes

    """
    # explode to avoid MultiLineStrings
    # double reset index due to the bug in GeoPandas explode
    df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

    if target is None:
        target = df
        itself = True
    else:
        itself = False

    # get underlying pygeos geometry
    geom = df.geometry.values.data

    # extract array of coordinates and number per geometry
    coords = pygeos.get_coordinates(geom)
    indices = pygeos.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = pygeos.points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    tree = pygeos.STRtree(geom)
    inp, res = tree.query_bulk(points, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])

    new_geoms = []
    # iterate over cul-de-sac-like segments and attempt to snap them to street network
    for line in ends:

        l_coords = pygeos.get_coordinates(geom[line])

        start = pygeos.points(l_coords[0])
        end = pygeos.points(l_coords[-1])

        first = list(tree.query(start, predicate="intersects"))
        second = list(tree.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        t = target if not itself else target.drop(line)

        if first and not second:
            snapped = _extend_line(l_coords, t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    pygeos.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(pygeos.linestrings(snapped))
                else:
                    new_geoms.append(
                        pygeos.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )
        elif not first and second:
            snapped = _extend_line(np.flip(l_coords, axis=0), t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    pygeos.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(pygeos.linestrings(snapped))
                else:
                    new_geoms.append(
                        pygeos.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )
        elif not first and not second:
            one_side = _extend_line(l_coords, t, tolerance)
            one_side_e = _extend_line(one_side, t, extension, snap=False)
            snapped = _extend_line(np.flip(one_side_e, axis=0), t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    pygeos.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(pygeos.linestrings(snapped))
                else:
                    new_geoms.append(
                        pygeos.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )

    df.iloc[ends, df.columns.get_loc(df.geometry.name)] = new_geoms
    return df


def _extend_line(coords, target, tolerance, snap=True):
    """
    Extends a line geometry to snap on the target within a tolerance.
    """
    if snap:
        extrapolation = _get_extrapolated_line(
            coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(), tolerance,
        )
        int_idx = target.sindex.query(extrapolation, predicate="intersects")
        intersection = pygeos.intersection(
            target.iloc[int_idx].geometry.values.data, extrapolation
        )
        if intersection.size > 0:
            if len(intersection) > 1:
                distances = {}
                ix = 0
                for p in intersection:
                    distance = pygeos.distance(p, pygeos.points(coords[-1]))
                    distances[ix] = distance
                    ix = ix + 1
                minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                new_point_coords = pygeos.get_coordinates(intersection[minimal])

            else:
                new_point_coords = pygeos.get_coordinates(intersection[0])
            coo = np.append(coords, new_point_coords)
            new = np.reshape(coo, (int(len(coo) / 2), 2))

            return new
        return coords

    extrapolation = _get_extrapolated_line(
        coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
        tolerance,
        point=True,
    )
    return np.vstack([coords, extrapolation])


def _get_extrapolated_line(coords, tolerance, point=False):
    """
    Creates a pygeos line extrapoled in p1->p2 direction.
    """
    p1 = coords[:2]
    p2 = coords[2:]
    a = p2

    # defining new point based on the vector between existing points
    if p1[0] >= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] <= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    else:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    if point:
        return b
    return pygeos.linestrings([a, b])


class Tessellation:
    """
    Generates tessellation.

    Three versions of tessellation can be created:

    1. Morphological tessellation around given buildings ``gdf`` within set ``limit``.
    2. Proximity bands around given street network ``gdf`` within set ``limit``.
    3. Enclosed tessellation based on given buildings ``gdf`` within ``enclosures``.

    Pass either ``limit`` to create morphological tessellation or proximity bands or
    ``enclosures`` to create enclosed tessellation.

    See :cite:`fleischmann2020` for details of implementation of morphological
    tessellation and :cite:`araldi2019` for proximity bands.

    Tessellation requires data of relatively high level of precision and there are three
    particular patterns causing issues.\n
    1. Features will collapse into empty polygon - these do not have tessellation
    cell in the end.\n
    2. Features will split into MultiPolygon - at some cases, features with narrow links
    between parts split into two during 'shrinking'. In most cases that is not an issue
    and resulting tessellation is correct anyway, but sometimes this result in a cell
    being MultiPolygon, which is not correct.\n
    3. Overlapping features - features which overlap even after 'shrinking' cause
    invalid tessellation geometry.\n
    All three types can be tested prior :class:`momepy.Tessellation` using
    :class:`momepy.CheckTessellationInput`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing building footprints or street network
    unique_id : str
        name of the column with unique id
    limit : MultiPolygon or Polygon (default None)
        MultiPolygon or Polygon defining the study area limiting
        morphological tessellation or proximity bands
        (otherwise it could go to infinity).
    shrink : float (default 0.4)
        distance for negative buffer to generate space between adjacent polygons
        (if geometry type of gdf is (Multi)Polygon).
    segment : float (default 0.5)
        maximum distance between points after discretization
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps
    enclosures : GeoDataFrame (default None)
        Enclosures geometry. Can  be generated using :func:`momepy.enclosures`.
    enclosure_id : str (default 'eID')
        name of the enclosure_id containing unique identifer for each row in ``enclosures``.
        Applies only if ``enclosures`` are passed.
    threshold : float (default 0.05)
        The minimum threshold for a building to be considered within an enclosure.
        Threshold is a ratio of building area which needs to be within an enclosure to
        inlude it in the tessellation of that enclosure. Resolves sliver geometry
        issues. Applies only if ``enclosures`` are passed.
    use_dask : bool (default True)
        Use parallelised algorithm based on ``dask.dataframe``. Requires dask.
        Applies only if ``enclosures`` are passed.
    n_chunks : None
        Number of chunks to be used in parallelization. Ideal is one chunk per thread.
        Applies only if ``enclosures`` are passed. Defualt automatically uses
        n == dask.system.cpu_count.

    Attributes
    ----------
    tessellation : GeoDataFrame
        GeoDataFrame containing resulting tessellation

        For enclosed tessellation, gdf contains three columns:
            - ``geometry``,
            - ``unique_id`` matching with parental building,
            - ``enclosure_id`` matching with enclosure integer index

    gdf : GeoDataFrame
        original GeoDataFrame
    id : Series
        Series containing used unique ID
    limit : MultiPolygon or Polygon
        limit
    shrink : float
        used shrink value
    segment : float
        used segment value
    collapsed : list
        list of unique_id's of collapsed features (if there are some)
        Applies only if ``limit`` is passed.
    multipolygons : list
        list of unique_id's of features causing MultiPolygons (if there are some)
        Applies only if ``limit`` is passed.

    Examples
    --------
    >>> tess = mm.Tessellation(
    ... buildings_df, 'uID', limit=mm.buffered_limit(buildings_df)
    ... )
    Inward offset...
    Generating input point array...
    Generating Voronoi diagram...
    Generating GeoDataFrame...
    Dissolving Voronoi polygons...
    >>> tess.tessellation.head()
        uID	geometry
    0	1	POLYGON ((1603586.677274485 6464344.667944215,...
    1	2	POLYGON ((1603048.399497852 6464176.180701573,...
    2	3	POLYGON ((1603071.342637536 6464158.863329805,...
    3	4	POLYGON ((1603055.834005827 6464093.614718676,...
    4	5	POLYGON ((1603106.417554705 6464130.215958447,...

    >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])
    >>> encl_tess = mm.Tessellation(
    ... buildings_df, 'uID', enclosures=enclosures
    ... )
    >>> encl_tess.tessellation.head()
         uID                                           geometry  eID
    0  109.0  POLYGON ((1603369.789 6464340.661, 1603368.754...    0
    1  110.0  POLYGON ((1603368.754 6464340.097, 1603369.789...    0
    2  111.0  POLYGON ((1603458.666 6464332.614, 1603458.332...    0
    3  112.0  POLYGON ((1603462.235 6464285.609, 1603454.795...    0
    4  113.0  POLYGON ((1603524.561 6464388.609, 1603532.241...    0

    """

    def __init__(
        self,
        gdf,
        unique_id,
        limit=None,
        shrink=0.4,
        segment=0.5,
        verbose=True,
        enclosures=None,
        enclosure_id="eID",
        threshold=0.05,
        use_dask=True,
        n_chunks=None,
        **kwargs,
    ):
        self.gdf = gdf
        self.id = gdf[unique_id]
        self.limit = limit
        self.shrink = shrink
        self.segment = segment
        self.enclosure_id = enclosure_id

        if limit is not None and enclosures is not None:
            raise ValueError(
                "Both `limit` and `enclosures` cannot be passed together. "
                "Pass `limit` for morphological tessellation or `enclosures` "
                "for enclosed tessellation."
            )
        if enclosures is not None:
            self.tessellation = self._enclosed_tessellation(
                gdf, enclosures, unique_id, enclosure_id, threshold, use_dask, n_chunks,
            )
        else:
            self.tessellation = self._morphological_tessellation(
                gdf, unique_id, limit, shrink, segment, verbose
            )

    def _morphological_tessellation(
        self, gdf, unique_id, limit, shrink, segment, verbose, check=True
    ):
        objects = gdf.copy()

        if isinstance(limit, (gpd.GeoSeries, gpd.GeoDataFrame)):
            limit = limit.unary_union
        if isinstance(limit, BaseGeometry):
            limit = pygeos.from_shapely(limit)

        bounds = pygeos.bounds(limit)
        centre_x = (bounds[0] + bounds[2]) / 2
        centre_y = (bounds[1] + bounds[3]) / 2
        objects["geometry"] = objects["geometry"].translate(
            xoff=-centre_x, yoff=-centre_y
        )

        if shrink != 0:
            print("Inward offset...") if verbose else None
            mask = objects.type.isin(["Polygon", "MultiPolygon"])
            objects.loc[mask, "geometry"] = objects[mask].buffer(
                -shrink, cap_style=2, join_style=2
            )

        objects = objects.reset_index(drop=True).explode()
        objects = objects.set_index(unique_id)

        print("Generating input point array...") if verbose else None
        points, ids = self._dense_point_array(
            objects.geometry.values.data, distance=segment, index=objects.index
        )

        # add convex hull buffered large distance to eliminate infinity issues
        series = gpd.GeoSeries(limit, crs=gdf.crs).translate(
            xoff=-centre_x, yoff=-centre_y
        )
        width = bounds[2] - bounds[0]
        leng = bounds[3] - bounds[1]
        hull = series.geometry[[0]].buffer(2 * width if width > leng else 2 * leng)
        # pygeos bug fix
        if (hull.type == "MultiPolygon").any():
            hull = hull.explode()
        hull_p, hull_ix = self._dense_point_array(
            hull.values.data, distance=pygeos.length(limit) / 100, index=hull.index
        )
        points = np.append(points, hull_p, axis=0)
        ids = ids + ([-1] * len(hull_ix))

        print("Generating Voronoi diagram...") if verbose else None
        voronoi_diagram = Voronoi(np.array(points))

        print("Generating GeoDataFrame...") if verbose else None
        regions_gdf = self._regions(voronoi_diagram, unique_id, ids, crs=gdf.crs)

        print("Dissolving Voronoi polygons...") if verbose else None
        morphological_tessellation = regions_gdf[[unique_id, "geometry"]].dissolve(
            by=unique_id, as_index=False
        )

        morphological_tessellation = gpd.clip(morphological_tessellation, series)

        morphological_tessellation["geometry"] = morphological_tessellation[
            "geometry"
        ].translate(xoff=centre_x, yoff=centre_y)

        if check:
            self._check_result(morphological_tessellation, gdf, unique_id=unique_id)

        return morphological_tessellation

    def _dense_point_array(self, geoms, distance, index):
        """
        geoms - array of pygeos lines
        """
        # interpolate lines to represent them as points for Voronoi
        points = np.empty((0, 2))
        ids = []

        if pygeos.get_type_id(geoms[0]) not in [1, 2, 5]:
            lines = pygeos.boundary(geoms)
        else:
            lines = geoms
        lengths = pygeos.length(lines)
        for ix, line, length in zip(index, lines, lengths):
            if length > distance:  # some polygons might have collapsed
                pts = pygeos.line_interpolate_point(
                    line,
                    np.linspace(0.1, length - 0.1, num=int((length - 0.1) // distance)),
                )  # .1 offset to keep a gap between two segments
                points = np.append(points, pygeos.get_coordinates(pts), axis=0)
                ids += [ix] * len(pts)

        return points, ids

        # here we might also want to append original coordinates of each line
        # to get a higher precision on the corners

    def _regions(self, voronoi_diagram, unique_id, ids, crs):
        """
        Generate GeoDataFrame of Voronoi regions from scipy.spatial.Voronoi.
        """
        vertices = pd.Series(voronoi_diagram.regions).take(voronoi_diagram.point_region)
        polygons = []
        for region in vertices:
            if -1 not in region:
                polygons.append(pygeos.polygons(voronoi_diagram.vertices[region]))
            else:
                polygons.append(None)

        regions_gdf = gpd.GeoDataFrame(
            {unique_id: ids}, geometry=polygons, crs=crs
        ).dropna()
        regions_gdf = regions_gdf.loc[
            regions_gdf[unique_id] != -1
        ]  # delete hull-based cells

        return regions_gdf

    def _check_result(self, tesselation, orig_gdf, unique_id):
        """
        Check whether result of tessellation matches buildings and contains only Polygons.
        """
        # check against input layer
        ids_original = list(orig_gdf[unique_id])
        ids_generated = list(tesselation[unique_id])
        if len(ids_original) != len(ids_generated):
            import warnings

            self.collapsed = set(ids_original).difference(ids_generated)
            warnings.warn(
                "Tessellation does not fully match buildings. {len} element(s) collapsed "
                "during generation - unique_id: {i}".format(
                    len=len(self.collapsed), i=self.collapsed
                )
            )

        # check MultiPolygons - usually caused by error in input geometry
        self.multipolygons = tesselation[tesselation.geometry.type == "MultiPolygon"][
            unique_id
        ]
        if len(self.multipolygons) > 0:
            import warnings

            warnings.warn(
                "Tessellation contains MultiPolygon elements. Initial objects should be edited. "
                "unique_id of affected elements: {}".format(list(self.multipolygons))
            )

    def _enclosed_tessellation(
        self,
        buildings,
        enclosures,
        unique_id,
        enclosure_id="eID",
        threshold=0.05,
        use_dask=True,
        n_chunks=None,
        **kwargs,
    ):
        """Enclosed tessellation
        Generate enclosed tessellation based on barriers defining enclosures and buildings
        footprints.

        Parameters
        ----------
        buildings : GeoDataFrame
            GeoDataFrame containing building footprints. Expects (Multi)Polygon geometry.
        enclosures : GeoDataFrame
            Enclosures geometry. Can  be generated using :func:`momepy.enclosures`.
        unique_id : str
            name of the column with unique id of buildings gdf
        threshold : float (default 0.05)
            The minimum threshold for a building to be considered within an enclosure.
            Threshold is a ratio of building area which needs to be within an enclosure to
            inlude it in the tessellation of that enclosure. Resolves sliver geometry
            issues.
        use_dask : bool (default True)
            Use parallelised algorithm based on ``dask.dataframe``. Requires dask.
        n_chunks : None
            Number of chunks to be used in parallelization. Ideal is one chunk per thread.
            Applies only if ``enclosures`` are passed. Defualt automatically uses
            n == dask.system.cpu_count.
        **kwargs
            Keyword arguments passed to Tessellation algorithm (as ``shrink``
            or ``segment``).

        Returns
        -------
        tessellation : GeoDataFrame
            gdf contains three columns:
                geometry,
                unique_id matching with parental building,
                enclosure_id matching with enclosure integer index

        Examples
        --------
        >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])
        >>> enclosed_tess = mm.enclosed_tessellation(buildings, enclosures)

        """
        enclosures = enclosures.reset_index(drop=True)

        # determine which polygons should be split
        inp, res = buildings.sindex.query_bulk(
            enclosures.geometry, predicate="intersects"
        )
        unique, counts = np.unique(inp, return_counts=True)
        splits = unique[counts > 1]
        single = unique[counts == 1]

        if use_dask:
            try:
                import dask.dataframe as dd
                from dask.system import cpu_count
            except ImportError:
                use_dask = False

                import warnings

                warnings.warn(
                    "dask.dataframe could not be imported. Setting `use_dask=False`."
                )

        if use_dask:
            if n_chunks is None:
                n_chunks = cpu_count()
            # initialize dask.series
            ds = dd.from_array(splits, chunksize=len(splits) // n_chunks - 1)
            # generate enclosed tessellation using dask
            new = (
                ds.apply(
                    self._tess,
                    meta=(None, "object"),
                    args=(enclosures, buildings, inp, res, threshold, unique_id),
                )
                .compute()
                .to_list()
            )

        else:
            new = [
                self._tess(
                    i,
                    enclosures,
                    buildings,
                    inp,
                    res,
                    threshold=threshold,
                    unique_id=unique_id,
                    **kwargs,
                )
                for i in splits
            ]

        # finalise the result
        clean_blocks = enclosures.drop(splits)
        clean_blocks.loc[single, "uID"] = clean_blocks.loc[single][enclosure_id].apply(
            lambda ix: buildings.iloc[res[inp == ix][0]][unique_id]
        )
        tessellation = pd.concat(new)

        return tessellation.append(clean_blocks).reset_index(drop=True)

    def _tess(
        self,
        ix,
        enclosure,
        buildings,
        query_inp,
        query_res,
        threshold,
        unique_id,
        **kwargs,
    ):
        poly = enclosure.geometry.values.data[ix]
        blg = buildings.iloc[query_res[query_inp == ix]]
        within = blg[
            pygeos.area(pygeos.intersection(blg.geometry.values.data, poly))
            > (pygeos.area(blg.geometry.values.data) * threshold)
        ]
        if len(within) > 1:
            tess = self._morphological_tessellation(
                within,
                unique_id,
                poly,
                shrink=self.shrink,
                segment=self.segment,
                verbose=False,
                check=False,
            )
            tess[self.enclosure_id] = enclosure[self.enclosure_id].iloc[ix]
            return tess
        return gpd.GeoDataFrame(
            {self.enclosure_id: enclosure[self.enclosure_id].iloc[ix], unique_id: None},
            geometry=[poly],
            index=[0],
        )
    

    
def enclosures(
    primary_barriers, limit=None, additional_barriers=None, enclosure_id="eID"
):
    """
    Generate enclosures based on passed barriers.

    Enclosures are areas enclosed from all sides by at least one type of
    a barrier. Barriers are typically roads, railways, natural features
    like rivers and other water bodies or coastline. Enclosures are a
    result of polygonization of the  ``primary_barrier`` and ``limit`` and its
    subdivision based on additional_barriers.

    Parameters
    ----------
    primary_barriers : GeoDataFrame, GeoSeries
        GeoDataFrame or GeoSeries containing primary barriers.
        (Multi)LineString geometry is expected.
    limit : GeoDataFrame, GeoSeries (default None)
        GeoDataFrame or GeoSeries containing external limit of enclosures,
        i.e. the area which gets partitioned. If None is passed,
        the internal area of ``primary_barriers`` will be used.
    additional_barriers : GeoDataFrame
        GeoDataFrame or GeoSeries containing additional barriers.
        (Multi)LineString geometry is expected.
    enclosure_id : str (default 'eID')
        name of the enclosure_id (to be created).

    Returns
    -------
    enclosures : GeoDataFrame
       GeoDataFrame containing enclosure geometries and enclosure_id

    Examples
    --------
    >>> enclosures = mm.enclosures(streets, admin_boundary, [railway, rivers])

    """
    if limit is not None:
        if limit.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            limit = limit.boundary
        barriers = pd.concat([primary_barriers.geometry, limit.geometry])
    else:
        barriers = primary_barriers
    unioned = barriers.unary_union
    polygons = polygonize(unioned)
    enclosures = gpd.GeoSeries(list(polygons), crs=primary_barriers.crs)

    if additional_barriers is not None:
        if not isinstance(additional_barriers, list):
            raise TypeError(
                "`additional_barriers` expects a list of GeoDataFrames or GeoSeries."
                f"Got {type(additional_barriers)}."
            )
        additional = pd.concat([gdf.geometry for gdf in additional_barriers])

        inp, res = enclosures.sindex.query_bulk(
            additional.geometry, predicate="intersects"
        )
        unique = np.unique(res)

        new = []

        for i in unique:
            poly = enclosures.values.data[i]  # get enclosure polygon
            crossing = inp[res == i]  # get relevant additional barriers
            buf = pygeos.buffer(poly, 0.01)  # to avoid floating point errors
            crossing_ins = pygeos.intersection(
                buf, additional.values.data[crossing]
            )  # keeping only parts of additional barriers within polygon
            union = pygeos.union_all(
                np.append(crossing_ins, pygeos.boundary(poly))
            )  # union
            polygons = np.array(
                list(polygonize(_pygeos_to_shapely(union)))
            )  # polygonize
            within = pygeos.covered_by(
                pygeos.from_shapely(polygons), buf
            )  # keep only those within original polygon
            new += list(polygons[within])

        final_enclosures = (
            gpd.GeoSeries(enclosures)
            .drop(unique)
            .append(gpd.GeoSeries(new))
            .reset_index(drop=True)
        ).set_crs(primary_barriers.crs)

        return gpd.GeoDataFrame(
            {enclosure_id: range(len(final_enclosures))}, geometry=final_enclosures
        )

    return gpd.GeoDataFrame({enclosure_id: range(len(enclosures))}, geometry=enclosures)


class NeighborDistance:
    """
    Calculate the mean distance to adjacent buildings (based on ``spatial_weights``)

    If no neighbours are found, return ``np.nan``.

    .. math::
        \\frac{1}{n}\\sum_{i=1}^n dist_i=\\frac{dist_1+dist_2+\\cdots+dist_n}{n}

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    spatial_weights : libpysal.weights
        spatial weights matrix based on unique_id
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index.
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID

    Examples
    --------
    >>> buildings_df['neighbour_distance'] = momepy.NeighborDistance(buildings_df, sw, 'uID').series
    100%|██████████| 144/144 [00:00<00:00, 345.78it/s]
    >>> buildings_df['neighbour_distance'][0]
    29.18589019096464
    """

    def __init__(self, gdf, spatial_weights, unique_id, verbose=True):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]
        # define empty list for results
        results_list = []

        data = gdf.set_index(unique_id).geometry

        # iterating over rows one by one
        for index, geom in tqdm(
            data.iteritems(), total=data.shape[0], disable=not verbose
        ):
            if geom is not None and index in spatial_weights.neighbors.keys():
                neighbours = spatial_weights.neighbors[index]
                building_neighbours = data.loc[neighbours]
                if len(building_neighbours) > 0:
                    results_list.append(
                        building_neighbours.geometry.distance(geom).mean()
                    )
                else:
                    results_list.append(np.nan)
            else:
                results_list.append(np.nan)

        self.series = pd.Series(results_list, index=gdf.index)

        
class Courtyards:
    """
    Calculate the number of courtyards within the joined structure.

    Adapted from :cite:`schirmer2015`.


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    block_id : (deprecated)
    spatial_weights : libpysal.weights, optional
        spatial weights matrix - If None, Queen contiguity matrix will be calculated
        based on objects. It is to denote adjacent buildings (note: based on integer index).
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix

    Examples
    --------
    >>> buildings_df['courtyards'] = mm.Courtyards(buildings_df).series
    Calculating spatial weights...
    """

    def __init__(self, gdf, block_id=None, spatial_weights=None, verbose=True):
        if block_id is not None:
            warnings.warn(
                "block_id is deprecated and will be removed in v0.4.",
                FutureWarning,
            )
        self.gdf = gdf

        results_list = []
        gdf = gdf.copy()

        # if weights matrix is not passed, generate it from objects
        if spatial_weights is None:
            print("Calculating spatial weights...") if verbose else None
            from libpysal.weights import Queen

            spatial_weights = Queen.from_dataframe(gdf, silence_warnings=True)

        self.sw = spatial_weights
        # dict to store nr of courtyards for each uID
        courtyards = {}
        components = pd.Series(spatial_weights.component_labels, index=gdf.index)
        for i, index in tqdm(
            enumerate(gdf.index), total=gdf.shape[0], disable=not verbose
        ):
            # if the id is already present in courtyards, continue (avoid repetition)
            if index in courtyards:
                continue
            else:
                comp = spatial_weights.component_labels[i]
                to_join = components[components == comp].index
                joined = gdf.loc[to_join]
                dissolved = joined.geometry.buffer(
                    0.01
                ).unary_union  # buffer to avoid multipolygons where buildings touch by corners only
                try:
                    interiors = len(list(dissolved.interiors))
                except (ValueError):
                    print("Something unexpected happened.")
                for b in to_join:
                    courtyards[b] = interiors  # fill dict with values

        results_list = [courtyards[index] for index in gdf.index]

        self.series = pd.Series(results_list, index=gdf.index)

        
class MeanInterbuildingDistance:
    """
    Calculate the mean interbuilding distance

    Interbuilding distances are calculated between buildings on adjacent cells based on
    ``spatial_weights``, while the extent is defined as order of contiguity.

    .. math::


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    unique_id : str
        name of the column with unique id used as ``spatial_weights`` index
    spatial_weights : libpysal.weights
        spatial weights matrix
    order : int
        Order of contiguity defining the extent
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    sw : libpysal.weights
        spatial weights matrix
    id : Series
        Series containing used unique ID
    sw_higher : libpysal.weights
        Spatial weights matrix of higher order
    order : int
        Order of contiguity.

    Notes
    -----
    Fix UserWarning.

    Examples
    --------
    >>> buildings_df['mean_interbuilding_distance'] = momepy.MeanInterbuildingDistance(buildings_df, sw, 'uID').series
    Computing mean interbuilding distances...
    100%|██████████| 144/144 [00:00<00:00, 317.42it/s]
    >>> buildings_df['mean_interbuilding_distance'][0]
    29.305457092042744
    """

    def __init__(
        self,
        gdf,
        spatial_weights,
        unique_id,
        spatial_weights_higher=None,
        order=3,
        verbose=True,
    ):
        self.gdf = gdf
        self.sw = spatial_weights
        self.id = gdf[unique_id]

        data = gdf.set_index(unique_id).geometry

        # define empty list for results
        results_list = []

        # define adjacency list from lipysal
        adj_list = spatial_weights.to_adjlist()
        adj_list["weight"] = (
            data.loc[adj_list.focal]
            .reset_index(drop=True)
            .distance(data.loc[adj_list.neighbor].reset_index(drop=True))
            .values
        )

        # generate graph
        G = nx.from_pandas_edgelist(
            adj_list, source="focal", target="neighbor", edge_attr="weight"
        )

        print("Computing mean interbuilding distances...") if verbose else None
        # iterate over subgraphs to get the final values
        for uid in tqdm(data.index, total=data.shape[0], disable=not verbose):
            try:
                sub = nx.ego_graph(G, uid, radius=order)
                results_list.append(
                    np.nanmean([x[-1] for x in list(sub.edges.data("weight"))])
                )
            except Exception:
                results_list.append(np.nan)
        self.series = pd.Series(results_list, index=gdf.index)

        
class Corners:
    """
    Calculates number of corners of each object in given GeoDataFrame.

    Uses only external shape (``shapely.geometry.exterior``), courtyards are not
    included.

    .. math::
        \\sum corner

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame


    Examples
    --------
    >>> buildings_df['corners'] = momepy.Corners(buildings_df).series
    100%|██████████| 144/144 [00:00<00:00, 1042.15it/s]
    >>> buildings_df.corners[0]
    24


    """

    def __init__(self, gdf, verbose=True):
        self.gdf = gdf

        # define empty list for results
        results_list = []

        # calculate angle between points, return true or false if real corner
        def _true_angle(a, b, c):
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            # TODO: add arg to specify these values
            if np.degrees(angle) <= 170:
                return True
            if np.degrees(angle) >= 190:
                return True
            return False

        # fill new column with the value of area, iterating over rows one by one
        for geom in tqdm(gdf.geometry, total=gdf.shape[0], disable=not verbose):
            if geom.type == "Polygon":
                corners = 0  # define empty variables
                points = list(geom.exterior.coords)  # get points of a shape
                stop = len(points) - 1  # define where to stop
                for i in np.arange(
                    len(points)
                ):  # for every point, calculate angle and add 1 if True angle
                    if i == 0:
                        continue
                    elif i == stop:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[1])

                        if _true_angle(a, b, c) is True:
                            corners = corners + 1
                        else:
                            continue

                    else:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[i + 1])

                        if _true_angle(a, b, c) is True:
                            corners = corners + 1
                        else:
                            continue
            elif geom.type == "MultiPolygon":
                corners = 0  # define empty variables
                for g in geom.geoms:
                    points = list(g.exterior.coords)  # get points of a shape
                    stop = len(points) - 1  # define where to stop
                    for i in np.arange(
                        len(points)
                    ):  # for every point, calculate angle and add 1 if True angle
                        if i == 0:
                            continue
                        elif i == stop:
                            a = np.asarray(points[i - 1])
                            b = np.asarray(points[i])
                            c = np.asarray(points[1])

                            if _true_angle(a, b, c) is True:
                                corners = corners + 1
                            else:
                                continue

                        else:
                            a = np.asarray(points[i - 1])
                            b = np.asarray(points[i])
                            c = np.asarray(points[i + 1])

                            if _true_angle(a, b, c) is True:
                                corners = corners + 1
                            else:
                                continue                 
            else:
                corners = np.nan
            
            results_list.append(corners)


        self.series = pd.Series(results_list, index=gdf.index)

        
class CentroidCorners:
    """
    Calculates mean distance centroid - corners and st. deviation.

    .. math::
        \\overline{x}=\\frac{1}{n}\\left(\\sum_{i=1}^{n} dist_{i}\\right);\\space \\mathrm{SD}=\\sqrt{\\frac{\\sum|x-\\overline{x}|^{2}}{n}}

    Adapted from :cite:`schirmer2015` and :cite:`cimburova2017`.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    verbose : bool (default True)
        if True, shows progress bars in loops and indication of steps

    Attributes
    ----------
    mean : Series
        Series containing mean distance values.
    std : Series
        Series containing standard deviation values.
    gdf : GeoDataFrame
        original GeoDataFrame

    Examples
    --------
    >>> ccd = momepy.CentroidCorners(buildings_df)
    100%|██████████| 144/144 [00:00<00:00, 846.58it/s]
    >>> buildings_df['ccd_means'] = ccd.means
    >>> buildings_df['ccd_stdev'] = ccd.std
    >>> buildings_df['ccd_means'][0]
    15.961531913184833
    >>> buildings_df['ccd_stdev'][0]
    3.0810634305400177
    """

    def __init__(self, gdf, verbose=True):
        from momepy.shape import _circle_radius
        
        self.gdf = gdf
        # define empty list for results
        results_list = []
        results_list_sd = []

        # calculate angle between points, return true or false if real corner
        def true_angle(a, b, c):
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            if np.degrees(angle) <= 170:
                return True
            if np.degrees(angle) >= 190:
                return True
            return False

        # iterating over rows one by one
        for geom in tqdm(gdf.geometry, total=gdf.shape[0], disable=not verbose):
            if geom.type == "Polygon":
                distances = []  # set empty list of distances
                centroid = geom.centroid  # define centroid
                points = list(geom.exterior.coords)  # get points of a shape
                stop = len(points) - 1  # define where to stop
                for i in np.arange(
                    len(points)
                ):  # for every point, calculate angle and add 1 if True angle
                    if i == 0:
                        continue
                    elif i == stop:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[1])
                        p = Point(points[i])

                        if true_angle(a, b, c) is True:
                            distance = centroid.distance(
                                p
                            )  # calculate distance point - centroid
                            distances.append(distance)  # add distance to the list
                        else:
                            continue

                    else:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[i + 1])
                        p = Point(points[i])

                        if true_angle(a, b, c) is True:
                            distance = centroid.distance(p)
                            distances.append(distance)
                        else:
                            continue
                if not distances:  # circular buildings
                    if geom.has_z:
                        coords = [
                            (coo[0], coo[1]) for coo in geom.convex_hull.exterior.coords
                        ]
                    else:
                        coords = geom.convex_hull.exterior.coords
                    results_list.append(_circle_radius(coords))
                    results_list_sd.append(0)
                else:
                    results_list.append(np.mean(distances))  # calculate mean
                    results_list_sd.append(np.std(distances))  # calculate st.dev
            else:
                results_list.append(np.nan)
                results_list_sd.append(np.nan)

        self.mean = pd.Series(results_list, index=gdf.index)
        self.std = pd.Series(results_list_sd, index=gdf.index)

        
def get_network_ratio(df, edges, initial_buffer=500):
    """
    Link polygons to network edges based on the proportion of overlap (if a cell
    intersects more than one edge)

    Useful if you need to link enclosed tessellation to street network. Ratios can
    be used as weights when linking network-based values to cells. For a purely
    distance-based link use :func:`momepy.get_network_id`.

    Links are based on the integer position of edge (``iloc``).

    Parameters
    ----------

    df : GeoDataFrame
        GeoDataFrame containing objects to snap (typically enclosed tessellation)
    edges : GeoDataFrame
        GeoDataFrame containing street network
    initial_buffer : float
        Initial buffer used to link non-intersecting cells.

    Returns
    -------

    DataFrame

    See also
    --------
    momepy.get_network_id
    momepy.get_node_id

    Examples
    --------
    >>> links = mm.get_network_ratio(enclosed_tessellation, streets)
    >>> links.head()
      edgeID_keys                              edgeID_values
    0        [34]                                      [1.0]
    1     [0, 34]  [0.38508998545027145, 0.6149100145497285]
    2        [32]                                        [1]
    3         [0]                                      [1.0]
    4        [26]                                        [1]

    """

    # intersection-based join
    buff = edges.buffer(0.01)  # to avoid floating point error
    inp, res = buff.sindex.query_bulk(df.geometry, predicate="intersects")
    intersections = (
        df.iloc[inp]
        .reset_index(drop=True)
        .intersection(buff.iloc[res].reset_index(drop=True))
    )
    mask = intersections.area > 0.0001
    intersections = intersections[mask]
    inp = inp[mask]
    lengths = intersections.area
    grouped = lengths.groupby(inp)
    totals = grouped.sum()
    ints_vect = []
    for name, group in grouped:
        ratios = group / totals.loc[name]
        ints_vect.append({res[item[0]]: item[1] for item in ratios.iteritems()})

    edge_dicts = pd.Series(ints_vect, index=totals.index)

    # nearest neighbor join
    nans = df.index.difference(edge_dicts.index)
    buffered = df.iloc[nans].buffer(initial_buffer)
    additional = []
    for orig, geom in zip(df.iloc[nans].geometry, buffered.geometry):
        query = edges.sindex.query(geom, predicate="intersects")
        b = initial_buffer
        while query.size == 0:
            query = edges.sindex.query(geom.buffer(b), predicate="intersects")
            b += initial_buffer
        additional.append({edges.iloc[query].distance(orig).idxmin(): 1})

    additional = pd.Series(additional, index=nans)
    ratios = pd.concat([edge_dicts, additional]).sort_index()
    result = pd.DataFrame()
    result["edgeID_keys"] = ratios.apply(lambda d: list(d.keys()))
    result["edgeID_values"] = ratios.apply(lambda d: list(d.values()))
    result.index = df.index
    return result