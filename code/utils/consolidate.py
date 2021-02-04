from itertools import combinations
import collections
import math
from time import time

import pygeos
import numpy as np
import pandas as pd
import geopandas as gpd
import momepy as mm

from shapely.ops import polygonize
from shapely.geometry import box
from scipy.spatial import Voronoi
from tqdm import tqdm

from libpysal.weights import Queen, W, w_union


def highway_fix(gdf, tick_length, allowed_error, tolerance):
    s = time()

    high = gdf[gdf.highway.astype(str) == "motorway"]
    print(
        f"generating Queen W. Elapsed time: {time() - s}.",
    )
    Q = Queen.from_dataframe(high, silence_warnings=True)

    neighbors = {}

    print(f"generating parallel W. Elapsed time: {time() - s}.")
    pygeos_lines = high.geometry.values.data
    for i, line in tqdm(enumerate(pygeos_lines), total=len(pygeos_lines)):
        pts = pygeos.line_interpolate_point(line, [1, 10])
        coo = pygeos.get_coordinates(pts)

        l1 = _get_line(coo[0], coo[1], tick_length)
        l2 = _get_line(coo[1], coo[0], tick_length)

        query = high.sindex.query_bulk(
            pygeos.linestrings([l1, l2]), predicate="intersects"
        )
        un, ct = np.unique(query[1], return_counts=True)
        double = un[(un != i) & (ct == 2)]
        if len(double) > 0:
            for d in range(len(double)):
                distances = pygeos.distance(pts, pygeos_lines[d])
                if abs(distances[0] - distances[1]) <= allowed_error:
                    neighbors[i] = [d]
                else:
                    neighbors[i] = []
        else:
            neighbors[i] = []

    w = W(neighbors, silence_warnings=True)
    union = w_union(Q, w, silence_warnings=True)

    non_high = gdf[gdf.highway.astype(str) != "motorway"]

    print(f"generating new geometry. Elapsed time: {time() - s}.")
    replacements = []
    removal_non = []
    snapped = []
    for c in tqdm(range(union.n_components), total=union.n_components):
        lines = high.geometry[union.component_labels == c]
        av = _average_geometry(lines)
        if len(av) > 0:
            qbulk = lines.sindex.query_bulk(av, predicate="intersects")
            comp = np.delete(av, qbulk[0])
            comp = comp[~pygeos.is_empty(comp)]
            replacements.append(comp)

            # snap
            coords = pygeos.get_coordinates(comp)
            indices = pygeos.get_num_coordinates(comp)

            # generate a list of start and end coordinates and create point geometries
            edges = [0]
            i = 0
            for ind in indices:
                ix = i + ind
                edges.append(ix - 1)
                edges.append(ix)
                i = ix
            edges = edges[:-1]
            component_nodes = pygeos.points(np.unique(coords[edges], axis=0))

            _, to_snap = non_high.sindex.query_bulk(
                high.geometry[union.component_labels == c], predicate="touches"
            )
            snap = pygeos.snap(
                non_high.iloc[np.unique(to_snap)].geometry.values.data,
                pygeos.union_all(component_nodes),
                tolerance,
            )

            removal_non.append(to_snap)
            snapped.append(snap)

    print(f"finalizing. Elapsed time: {time() - s}.")
    clean = non_high.drop(
        non_high.iloc[np.concatenate([a.flatten() for a in removal_non])].index
    )
    final = np.concatenate(
        [
            clean.geometry.values.data,
            np.concatenate([a.flatten() for a in replacements]),
            np.concatenate([a.flatten() for a in snapped]),
        ]
    )
    return gpd.GeoSeries(final, crs=gdf.crs)


def _getAngle(pt1, pt2):
    """
    pt1, pt2 : tuple
    """
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    return math.degrees(math.atan2(y_diff, x_diff))


def _getPoint1(pt, bearing, dist):
    """
    pt : tuple
    """
    angle = bearing + 90
    bearing = math.radians(angle)
    x = pt[0] + dist * math.cos(bearing)
    y = pt[1] + dist * math.sin(bearing)
    return (x, y)


def _getPoint2(pt, bearing, dist):
    """
    pt : tuple
    """
    bearing = math.radians(bearing)
    x = pt[0] + dist * math.cos(bearing)
    y = pt[1] + dist * math.sin(bearing)
    return (x, y)


def _get_line(pt1, pt2, tick_length):
    angle = _getAngle(pt1, pt2)
    line_end_1 = _getPoint1(pt1, angle, tick_length / 2)
    angle = _getAngle(line_end_1, pt1)
    line_end_2 = _getPoint2(line_end_1, angle, tick_length)
    return [line_end_1, line_end_2]


def consolidate_nodes(gdf, tolerance):
    """Return geoemtry with consolidated nodes.

    Replace clusters of nodes with a single node (weighted centroid
    of a cluster) and snap linestring geometry to it. Cluster is
    defined using DBSCAN on coordinates with ``tolerance``==``eps`.

    Does not preserve any attributes, function is purely geometric.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with LineStrings (usually representing street network)
    tolerance : float
        The maximum distance between two nodes for one to be considered
        as in the neighborhood of the other. Nodes within tolerance are
        considered a part of a single cluster and will be consolidated.

    Returns
    -------
    GeoSeries
    """
    from sklearn.cluster import DBSCAN
    import momepy as mm
    import pygeos
    import numpy as np
    from tqdm import tqdm

    # get nodes and edges
    G = mm.gdf_to_nx(gdf)
    nodes, edges = mm.nx_to_gdf(G)

    # get clusters of nodes which should be consolidated
    db = DBSCAN(eps=tolerance, min_samples=2).fit(
        pygeos.get_coordinates(nodes.geometry.values.data)
    )
    nodes["lab"] = db.labels_
    nodes["lab"] = nodes["lab"].replace({-1: np.nan})  # remove unassigned nodes
    change = nodes.dropna().set_index("lab")

    # get pygeos geometry
    geom = edges.geometry.values.data

    # loop over clusters, cut out geometry within tolerance / 2 and replace it
    # with spider-like geometry to the weighted centroid of a cluster
    spiders = []
    midpoints = []
    clusters = change.dissolve("lab")
    cookies = clusters.buffer(tolerance / 2)
    diff = gpd.overlay(edges, gpd.GeoDataFrame(geometry=cookies), how="difference")

    inp_cookies, res_edges = edges.sindex.query_bulk(
        cookies.boundary, predicate="intersects"
    )
    for i, cookie in tqdm(enumerate(cookies.boundary.values.data), total=len(cookies)):
        pts = pygeos.get_coordinates(
            pygeos.intersection(geom[res_edges[inp_cookies == i]], cookie)
        )
        if pts.shape[0] > 0:
            midpoint = np.mean(
                pygeos.get_coordinates(pygeos.from_shapely(clusters.iloc[i].geometry)),
                axis=0,
            )
            midpoints.append(midpoint)
            mids = np.array(
                [
                    midpoint,
                ]
                * len(pts)
            )
            spider = pygeos.linestrings(
                np.array([pts[:, 0], mids[:, 0]]).T,
                y=np.array([pts[:, 1], mids[:, 1]]).T,
            )
            spiders.append(spider)

    # combine geometries
    geometry = np.append(diff.geometry.values.data, np.hstack(spiders))
    geometry = geometry[~pygeos.is_empty(geometry)]
    topological = topology(gpd.GeoSeries(geometry, crs=gdf.crs))

    midpoints = gpd.GeoSeries(pygeos.points(midpoints), crs=gdf.crs)
    return topological, midpoints


# helper functions
def get_ids(x, ids):
    return ids[x]


mp = np.vectorize(get_ids, excluded=["ids"])


def dist(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_verts(x, voronoi_diagram):
    return voronoi_diagram.vertices[x]


def _average_geometry(lines, poly=None, distance=2):
    """
    Returns average geometry.


    Parameters
    ----------
    lines : list
        LineStrings connected at endpoints forming a closed polygon
    poly : shapely.geometry.Polygon
        polygon enclosed by `lines`
    distance : float
        distance for interpolation

    Returns list of averaged geometries
    """
    if not poly:
        poly = box(*lines.total_bounds)
    # get an additional line around the lines to avoid infinity issues with Voronoi
    extended_lines = [poly.buffer(distance).exterior] + list(lines)

    # interpolate lines to represent them as points for Voronoi
    points = np.empty((0, 2))
    ids = []

    pygeos_lines = pygeos.from_shapely(extended_lines)
    lengths = pygeos.length(pygeos_lines)
    for ix, (line, length) in enumerate(zip(pygeos_lines, lengths)):
        pts = pygeos.line_interpolate_point(
            line, np.linspace(0.1, length - 0.1, num=int((length - 0.1) // distance))
        )  # .1 offset to keep a gap between two segments
        points = np.append(points, pygeos.get_coordinates(pts), axis=0)
        ids += [ix] * len(pts)

        # here we might also want to append original coordinates of each line
        # to get a higher precision on the corners, but it does not seem to be
        # necessary based on my tests.

    # generate Voronoi diagram
    voronoi_diagram = Voronoi(points)

    # get all rigdes and filter only those between the two lines
    pts = voronoi_diagram.ridge_points
    mapped = mp(pts, ids=ids)
    rigde_vertices = np.array(voronoi_diagram.ridge_vertices)

    # iterate over segment-pairs
    edgelines = []
    combs = combinations(range(1, len(lines) + 1), 2)
    for a, b in combs:
        mask = (
            np.isin(mapped[:, 0], [a, b])
            & np.isin(mapped[:, 1], [a, b])
            & (mapped[:, 0] != mapped[:, 1])
        )
        verts = rigde_vertices[mask]

        # generate the line in between the lines
        edgeline = pygeos.line_merge(
            pygeos.multilinestrings(get_verts(verts, voronoi_diagram))
        )
        snapped = pygeos.snap(edgeline, pygeos_lines[a], distance)
        edgelines.append(snapped)
    return edgelines


def topology(gdf):
    """
    Clean topology of existing LineString geometry by removal of nodes of degree 2.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries, array of pygeos geometries
        (Multi)LineString data of street network
    """
    if isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        # explode to avoid MultiLineStrings
        # double reset index due to the bug in GeoPandas explode
        df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

        # get underlying pygeos geometry
        geom = df.geometry.values.data
    else:
        geom = gdf

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
    merge = res[np.isin(inp, unique[counts == 2])]

    if len(merge) > 0:
        # filter duplications and create a dictionary with indication of components to
        # be merged together
        dups = [item for item, count in collections.Counter(merge).items() if count > 1]
        split = np.split(merge, len(merge) / 2)
        components = {}
        for i, a in enumerate(split):
            if a[0] in dups or a[1] in dups:
                if a[0] in components.keys():
                    i = components[a[0]]
                elif a[1] in components.keys():
                    i = components[a[1]]
            components[a[0]] = i
            components[a[1]] = i

        # iterate through components and create new geometries
        new = []
        for c in set(components.values()):
            keys = []
            for item in components.items():
                if item[1] == c:
                    keys.append(item[0])
            new.append(pygeos.line_merge(pygeos.union_all(geom[keys])))

        # remove incorrect geometries and append fixed versions
        df = df.drop(merge)
        final = gpd.GeoSeries(new).explode().reset_index(drop=True)
        if isinstance(gdf, gpd.GeoDataFrame):
            return df.append(
                gpd.GeoDataFrame({df.geometry.name: final}, geometry=df.geometry.name),
                ignore_index=True,
            )
        return df.append(final, ignore_index=True)


class ConsolidateEdges:
    """
    Consolidate edges based on specific shape index
    """

    def __init__(self, network):
        from momepy.shape import _circle_area
        from shapely.ops import polygonize

        self.network = network

        polygonized = polygonize(network.geometry.unary_union)
        self.polygons = gpd.GeoDataFrame(
            geometry=[g for g in polygonized], crs=network.crs
        )

        self.index = self.polygons.area / np.sqrt(
            self.polygons.convex_hull.exterior.apply(
                lambda hull: _circle_area(list(hull.coords))
            )
        )
        

    def get_mask(self, threshold):
        self.mask = self.index < threshold
        print(f"{self.mask.sum()} polygons out of {len(self.index)} masked.")
    
    def remove_built_up(self, buildings):
        inp, res = self.polygons.sindex.query_bulk(buildings.geometry, predicate='intersects')
        m2 = self.polygons.index.isin(res)
        self.mask = self.mask & ~m2
        print(f"{self.mask.sum()} polygons out of {len(self.index)} masked.")
    
    def remove_self_loops(self):
        inp, res = pygeos.STRtree(self.polygons.geometry.values.data).query_bulk(
            self.network.geometry.values.data, predicate="covered_by"
        )
        unique, counts = np.unique(res, return_counts=True)
        m2 = self.polygons.index.isin(self.polygons.iloc[unique[counts > 1]].index)
        self.mask = self.mask & m2
        print(f"{self.mask.sum()} polygons out of {len(self.index)} masked.")
        
    def consolidate_nodes(self, tolerance):
        """Consolidate nodes of masked polygons
        
        Replace clusters of nodes with a single node (weighted centroid
        of a cluster) and snap linestring geometry to it. Cluster is
        defined using DBSCAN on coordinates with ``tolerance``==``eps`.

        Does not preserve any attributes, function is purely geometric.

        Parameters
        ----------
        tolerance : float
            The maximum distance between two nodes for one to be considered
            as in the neighborhood of the other. Nodes within tolerance are
            considered a part of a single cluster and will be consolidated.

        Returns
        -------
        GeoSeries
        """
        from sklearn.cluster import DBSCAN
        import momepy as mm
        import pygeos
        import numpy as np
        from tqdm import tqdm
        from momepy.shape import _circle_area

        # get nodes and edges
        G = mm.gdf_to_nx(self.network)
        nodes = mm.nx_to_gdf(G, lines=False)
        
        inp, res = self.polygons[self.mask].sindex.query_bulk(
            nodes.geometry,
            predicate='intersects'
        )
        nodes = nodes.iloc[np.unique(inp)]

        # get clusters of nodes which should be consolidated
        db = DBSCAN(eps=tolerance, min_samples=2).fit(
            pygeos.get_coordinates(nodes.geometry.values.data)
        )
        nodes["lab"] = db.labels_
        nodes["lab"] = nodes["lab"].replace({-1: np.nan})  # remove unassigned nodes
        change = nodes.dropna().set_index("lab")

        # get pygeos geometry
        geom = self.network.geometry.values.data

        # loop over clusters, cut out geometry within tolerance / 2 and replace it
        # with spider-like geometry to the weighted centroid of a cluster
        spiders = []
        midpoints = []
        clusters = change.dissolve("lab")
        cookies = clusters.buffer(tolerance / 2)
        diff = gpd.overlay(self.network, gpd.GeoDataFrame(geometry=cookies), how="difference")

        inp_cookies, res_edges = self.network.sindex.query_bulk(
            cookies.boundary, predicate="intersects"
        )
        for i, cookie in tqdm(enumerate(cookies.boundary.values.data), total=len(cookies)):
            pts = pygeos.get_coordinates(
                pygeos.intersection(geom[res_edges[inp_cookies == i]], cookie)
            )
            if pts.shape[0] > 0:
                midpoint = np.mean(
                    pygeos.get_coordinates(pygeos.from_shapely(clusters.iloc[i].geometry)),
                    axis=0,
                )
                midpoints.append(midpoint)
                mids = np.array(
                    [
                        midpoint,
                    ]
                    * len(pts)
                )
                spider = pygeos.linestrings(
                    np.array([pts[:, 0], mids[:, 0]]).T,
                    y=np.array([pts[:, 1], mids[:, 1]]).T,
                )
                spiders.append(spider)

        # combine geometries
        geometry = np.append(diff.geometry.values.data, np.hstack(spiders))
        geometry = geometry[~pygeos.is_empty(geometry)]
        
        self.consolidated_nodes = topology(gpd.GeoSeries(geometry, crs=self.network.crs))
        self.midpoints = gpd.GeoSeries(pygeos.points(midpoints), crs=self.network.crs)
        
        # regenerate polygons and index based on new geometry
        mask_centroids = self.polygons[self.mask].centroid
        polygonized = polygonize(self.consolidated_nodes.geometry.unary_union)
        self.polygons = gpd.GeoDataFrame(
            geometry=[g for g in polygonized], crs=self.network.crs
        )
        self.index = self.polygons.area / np.sqrt(
            self.polygons.convex_hull.exterior.apply(
                lambda hull: _circle_area(list(hull.coords))
            )
        )
        print("New geometry created. Regenerate mask.")
    
    def consolidate_edges(self, distance=2, epsilon=2):
        """
        distance : float
            distance for interpolation

        epsilon : float
            tolerance for simplification

        self_loops : bool
            consolidate self-loops
        """
        import pandas as pd
        import pygeos

        invalid = self.polygons.loc[self.mask]

        sindex = self.consolidated_nodes.sindex

        # iterate over polygons which are marked to be consolidated
        # list segments to be removed and the averaged geoms replacing them
        averaged = []
        to_remove = []
        for poly in tqdm(invalid.geometry, total=len(invalid)):
            real = self.consolidated_nodes.iloc[
                sindex.query(poly.exterior, predicate="intersects")
            ]
            mask = real.intersection(poly.exterior).type.isin(
                ["LineString", "MultiLineString"]
            )
            real = real[mask]
            lines = real.geometry
            to_remove += list(real.index)

            if lines.shape[0] > 0:
                av = _average_geometry(lines, poly, distance)
                averaged += av

        # drop double lines
        clean = self.consolidated_nodes.drop(set(to_remove))

        # merge new geometries with the existing network
        averaged = (
            gpd.GeoSeries(averaged, crs=self.network.crs).simplify(epsilon).explode()
        )
        result = pd.concat([clean, averaged])
        merge = topology(result)

        self.consolidated_edges = merge

    def hist(self, **kwargs):
        return self.index.plot.hist(**kwargs)

    def plot(self, threshold, **kwargs):
        ax = self.polygons[self.mask].plot(color="r", **kwargs)
        self.network.plot(ax=ax, linewidth=0.1)
        return ax