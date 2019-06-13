from collections import namedtuple
import rasterio as rio
from rasterio.windows import Window
from shapely.geometry import box as Box


def extract_patches(
    image, polygons, shape, stride=None, threshold=0.5
):
    """Extract patches from a geocoded image

    Args:
        image: Filename of geocoded image or rasterio.Dataset
        polygons: GeoDataFrame with labelled polygons.
        shape: Desired shape of the extracted patches. Is also used to
            calculate the number of extracted patches.
        stride:
        threshold: How many points of the extracted rectangle should be covered
            by the labelled polygon? Must be a float be 0 and 1.
    Yields:
        A tuple of two elements: the index of the polygon and the corresponding
        patch.
    """
    if stride is None:
        stride = shape

    image_bounds = Box(*image.bounds)
    BoundingBox = namedtuple(
        'BoundingBox',
        ['col_start', 'col_end', 'row_start', 'row_end', 'height', 'width']
    )

    # We want to extract as many rectangles from the labelled polygons as possible.
    # We are working with two coordinate systems: the index system (row, column) to
    # extract the pixel values from the image matrix and the geo-referenced
    # coordinates (x,y) to find the patches corresponding to the labelled polygons.
    for idx, polygon in polygons.iterrows():
        if not polygon.geometry.is_valid:
            print('Polygon {} is not valid!'.format(idx))
            continue
        
        if not polygon.geometry.intersects(image_bounds):
            continue

        # Extract the bounding box of the polygon, so we can divide it easily
        # into sub-rectangles:
        row1, col1 = image.index(polygon.geometry.bounds[0], polygon.geometry.bounds[1])
        row2, col2 = image.index(polygon.geometry.bounds[2], polygon.geometry.bounds[3])
        polygon_bbox = BoundingBox(
            col_start=min([col1, col2]), col_end=max([col1, col2]),
            row_start=min([row1, row2]), row_end=max([row1, row2]),
            height=abs(row2-row1), width=abs(col2-col1)
        )

        for column in range((polygon_bbox.width // stride[0]) + 1):
            for row in range((polygon_bbox.height // stride[1]) + 1):
                # Transform the patch bounding box indices to coordinates to
                # calculate the percentage that is covered by the labelled
                # polygon:
                x1, y1 = image.xy(
                    polygon_bbox.row_start + row*stride[1],
                    polygon_bbox.col_start + column*stride[0]
                )
                x2, y2 = image.xy(
                    polygon_bbox.row_start + row*stride[1] + shape[1],
                    polygon_bbox.col_start + column*stride[0] + shape[1]
                )
                patch_bounds = Box(x1, y1, x2, y2)

                # We check first whether the threshold condition is fullfilled
                # and then we extract the patch from the image:
                overlapping_area = \
                    polygon.geometry.intersection(patch_bounds).area
                if overlapping_area / patch_bounds.area <  threshold:
                    # The labelled polygon covers less than $threshold$ of the
                    # whole patch, i.e. we reject it:
                    continue

                # rasterio returns data in CxHxW format, so we have to transpose
                # it:
                patch = image.read(
                    window=Window(
                        polygon_bbox.col_start + column*stride[0],
                        polygon_bbox.row_start + row*stride[1],
                        shape[1], shape[0]
                    )
                ).T
                
                # The fourth channel shows the alpha (transparency) value. We do not 
                # allow patch with transparent pixels: 
                if not patch.size or (patch.shape[-1] == 4 and 0 in patch[..., 3]):
                    continue
                    
                yield idx, patch
