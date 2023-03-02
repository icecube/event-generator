from __future__ import print_function, division

def point_in_polygon(x, y, cx, cy):
    """
    Determine if points are inside the polygon defined by vertices (cx, cy)
    by casting a ray to the right of the point and counting its intersections
    with the edges of the polygon. If it crosses an odd number of times, the
    point is inside the polygon.

    :param x: x coordinate(s) of test point(s)
    :param y: y coordinate(s) of test point(s)
    :param cx: x coordinates of polygon vertices
    :param cy: x coordinates of polygon vertices

    See: http://paulbourke.net/geometry/insidepoly/
    """
    import numpy

    x = numpy.asarray(x)
    y = numpy.asarray(y)
    cx = numpy.asarray(cx); cx = cx.reshape((cx.size, 1))
    cy = numpy.asarray(cy); cy = cy.reshape((cy.size, 1))

    # Coordindates of the "next" vertex. Since the polygon is closed, the
    # last vertex is next to the first.
    nx = numpy.roll(cx, 1, axis=0)
    ny = numpy.roll(cy, 1, axis=0)

    # Draw a horizontal line at y. Which edges does it intersect?
    crossings = ((cy <= y)&(y < ny))|((ny <= y)&(y < cy))

    # Now, cast a ray to the right. Which edges does it intersect?
    crossings &= (x < (nx-cx)*(y-cy)/(ny-cy) + cx)

    # Count the number of crossings.
    inside = (crossings.sum(axis=0) % 2) != 0
    if inside.size == 1:
        return inside[0]
    else:
        return inside

def get_demo_polygon(size, gcdfile=None):
    import numpy

    if (gcdfile):
        from icecube import icetray, dataio, dataclasses
        from .mlb_CutParams import RingFinder
        f = dataio.I3File(gcdfile)
        fr = f.pop_frame()
        geo = fr['I3Geometry'].omgeo

        xe = list(); ye = list()
        for string in RingFinder().ring3:
            key = icetray.OMKey(string, 1)
            g = geo[key]
            print(key)
            xe.append(g.position.x)
            ye.append(g.position.y)

        xe = numpy.array(xe)
        ye = numpy.array(ye)
        order = numpy.argsort(numpy.arctan2(ye, xe))
        xe = xe[order]
        ye = ye[order]

        xs = numpy.random.uniform(low=-600, high=600, size=size)
        ys = numpy.random.uniform(low=-600, high=600, size=size)

    else:
        xe = numpy.array([ 2,  -1, -1, -2, -1,  1])
        ye = numpy.array([ 0,   0,  1,  0, -1, -1])

        xs = numpy.random.uniform(low=-3, high=3, size=size)
        ys = numpy.random.uniform(low=-2, high=2, size=size)

    return xe, ye, xs, ys


def demo(gcdfile=None):
    import pylab
    import numpy

    xe, ye, xs, ys = get_demo_polygon(1000, gcdfile)

    c = point_in_polygon(xs, ys, xe, ye)
    nc = numpy.logical_not(c)

    fig = pylab.figure()
    pylab.scatter(xs[c], ys[c], c='r')
    pylab.scatter(xs[nc], ys[nc], c='b')
    pylab.plot(xe, ye, c='k')
    pylab.plot(numpy.roll(xe, 1)[:3], numpy.roll(ye, 1)[:3], c='k')
    pylab.gca().set_aspect('equal', 'datalim')

