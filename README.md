# python-stl-renderer

A rudimentary python-based stl renderer for stl files. This script takes a generic .stl file, reads it and parses it into triangles, and then projects the triangles onto a viewing screen, computing intersections and overlaps, and applies shading as appropriate.

## Usage

To run the script, simple clone the repository and run

```
python render.py [myfile.stl]
```

where myfile.stl is your favorite binary stl file. The library provides an example teapot stl file in the examples folder, so you can run

```
python render.py examples/teapot.stl
```

to generate a plot and save it to the directory.

## Dependencies

This script depends on matplotlib, numpy, and the stl python libraries. To install these, run

```
pip install matplotlib numpy stl
```

from a command line.
