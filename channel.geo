// Rectangular channel: length = 10, width = 1
Lx = 10;
Ly = 1;

// Characteristic length (target edge size)
lc = 0.1;

// Define corner points
Point(1) = {0,   0, 0, lc};
Point(2) = {Lx,  0, 0, lc};
Point(3) = {Lx, Ly, 0, lc};
Point(4) = {0, Ly, 0, lc};

// Define lines forming the rectangle
Line(1) = {1,2}; // bottom wall
Line(2) = {2,3}; // outlet
Line(3) = {3,4}; // top wall
Line(4) = {4,1}; // inlet

// Create line loop and surface
Line Loop(1)   = {1,2,3,4};
Plane Surface(1) = {1};

// Set mesh options for triangles only (no recombination)
Mesh.RecombineAll       = 0;
Mesh.Algorithm          = 6; // Delaunay algorithm
Mesh.CharacteristicLengthMax = lc;

// Physical groups with integer tags
Physical Surface("FluidDomain", 10)   = {1};
Physical Line("Inlet",        1)     = {4};
Physical Line("Walls",        2)     = {1,3};
Physical Line("Outlet",       3)     = {2};
