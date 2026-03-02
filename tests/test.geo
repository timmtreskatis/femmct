// Rectangular channel
Lx = 5;
Ly = 1;

nx = 10;  // divisions in x-direction
ny = 2;   // divisions in y-direction

// Define corner points
Point(1) = {0,   0, 0};
Point(2) = {Lx,  0, 0};
Point(3) = {Lx, Ly, 0};
Point(4) = {0, Ly, 0};

// Define lines forming the rectangle
Line(1) = {1,2}; // bottom wall
Line(2) = {2,3}; // outlet
Line(3) = {3,4}; // top wall
Line(4) = {4,1}; // inlet

// Structured grid
Transfinite Line{1, 3} = nx + 1;
Transfinite Line{2, 4} = ny + 1;

// Create line loop and surface
Line Loop(1)   = {1,2,3,4};
Plane Surface(1) = {1};

// Create quadrangles
Transfinite Surface{1} AlternateLeft;
//Recombine Surface{1};

// Subdivide each quad into 4 triangles
//Mesh.SubdivisionAlgorithm = 1;
//SubdivideMesh;

// Physical groups with integer tags
Physical Surface("FluidDomain", 10)   = {1};
Physical Line("Inlet",        1)     = {4};
Physical Line("Walls",        2)     = {1,3};
Physical Line("Outlet",       3)     = {2};
