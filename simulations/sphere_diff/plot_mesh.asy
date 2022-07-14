
// PARTE MIA

settings.render = 4; 
settings.outformat = "pdf"; // "pdf", "png", "svg", "eps"
//settings.prc = true; // if true animation 3d active

import three;
import graph3;
import settings;
import plain;


//currentprojection = perspective((10,10,10));
// sphere 
//currentprojection = perspective(12*(1, 1, 1),  showtarget=true, autoadjust=false, center=true);
//defaultrender = render(merge = true);
//currentlight = (5, 10, 10);


size(230);
currentprojection = perspective((14, 10,10),up=(0,0,1));
//currentprojection = orthographic((0,0,1),up=(0,0,1));
//defaultrender = render(merge = true);
//currentlight = Viewport;

// Soft, frontal lighting + light fill from behind

currentlight = light(
  diffuse = new pen[] {gray(1.0), gray(0.6)},     // brighter light
  //specular = new pen[] {gray(0.3), gray(0.2)},    // subtle highlight
  position = new triple[] {(2, 2, 3), (-2, -1, 2)} // same directions
);


// === SETTINGS ===
int num_points_per_curve = 10;
pen interiorEdgePen = black + 0.5bp;
pen boundaryEdgePen = interiorEdgePen;//red + 1.6bp;
pen quadPen = lightblue ;

user = substr(settings.user, 0, length(settings.user) - 1);

if(user == ""){
  user = "3";
}

string folder = "./results/ref" + user + "/solution/" ;


// === HELPERS ===

triple[] loadTriples(string filename) {
  triple[] result;
  file f = input(filename);
  while (!eof(f)) {
    string line = f;
    string[] p = split(line);
    if (p.length >= 3)
      result.push(((real) p[0], (real) p[1], (real) p[2]));
  }
  return result;
}

int[][] loadEdgeList(string filename) {
  int[][] edges;
  file f = input(filename);
  while (!eof(f)) {
    string line = f;
    string[] p = split(line);
    if (p.length >= 2)
      edges.push(new int[] {((int) p[0] ) , ((int) p[1]) }); // Convert to 1-based indexing
  }
  return edges;
}

int[] loadFlags(string filename) {
  int[] flags;
  file f = input(filename);
  while (!eof(f)) {
    string line = f;
    string[] p = split(line);
    if (p.length >= 1)
      flags.push((int) p[0]);
  }
  return flags;
}

// === LOAD DATA ===
triple[] nodes = loadTriples(folder + "nodes.txt");
triple[] nurbs_edges = loadTriples(folder + "edge_refinement.txt");
int[][] edges = loadEdgeList(folder + "edges.txt");
int[] bflags = loadFlags(folder + "boundary_edges.txt");


triple origin = 0.8*(1,-1,-1); // bottom-left corner of the merged surface

real axisLength = .2; // adjust as needed

draw(origin -- (origin + (axisLength,0,0)), Arrow3(5bp)); label("$x$", origin + (axisLength+0.04,0,0),fontsize(9pt));
draw(origin -- (origin + (0,axisLength,0)), Arrow3(5bp)); label("$y$", origin + (0,axisLength+0.04,0),fontsize(9pt));
draw(origin -- (origin + (0,0,axisLength)), Arrow3(5bp)); label("$z$", origin + (0,0,axisLength+0.04),fontsize(9pt));;


// === LOAD & PLOT SURFACE PATCHES ===

file surfFile = input(folder + "refined_surface_points.csv");
triple[][][] grid; // [cell][i][j]
int last_cid = -1;
int cid_index = -1;
int N = num_points_per_curve - 1;




while (!eof(surfFile)) {
  string line = surfFile;
  string[] p = split(line, ",");

  if (p.length >= 6) {
    int cid = (int) p[0];
    int i = (int) p[1];
    int j = (int) p[2];
    real x = (real) p[3];
    real y = (real) p[4];
    real z = (real) p[5];

    if (cid != last_cid) {
      grid.push(new triple[N+1][N+1]);
      cid_index += 1;
      last_cid = cid;
    }

    grid[cid_index][i][j] = (x, y, z);
  }
}
surface wholeSurface;

// === PLOT SURFACE ===

for (int c = 0; c < grid.length; ++c) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      triple p1 = grid[c][i][j];
      triple p2 = grid[c][i+1][j];
      triple p3 = grid[c][i+1][j+1];
      triple p4 = grid[c][i][j+1];

       triple[][] quad = new triple[2][2];
        quad[0][0] = p1;
        quad[0][1] = p2;
        quad[1][0] = p4;
        quad[1][1] = p3;
        //draw(surface(quad), quadPen);
        surface patch = surface(quad);
          wholeSurface = surface(wholeSurface, patch); // merge the new patch
    }
    }
  }

material Spen  = material(paleblue+opacity(0.9),emissivepen=gray(0.01),specularpen =black);

draw(wholeSurface,surfacepen=Spen,render(compression=Low,merge=true));


// === PLOT CURVED EDGES ===
int num_edges = edges.length;
for (int i = 0; i < num_edges; ++i) {
  triple[] curve;
  int start = i * num_points_per_curve;
  for (int j = 0; j < num_points_per_curve; ++j) {
    triple pt = nurbs_edges[start + j];
    curve.push(pt); // lift radially from origin
  }

  // Select pens
  pen edgePenVisible = bflags[i] == 1 ? boundaryEdgePen : interiorEdgePen;
  pen edgePenHidden = edgePenVisible + opacity(0.2); // faded version

  // Draw each segment with visibility check
  for (int j = 0; j < curve.length - 1; ++j) {
    triple p = curve[j];
    triple q = curve[j + 1];
    draw(p -- q, edgePenVisible );
  }
}
  
/*

// === LOAD CONTROL POINTS GRID ===

pen visibleLine = black + 0.8bp;
pen hiddenLine = black + opacity(0.5) + 0.8bp;
pen visibleDot = black + 5bp;
pen hiddenDot = black + 5bp + opacity(0.6);

triple[] control_points_flat;
int num_rows = 0;
int num_cols = 0;

file fcp = input(folder + "control_points.txt");
bool firstLine = true;

while (!eof(fcp)) {
  string line = fcp;
  string[] p = split(line);

  if (firstLine) {
    if (p.length >= 2) {
      num_rows = (int) p[0];
      num_cols = (int) p[1];
      firstLine = false;
    }
    continue;
  }

  if (p.length >= 3)
    control_points_flat.push(((real) p[0], (real) p[1], (real) p[2]));
}

// === RECONSTRUCT GRID ===
if (control_points_flat.length != num_rows * num_cols) {
  write("Error: mismatch in control point count.");
} else {
  triple[][] control_grid;
  int idx = 0;
  for (int i = 0; i < num_rows; ++i) {
    triple[] row;
    for (int j = 0; j < num_cols; ++j) {
      row.push(control_points_flat[idx]);
      idx = idx+1;
    }
    control_grid.push(row);
  }
  
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      triple p = control_grid[i][j];

      if (j + 1 < num_cols) {
        triple q = control_grid[i][j + 1];
        draw(p -- q, visibleLine);
      }

      if (i + 1 < num_rows) {
        triple q = control_grid[i + 1][j];
        draw(p -- q, visibleLine);
      }

      dot(p, visibleDot);
    }
  }
}

*/




