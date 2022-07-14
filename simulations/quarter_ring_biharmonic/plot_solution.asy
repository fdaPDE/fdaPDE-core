
// PARTE MIA

settings.render = 3; 
//settings.prc = true; // if true animation 3d active

import three;
import graph3;
import settings;
import plain;



size(250);
//currentprojection = perspective((0,5,-10),up=(0,1,0));
// sphere ,showtarget=true, autoadjust=false, center=true
currentprojection = orthographic((0, 0, 1), up = (0, 0, 1));
//currentprojection = perspective((14, 10,3),up=(0,0,1)); // sfera 100 toro 010 , up = (1,1,0)
//currentprojection = perspective((5, 5, 5));
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
pen interiorEdgePen = gray + 1bp;
pen boundaryEdgePen = blue + 1.2bp;
pen quadPen = lightblue ;

//folder = folder + "/";

string user = substr(settings.user, 0, length(settings.user) - 1);

if(user == ""){
  user = "3";
}

string folder = "./results/ref" + user + "/solution/" ;

// eliminate the last caracter of the string



// === HELPERS ===
real clamp(real x, real xmin, real xmax) {
  return max(xmin, min(x, xmax));
}

pen colormap(real t) {
  t = clamp(t, 0, 1);

  // Bordeaux and deep blue
  real blueR = 0.15, blueG = 0.2, blueB = 0.7;
  real redR  = 0.5,  redG  = 0.1, redB  = 0.2;
  real grayR = 0.6,  grayG = 0.6, grayB = 0.6;

  real r, g, b;

  if (t < 0.5) {
    real k = t / 0.5;
    r = (1 - k) * blueR + k * grayR;
    g = (1 - k) * blueG + k * grayG;
    b = (1 - k) * blueB + k * grayB;
  } else {
    real k = (t - 0.5) / 0.5;
    r = (1 - k) * grayR + k * redR;
    g = (1 - k) * grayG + k * redG;
    b = (1 - k) * grayB + k * redB;
  }

  return rgb(r, g, b);
}

pen colormap2(real t) {
  t = clamp(t, 0, 1);
  real r, g, b;

  if (t < 0.2) {
    // Blue to Cyan
    real k = t / 0.2;
    r = 0.0;
    g = k;
    b = 1.0;
  }
  else if (t < 0.4) {
    // Cyan to Green
    real k = (t - 0.2) / 0.2;
    r = 0.0;
    g = 1.0;
    b = 1.0 - k;
  }
  else if (t < 0.6) {
    // Green to Yellow
    real k = (t - 0.4) / 0.2;
    r = k;
    g = 1.0;
    b = 0.0;
  }
  else if (t < 0.8) {
    // Yellow to Orange
    real k = (t - 0.6) / 0.2;
    r = 1.0;
    g = 1.0 - 0.5 * k;
    b = 0.0;
  }
  else {
    // Orange to Red
    real k = (t - 0.8) / 0.2;
    r = 1.0;
    g = 0.5 - 0.5 * k;
    b = 0.0;
  }

  return rgb(r, g, b);
}

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



triple origin = O; //0.8*(1,-1,-1); // bottom-left corner of the merged surface

real axisLength = .2; // adjust as needed

draw(origin -- (origin + (axisLength,0,0)), Arrow3(5bp)); label("$x$", origin + (axisLength+0.04,0,0),fontsize(9pt));
draw(origin -- (origin + (0,axisLength,0)), Arrow3(5bp)); label("$y$", origin + (0,axisLength+0.04,0),fontsize(9pt));
draw(origin -- (origin + (0,0,axisLength)), Arrow3(5bp)); label("$z$", origin + (-0.04,-0.04,0),fontsize(9pt));


// === LOAD & PLOT SURFACE PATCHES ===
file surfFile = input(folder + "refined_surface_points.csv");
triple[][][] grid; // [cell][i][j]
real[][][] scalarGrid; // scalar values at each point
int last_cid = -1;
int cid_index = -1;
int N = num_points_per_curve - 1;

real minScalar = 1e9;
real maxScalar = -1e9;

while (!eof(surfFile)) {
  string line = surfFile;
  string[] p = split(line, ",");

  if (p.length >= 7) { // now includes scalar value
    int cid = (int) p[0];
    int i = (int) p[1];
    int j = (int) p[2];
    real x = (real) p[3];
    real y = (real) p[4];
    real z = (real) p[5];
    real s = (real) p[6];

    if (cid != last_cid) {
      grid.push(new triple[N+1][N+1]);
      scalarGrid.push(new real[N+1][N+1]);
      cid_index += 1;
      last_cid = cid;
    }

    grid[cid_index][i][j] = (x, y, z);
    scalarGrid[cid_index][i][j] = s;

    if (s < minScalar) minScalar = s; //s
    if (s > maxScalar) maxScalar = s; //s
  }
}

// print the min and max scalar values
write("Min scalar: " + string(minScalar) );
write("Max scalar: " + string(maxScalar) );



surface wholeSurface;

// === PLOT SURFACE ===

for (int c = 0; c < grid.length; ++c) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      triple p1 = grid[c][i][j];
      triple p2 = grid[c][i+1][j];
      triple p3 = grid[c][i+1][j+1];
      triple p4 = grid[c][i][j+1];

      real s1 = scalarGrid[c][i][j];
      real s2 = scalarGrid[c][i+1][j];
      real s3 = scalarGrid[c][i+1][j+1];
      real s4 = scalarGrid[c][i][j+1];

      // Triangle 1: p1-p2-p3
      real t1 = ((s1 + s2 + s3) / 3 - minScalar) / (maxScalar - minScalar + 1e-10);
      pen color1 = colormap2(t1);
      draw(surface(p1--p2--p3--cycle),      surfacepen = material(
        diffusepen = color1              // gives surface its color under light
        ,emissivepen = gray(0.1)       // Helps in shadows
        ,specularpen = black        // Mild highlight
      ));

      // Triangle 2: p1-p3-p4
      real t2 = ((s1 + s3 + s4) / 3 - minScalar) / (maxScalar - minScalar + 1e-10);
      pen color2 = colormap2(t2);
      draw(surface(p1--p3--p4--cycle), surfacepen = material(
        diffusepen = color2
        ,emissivepen = gray(0.1)      // Helps in shadows
        ,specularpen = black        // Mild highlight
      ));
    }
  }
}
//material Spen  = material(white+opacity(0.8),emissivepen=gray(0.05),specularpen =mediumgray);

//draw(wholeSurface,surfacepen=Spen,render(compression=Low,merge=true));

// === PLOT CURVED EDGES ===
/*

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


  
  
*/

  



// === LOAD CONTROL POINTS GRID ===
/*

pen visibleLine = gray + 0.8bp;
pen hiddenLine = gray + opacity(0.5) + 0.8bp;
pen visibleDot = gray + 3bp;
pen hiddenDot = gray + 3bp + opacity(0.6);

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




