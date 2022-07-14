settings.render = 3; 
settings.outformat = "pdf"; // "pdf", "png", "svg", "eps"

import three;


// === GRAPHICS ===
size(300);
currentprojection = orthographic((0, 0, 1), up = (0, 0, 1));
currentlight = light(
  diffuse = new pen[] {gray(1.0), gray(0.6)},     // brighter light
  position = new triple[] {(2, 2, 3), (-2, -1, 2)} // same directions
);



// === SETTINGS ===
int num_points_per_curve = 10; // number of points per curve in the surface patch
string user = substr(settings.user, 0, length(settings.user) - 1);

if(user == ""){
  user = "3";
}

string folder = "./results/ref" + user + "/solution/" ;


// === HELPERS ===
real clamp(real x, real xmin, real xmax) {
  return max(xmin, min(x, xmax));
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


triple origin = O - 0.008*(1,1,0);//0.8*(1,-1,-1); // bottom-left corner of the merged surface

real axisLength = .15; // adjust as needed

draw(origin -- (origin + (axisLength,0,0)), Arrow3(5bp)); label("$x$", origin + (axisLength,-0.02,0),fontsize(11pt));
draw(origin -- (origin + (0,axisLength,0)), Arrow3(5bp)); label("$y$", origin + (-0.02,axisLength,0),fontsize(11pt));
draw(origin -- (origin + (0,0,axisLength)), Arrow3(5bp)); label("$z$", origin + (-0.02,-0.02,0),fontsize(11pt));


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
write("Min value (blue): " + string(minScalar) );
write("Max value (red): " + string(maxScalar) );



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


