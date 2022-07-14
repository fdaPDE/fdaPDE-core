import graph;

settings.outformat = "pdf";

size(500, 400, IgnoreAspect); // Match other figures
scale(Log, Log);

// Load data
string path = "./results/L2_error.csv";
file f = input(path);

string header = f;

real[] x, y, z;
while (!eof(f)) {
  string line = f;
  if (line == "") continue;
  string[] fields = split(line, ",");
  if (fields.length < 3) continue;
  x.push((real)fields[0]);
  y.push((real)fields[1]);
  z.push((real)fields[2]); // H1 error
}

// Log-space padding
real x_factor = 1.5, y_factor = 3;
real x_min = min(x), x_max = max(x);
real y_min = min(min(y), min(z)), y_max = max(max(y), max(z));
xlimits(x_min / x_factor, x_max * x_factor);
ylimits(y_min / y_factor, y_max * y_factor);

// Font size
real font = 17pt;

// Plot L2 error
marker markL2 = marker(scale(1.4mm)*unitcircle, red, Fill);
pen dataPenL2 = red + 2bp;
Label err_label = Label("$L^2$", fontsize(font));
draw(graph(x, y), dataPenL2, err_label, markL2);

// Plot H1 error
marker markH1 = marker(scale(1.4mm)*unitcircle, blue, Fill);
pen dataPenH1 = blue + 2bp;
Label err_labelH1 = Label("$H^1$", fontsize(font));
draw(graph(x, z), dataPenH1, err_labelH1, markH1);

// Reference slope lines
real x1 = 0.18, x2 = 0.045;
real[] refx = {x1, x2};

// h^3 line (L2 reference)
real y_ref = 0.007;
real[] refy3 = {y_ref, y_ref * (refx[1]/refx[0])^3};
pen refPen3 = rgb(1, 0.6, 0.6) + linetype("4 2") + 1.9bp;
Label h3_label = Label("$ h^3$", fontsize(font));
draw(graph(refx, refy3), refPen3, h3_label );

// h^2 line (H1 reference)
real y_ref2 = 0.4;
real[] refy2 = {y_ref2, y_ref2 * (refx[1]/refx[0])^2};
pen refPen2 = rgb(0.6, 0.6, 1) + linetype("4 2") + 1.9bp;
Label h2_label = Label("$ h^2$", fontsize(font));
draw(graph(refx, refy2), refPen2, h2_label );

// Axis ticks and grids
pen thin = invisible;
pen thick = gray + linetype("0 2") + linewidth(0.9);
pen minorTickPen = gray + 0.4bp;

Label xlabel = shift(0, -1.2)*Label("$h$", fontsize(font));
Label ylabel = shift(3mm*W)*rotate(90)*Label("Error", fontsize(font));

xaxis(xlabel, BottomTop,
  LeftTicks(Label(fontsize(font)), begin=true, end=true, extend=true, ptick=thin, pTick=thick));
xaxis("", BottomTop,
  LeftTicks(format="%", ticklabel=null, ptick=minorTickPen, pTick=minorTickPen, extend=false));

yaxis(ylabel, LeftRight,
  RightTicks(Label(fontsize(font)), begin=true, end=true, extend=true, ptick=thin, pTick=thick));
yaxis("", LeftRight,
  RightTicks(format="%", ticklabel=null, ptick=minorTickPen, pTick=minorTickPen, extend=false));

// Legend and title
attach(legend(linelength=30bp, 1), point(SE), -15S + 16W, UnFill);
label(shift(2mm*N)*Label("\textbf{Sphere: } $p=2$", fontsize(font)), point(N), N);