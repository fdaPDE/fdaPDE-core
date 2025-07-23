# LIBRARIES PATHS

EIGEN_DIR := /home/francesca16/eigen
FDAPDE_DIR := fdaPDE

SYS_INCLUDE_DIR := /usr/include
ARCH_INCLUDE_DIR := /usr/include/x86_64-linux-gnu
NLOHMANN_DIR := /usr/include/nlohmann

TRIANGLE_SRC := Meshes/Test_triangle/triangle.c
TRIANGLE_NAME := skyline
TRIANGLE_OUT := Meshes/Test_triangle/triangle_standalone
TRIANGLE_POLY := Meshes/Test_triangle/$(TRIANGLE_NAME).poly

FMESHER_SCRIPT := Meshes/Test_fmesher/star.r

# FLAGS                  
CXXFLAGS := -std=c++20 -g -march=native -O2 -DFDAPDE_NO_DEBUG 
#CXXFLAGS := g++ -fsanitize=address -g -O1
INCLUDES := -I$(EIGEN_DIR) -I$(FDAPDE_DIR) -I$(SYS_INCLUDE_DIR) -I$(ARCH_INCLUDE_DIR) -I$(NLOHMANN_DIR)

# Run all targets
.PHONY: all
all: delaunay conflict refinement triangle fmesher comparisons

# Plot a Delaunay mesh
.PHONY: delaunay
delaunay:
	g++ $(CXXFLAGS) $(INCLUDES) -o Meshes/Delaunay/main Meshes/Delaunay/main.cpp
	Meshes/Delaunay/main
	python3 Meshes/Delaunay/plot_mesh.py

# Efficiency of Delaunay conflict graph algorithm 
.PHONY: conflict
conflict: 
	g++ $(CXXFLAGS) $(INCLUDES) -o Meshes/Delaunay/conflict Meshes/Delaunay/conflict_efficiency.cpp
	Meshes/Delaunay/conflict
	python3 Meshes/Delaunay/plot_timing.py

# Efficiency of delaunay refinement algorithm
.PHONY: refinement
refinement:
	g++ $(CXXFLAGS) $(INCLUDES) -o Meshes/Delaunay/refinement Meshes/Delaunay/refinement_efficiency.cpp
	./Meshes/Delaunay/refinement
	python3 ./Meshes/Delaunay/plot_timing.py


# Plot a Triangle mesh, and calculate Triangle efficiency
.PHONY: triangle
triangle:
	gcc $(TRIANGLE_SRC) -o $(TRIANGLE_OUT) -lm
	time $(TRIANGLE_OUT) -pq20a1 $(TRIANGLE_POLY)
	python3 Meshes/Test_triangle/plot_mesh.py $(TRIANGLE_NAME)
	Meshes/Test_triangle/timing_triangle.sh

# Plot a fmesher mesh, and calculate fmesher efficiency
.PHONY: fmesher
fmesher:
	docker run --rm -v $(PWD):/mnt rocker/geospatial:latest /bin/bash -c '\
		R --vanilla -e "install.packages(\"fmesher\", repos=\"https://cloud.r-project.org\")" && \
		Rscript /mnt/$(FMESHER_SCRIPT) && \
		Rscript /mnt/Meshes/Test_fmesher/rectangle_timing.r'	


# Plot quality metrics of the 3 meshes 
.PHONY: comparisons
comparisons:
	python3 Comparisons/Statistics/statistics_analysis.py

# Cleaning directories
.PHONY: clean
clean:
	@rm -f Meshes/Test_fdaPDEmesher/main  Meshes/Test_fdaPDEmesher/conflict Meshes/Test_fdaPDEmesher/refinement
	@rm -f Meshes/Test_fdaPDEmesher/*.csv
	@rm -f Meshes/Test_triangle/*.csv
	@rm -f $(TRIANGLE_OUT) Meshes/Test_triangle/triangle
	@rm -f Meshes/Test_triangle/*.1.node
	@rm -f Meshes/Test_triangle/*.1.ele
	@rm -f Meshes/Test_triangle/*.1.poly
	@rm -f Meshes/Test_fmesher/*.csv
	@rm -f workingdir/test/script


