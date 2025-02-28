#ifndef _FDAPDE_DELAUNAY_H_
#define _FDAPDE_DELAUNAY_H_

// delaunay class implementing Constrained Delaunay Triangulations (CDTs),
// capable of handling concave domains, internal holes, and subregions.
// Also supports mesh refinement with constraints on minimum angle and maximum area.

// delaunay class implementing Constrained Delaunay Triangulations (CDTs),
// capable of handling concave domains, internal holes, and subregions.
// Also supports mesh refinement with constraints on minimum angle and maximum area.

#include "header_check.h"
namespace fdapde {
  
template <int LocalDim, int EmbedDim>
class Delaunay {
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    static constexpr int n_nodes_cell = 3;

    using coords_t = Eigen::Matrix<double, 1, embed_dim>;
    using node_t = typename DCEL<local_dim, embed_dim>::node_t;
    using halfedge_t = typename DCEL<local_dim, embed_dim>::halfedge_t;
    using cell_t = typename DCEL<local_dim, embed_dim>::cell_t;
    using triangulation_t = TriangulationBase<local_dim, embed_dim, Triangulation<local_dim,embed_dim>>;
    using dcel_t = DCEL<local_dim, embed_dim>;
    using polygon_t = Polygon<local_dim, embed_dim>;


    // costructor with random generated points and refinement
    Delaunay(const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& boundaries, double min_angle, double max_area, int N=0, const std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>>& holes = {{}}){
        triangulate_(N, boundaries, holes);
        refinement(min_angle, max_area);
        check_quality_(min_angle, max_area);
    }  
    // costructor with random generated points, no refinement
    Delaunay(const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& boundaries, int N=0, const std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>>& holes = {{}}){
        triangulate_(N, boundaries, holes);
    }   
    // costructor with given internal points from the user, and refinement
    // user needs to provide internal points correctly located inside the domain 
    Delaunay(const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& boundaries, const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& internal, double min_angle, double max_area, const std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>>& holes = {{}}) {
        triangulate_(internal, boundaries, holes);
        refinement(min_angle, max_area);
        check_quality_(min_angle, max_area);
    }
    // costructor with given internal points from the user, no refinement
    Delaunay(const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& boundaries, const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& internal, const std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>>& holes = {{}}) {
        triangulate_(internal, boundaries, holes);
    }

    // getter const, to ensure delaunay triangulation is preserved
    const dcel_t& dcel() const {
        return dcel_;
    }

    triangulation_t triangulation() const{
        return dcel_.template to_triangulation<triangulation_t>();
    }

    // function computing the total area of the domain if the user wants to use it to impose 
    // the maximum area constraint in the refinement internally 
    // function computing the total area of the domain if the user wants to use it to impose 
    // the maximum area constraint in the refinement internally 
    double domain_area() const{
        double total_area = 0.0;
        for (auto it = dcel_.cells_cbegin(); it != dcel_.cells_cend(); ++it) {
            const cell_t* t = &(*it);
            coords_t A = t->halfedge()->prev()->node()->coords();
            coords_t B = t->halfedge()->node()->coords();
            coords_t C = t->halfedge()->next()->node()->coords();
            total_area += fdapde::internals::measure_2d_tri(A, B, C);
        }
        return total_area;
    }


    // function running refinement algorithm
    void refinement(double min_angle, double max_area) {
        // set of edges that are encroached (i.e. internal points are present in the diametral lens of the corresponding edge)
        std::unordered_set<halfedge_t*> encroached_segments;

        // set of triangles that do not meet the quality criteria (angle/area), stored with priority given by the key of the multimap
        // a multimap is implemented in order to correctly store different triangles with same priority 
        std::multimap<double, cell_t*> bad_triangles;

        // set of all boundary edges (marked as segments)
        std::unordered_set<halfedge_t*> segments;

        // initialize encroached and boundary edges
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
            halfedge_t* e = &(*it);

            // skip halfedges of the external part of the boundary 
            if (e->on_boundary() && !e->cell()) continue;

            // a segment characterizes edges of the boundary of the whole domain, of holes and of internal regions
            if (e->is_segment()) {
                segments.insert(e);

                // if the edge is encroached, add it to the set
                if (check_encroachment_(e)) {
                    encroached_segments.insert(e);
                }
            }
        }

        // initialize bad triangle set based on area and angle criteria 
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            cell_t* t = &(*it);
            double priority = is_bad_triangle_(t, min_angle, max_area);
            if (priority >= 0.0 &&
                std::find_if(bad_triangles.begin(), bad_triangles.end(),
                            [t](const auto& entry) { return entry.second == t; }) == bad_triangles.end()) {
                bad_triangles.insert({priority, t});
            }
        }

        // iterative refinement loop 
        // continues until there are no more encroached edges or bad triangles
        // iterative refinement loop 
        // continues until there are no more encroached edges or bad triangles
        while (true) {
            // attempt to split an encroached segment (has highest priority wrt to bad triangles)
            if (split_first_encroached_segment_(segments, encroached_segments, bad_triangles, min_angle, max_area))
                continue;

            // otherwise, attempt to split the worst triangle in order of priority
            if (split_first_bad_triangle_(segments, encroached_segments, bad_triangles, min_angle, max_area)) {
                continue;
            }

            // terminate since no encroached edges or bad trinangles are there

            // terminate since no encroached edges or bad trinangles are there
            break;
        }

        // reassign IDs for cells and halfedges after refinement 
        
        int cont = 0;
        // reassign IDs for cells and halfedges after refinement 
        
        int cont = 0;
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it)
            it->set_id(cont++);

        cont = 0;
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it)
            it->set_id(cont++);
    }

    // function perfoming the flip alghoritm to tranform every non-Delaunay triangulation into a Delaunay one
    // not used for the actual costruction for the O(n^2) complexity
    void flip() {

        // create a list of halfedges to check whether they are locally delaunay or not (in this case flippable)
        std::unordered_set<halfedge_t*> halfedges_to_check;
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it,++it) {
            // exclude already the edges of the triangulation being automatically locally delaunay
            if(!it->is_segment()) {
                halfedges_to_check.insert(&(*it));
            }
        }
        // flip algorithm
        while (!halfedges_to_check.empty()) {
            halfedge_t* edge = *halfedges_to_check.begin();
            halfedges_to_check.erase(edge);
            halfedges_to_check.erase(edge->twin()); 
            cell_t* neighbor = edge->twin()->cell();

            // obtaine the 4 vertices of the quadrilateral formed by the two adjoining triangles 
            coords_t A = edge->node()->coords();
            coords_t B = edge->twin()->node()->coords();
            coords_t C = edge->prev()->node()->coords();
            coords_t D = edge->twin()->prev()->node()->coords();
            // test if D lies inside the circumcircle of triangle ABC or C lies inside the circumcircle of triangle ABD
            if (fdapde::internals::in_circle(A, B, C, D) || fdapde::internals::in_circle(A, D, B, C)) {
                
                // flip since edge is not locally delaunay
                halfedge_t* prev= edge->prev();
                halfedge_t* twin_prev = edge->twin()->prev();
                dcel_.remove_edge(edge);
                halfedge_t* new_edge = dcel_.insert_edge(prev, twin_prev);
                
                if (new_edge) {
                    // insert the new halfedges created by the flip into the list to check
                    if(!new_edge->prev()->is_segment())
                        halfedges_to_check.insert(new_edge->prev());
                    if(!new_edge->next()->is_segment())
                        halfedges_to_check.insert(new_edge->next());
                    if(!new_edge->twin()->prev()->is_segment())
                        halfedges_to_check.insert(new_edge->twin()->prev());
                    if(!new_edge->twin()->next()->is_segment())
                        halfedges_to_check.insert(new_edge->twin()->next());
                }
            }
        }
    }
    
    // function printing global mesh statistics in order to make comparisons with other meshers:
    // number of vertices, triangles, edge lengths, triangle quality metrics, and histograms
    void print_statistics() {
        std::cout << "\nStatistics:\n\n";

        std::cout << "\n  Mesh vertices: " << dcel_.n_nodes() << "\n";
        std::cout << "  Mesh triangles: " << dcel_.n_cells() << "\n";
        std::cout << "  Mesh edges: " << dcel_.n_halfedges() / 2 << "\n";

        // initialize extrema for each metric
        // initialize extrema for each metric
        double min_area = std::numeric_limits<double>::max();
        double max_area = 0.0;


        double min_edge = std::numeric_limits<double>::max();
        double max_edge = 0.0;


        double min_altitude = std::numeric_limits<double>::max();
        double max_aspect_ratio = 0.0;


        double min_angle = std::numeric_limits<double>::max();
        double max_angle = 0.0;

        // binning histograms for aspect ratio and angles
        // binning histograms for aspect ratio and angles
        std::map<std::string, int> aspect_bins = {
            {"1.1547 - 1.5", 0}, {"1.5 - 2", 0}, {"2 - 2.5", 0}, {"2.5 - 3", 0},
            {"3 - 4", 0}, {"4 - 6", 0}, {"6 - 10", 0}, {"10 - 15", 0},
            {"15 - 25", 0}, {"25 - 50", 0}, {"50 - 100", 0}, {"100 - 300", 0},
            {"300 - 1000", 0}, {"1000 - 10000", 0}, {"10000 - 100000", 0}, {"100000 -", 0}
        };

        std::map<std::string, int> angle_bins = {
            {"0 - 10", 0}, {"10 - 20", 0}, {"20 - 30", 0}, {"30 - 40", 0}, {"40 - 50", 0},
            {"50 - 60", 0}, {"60 - 70", 0}, {"70 - 80", 0}, {"80 - 90", 0},
            {"90 - 100", 0}, {"100 - 110", 0}, {"110 - 120", 0}, {"120 - 130", 0},
            {"130 - 140", 0}, {"140 - 150", 0}, {"150 - 160", 0}, {"160 - 170", 0}, {"170 - 180", 0}
        };

        // iterate through all triangles to compute stats
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            cell_t* t = &(*it);
        // iterate through all triangles to compute stats
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            cell_t* t = &(*it);
            coords_t A = t->halfedge()->prev()->node()->coords();
            coords_t B = t->halfedge()->node()->coords();
            coords_t C = t->halfedge()->next()->node()->coords();

            // compute area 
            double area = fdapde::internals::measure_2d_tri(A, B, C);
            // compute area 
            double area = fdapde::internals::measure_2d_tri(A, B, C);
            min_area = std::min(min_area, area);
            max_area = std::max(max_area, area);

            // compute squared edge lengths
            // compute squared edge lengths
            double ab2 = (B - A).squaredNorm();
            double bc2 = (C - B).squaredNorm();
            double ca2 = (A - C).squaredNorm();
            double longest2 = std::max({ab2, bc2, ca2});
            double longest = std::sqrt(longest2);
            min_edge = std::min(min_edge, std::sqrt(std::min({ab2, bc2, ca2})));
            max_edge = std::max(max_edge, longest);

            // compute triangle altitude (from longest edge)
            // compute triangle altitude (from longest edge)
            double triminaltitude2 = (2 * area) * (2 * area) / longest2;
            double altitude = std::sqrt(triminaltitude2);
            min_altitude = std::min(min_altitude, altitude);

            // aspect ratio = longest edge / shortest altitude
            // aspect ratio = longest edge / shortest altitude
            double aspect2 = longest2 / triminaltitude2;
            double aspect = std::sqrt(aspect2);
            max_aspect_ratio = std::max(max_aspect_ratio, aspect);

            // compute internal angles at all three vertices
            // compute internal angles at all three vertices
            double angleA = fdapde::internals::angle_between(C, A, B); // ∠CAB
            double angleB = fdapde::internals::angle_between(A, B, C); // ∠ABC
            double angleC = fdapde::internals::angle_between(B, C, A); // ∠BCA
            min_angle = std::min({min_angle, angleA, angleB, angleC});
            max_angle = std::max({max_angle, angleA, angleB, angleC});

            // fill histograms
            // fill histograms
            auto bin_angle = [&](double deg) -> std::string {
                int d = static_cast<int>(deg);
                if (d < 10) return "0 - 10";
                if (d >= 170) return "170 - 180";
                int lower = (d / 10) * 10;
                int upper = lower + 10;
                return std::to_string(lower) + " - " + std::to_string(upper);
            };
            angle_bins[bin_angle(angleA)]++;
            angle_bins[bin_angle(angleB)]++;
            angle_bins[bin_angle(angleC)]++;

            auto bin_aspect = [&](double r) -> std::string {
                if (r < 1.5) return "1.1547 - 1.5";
                if (r < 2) return "1.5 - 2";
                if (r < 2.5) return "2 - 2.5";
                if (r < 3) return "2.5 - 3";
                if (r < 4) return "3 - 4";
                if (r < 6) return "4 - 6";
                if (r < 10) return "6 - 10";
                if (r < 15) return "10 - 15";
                if (r < 25) return "15 - 25";
                if (r < 50) return "25 - 50";
                if (r < 100) return "50 - 100";
                if (r < 300) return "100 - 300";
                if (r < 1000) return "300 - 1000";
                if (r < 10000) return "1000 - 10000";
                if (r < 100000) return "10000 - 100000";
                return "100000 -";
            };
            aspect_bins[bin_aspect(aspect)]++;
        }

        // print scalar statistics
        // print scalar statistics
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "\n  Smallest area:    " << min_area << "   |  Largest area:          " << max_area;
        std::cout << "\n  Shortest edge:    " << min_edge << "   |  Longest edge:         " << max_edge;
        std::cout << "\n  Shortest altitude:" << min_altitude << "   |  Largest aspect ratio: " << max_aspect_ratio << "\n";

        // print aspect ratio histogram 
        // print aspect ratio histogram 
        std::cout << "\n  Triangle aspect ratio histogram:\n";
        int aspect_i = 0;
        for (const auto& [range, count] : aspect_bins) {
            std::cout << "  " << std::setw(17) << std::left << range << ":  " << std::setw(8) << count;
            if (++aspect_i % 2 == 0) std::cout << "\n";
            else std::cout << "  |  ";
        }

        // print angle statistics and histogram 
        // print angle statistics and histogram 
        std::cout << "\n\n  Smallest angle:   " << min_angle << "   |  Largest angle:        " << max_angle << "\n";
        std::cout << "\n  Angle histogram:\n";
        int i = 0;
        for (const auto& [range, count] : angle_bins) {
            std::cout << "  " << std::setw(17) << std::left << range + " degrees:" << std::setw(8) << count;
            if (++i % 2 == 0) std::cout << "\n";
            else std::cout << "  |  ";
        }

        std::cout << std::endl;
    }



   private:
    dcel_t dcel_;

    // function triangulating the domain, adding N random internal points, and ensuring the triangulation is Delaunay compliant
    void triangulate_(int N, const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& boundaries_entry, const std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>>& holes_entry ) {

        // computing the bounding box
        double min_x = boundaries_entry[0].col(0).minCoeff();
        double max_x = boundaries_entry[0].col(0).maxCoeff();
        double min_y = boundaries_entry[0].col(1).minCoeff();
        double max_y = boundaries_entry[0].col(1).maxCoeff();
    
        // creating the generator of causal numbers
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist_x(min_x, max_x);
        std::uniform_real_distribution<double> dist_y(min_y, max_y);

        // check if boundaries are counterclockwise sorted
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>> boundaries;
        for(const auto& bd: boundaries_entry){
            if (!internals::are_2d_counterclockwise_sorted(bd)) {
                Eigen::Matrix<double, Eigen::Dynamic, embed_dim> reversed_bd(bd.rows(), embed_dim);
                for (int i = 0; i < bd.rows(); ++i)
                    reversed_bd.row(i) = bd.row(bd.rows() - 1 - i);
                boundaries.push_back(reversed_bd);
            }
            else{
                boundaries.push_back(bd);
            }
        }
        std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>> holes;
        for(const auto& hole_vect : holes_entry){
            std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>> new_vector;
            for (const auto& hole : hole_vect) {
                if (internals::are_2d_counterclockwise_sorted(hole)) {
                    Eigen::Matrix<double, Eigen::Dynamic, embed_dim> reversed_hole(hole.rows(), embed_dim);
                    for (int i = 0; i < hole.rows(); ++i)
                        reversed_hole.row(i) = hole.row(hole.rows() - 1 - i);
                    new_vector.push_back(reversed_hole);
                } else {
                    new_vector.push_back(hole);
                }
            }
            holes.push_back(new_vector);
        }
        
        // create boundary matrices with no 3 collinear points (methods in polygon.h work if the vertices are only the points defining edge changes)
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>> boundary_vertices=boundaries; 
        if(boundaries.size()==1)
            boundary_vertices[0] = split_boundary_points_(boundaries[0]);
        else{ 
            for(int i=1; i< boundaries.size(); ++i){   
                boundary_vertices[i] = split_boundary_points_(boundaries[i]);
            }
            // external boundary needs to include the attachment nodes of the regions, even if they are collinear points, otherwise the initial make_polygon will not work correctly
            boundary_vertices[0] = split_boundary_points_with_attachments_(boundary_vertices);
        }
        std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>> holes_vertices(holes.size());
        for(int j=0; j< holes.size(); ++j){
            std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>> holes_vertices_i(holes[j].size());
            for(int i=0; i< holes[j].size(); ++i){
                holes_vertices_i[i] = split_boundary_points_(holes[j][i]);
            }
            holes_vertices[j] = holes_vertices_i;
        }
        
        // initialize the triangulation
        initialize_triangulation_(boundary_vertices, holes_vertices);
        
        // now add all the segments' collinear points back to the triangulation
        if(boundaries.size()==1) 
            complete_boundary_(boundaries[0], boundary_vertices[0]);
        else{ // completing internal boundaries implies adding back also the points of the external boundary
            for(int i=1; i< boundaries.size(); ++i)
                complete_boundary_(boundaries[i], boundary_vertices[i]);
        }
        for(int j=1; j< holes.size(); ++j){
            for(int i=0; i< holes[j].size(); ++i){
                complete_boundary_(holes[j][i], holes_vertices[j][i]);
            }
        }
        
        flip();  // ensures initial triangulation satisfies Delaunay property

        int n_nodes_boundaries = dcel_.n_nodes();
        int generated_points = 0;
        // generate N random points inside the bounding box, ensuring they are within the polygon defined by boundaries and holes
        while (generated_points < N) {
            coords_t u;
            u << dist_x(gen), dist_y(gen);

                if (!fdapde::internals::is_point_in_polygon(boundaries[0], holes, u)){
                    continue;
                }

                node_t* n = dcel_.insert_node(node_t(dcel_.n_nodes(), false, u));
                detect_conflicts_(n);
                ++generated_points;
        }
        
        // insert all internal points into the triangulation using the conflict graph
        auto it = dcel_.nodes_begin();
        std::advance(it, n_nodes_boundaries);  // skip boundary/holes nodes
        for (; it != dcel_.nodes_end(); ++it) {
            node_t* u = &(*it);
            insert_vertex_at_conflict_(u);
        }
        
        // reassign consecutive IDs to all cells and half-edges for consistency
        int cont = 0;
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it)
            it->set_id(cont++);
        cont = 0;
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it)
            it->set_id(cont++);
        
    }
    
    //overloaded triangulate if user wants to pass manually the internal points
    //the user must know the given internal points lie all inside the domain 
    void triangulate_(const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& internal,
        const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& boundaries_entry, const std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>>& holes_entry) {
        
        // check if boundaries are counterclockwise sorted
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>> boundaries;
        for(const auto& bd: boundaries_entry){
            if (!internals::are_2d_counterclockwise_sorted(bd)) {
                Eigen::Matrix<double, Eigen::Dynamic, embed_dim> reversed_bd(bd.rows(), embed_dim);
                for (int i = 0; i < bd.rows(); ++i)
                    reversed_bd.row(i) = bd.row(bd.rows() - 1 - i);
                boundaries.push_back(reversed_bd);
            }
            else{
                boundaries.push_back(bd);
            }
        }
        std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>> holes;
        for(const auto& hole_vect : holes_entry){
            std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>> new_vector;
            for (const auto& hole : hole_vect) {
                if (internals::are_2d_counterclockwise_sorted(hole)) {
                    Eigen::Matrix<double, Eigen::Dynamic, embed_dim> reversed_hole(hole.rows(), embed_dim);
                    for (int i = 0; i < hole.rows(); ++i)
                        reversed_hole.row(i) = hole.row(hole.rows() - 1 - i);
                    new_vector.push_back(reversed_hole);
                } else {
                    new_vector.push_back(hole);
                }
            }
            holes.push_back(new_vector);
        }
        
        // create boundary matrices with no 3 collinear points (methods in polygon.h work if the vertices are only the points defining edge changes)
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>> boundary_vertices=boundaries; 
        if(boundaries.size()==1)
            boundary_vertices[0] = split_boundary_points_(boundaries[0]);
        else{ 
            for(int i=1; i< boundaries.size(); ++i){   
                boundary_vertices[i] = split_boundary_points_(boundaries[i]);
            }
            // external boundary needs to include the attachment nodes of the regions, even if they are collinear points, otherwise the initial make_polygon will not work correctly
            boundary_vertices[0] = split_boundary_points_with_attachments_(boundary_vertices);
        }
        std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>> holes_vertices(holes.size());
        for(int j=0; j< holes.size(); ++j){
            std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>> holes_vertices_i(holes[j].size());
            for(int i=0; i< holes[j].size(); ++i){
                holes_vertices_i[i] = split_boundary_points_(holes[j][i]);
            }
            holes_vertices[j] = holes_vertices_i;
        }
        
        // initialize the triangulation
        initialize_triangulation_(boundary_vertices, holes_vertices);
        
        // now add all the segments' collinear points back to the triangulation
        if(boundaries.size()==1) 
            complete_boundary_(boundaries[0], boundary_vertices[0]);
        else{ // completing internal boundaries implies adding back also the points of the external boundary
            for(int i=1; i< boundaries.size(); ++i)
                complete_boundary_(boundaries[i], boundary_vertices[i]);
        }
        for(int j=1; j< holes.size(); ++j){
            for(int i=0; i< holes[j].size(); ++i){
                complete_boundary_(holes[j][i], holes_vertices[j][i]);
            }
        }
        
        flip();  // ensures initial triangulation satisfies Delaunay property

        int n_nodes_boundaries = dcel_.n_nodes();

        //inserting the internal points in the triangulation
        for (int i = 0; i < internal.rows(); ++i) {
            node_t* n = dcel_.insert_node(node_t(dcel_.n_nodes(), false, internal.row(i).eval())); 
            detect_conflicts_(n);
        }

        // insert all internal points into the triangulation using the conflict graph
        auto it = dcel_.nodes_begin();
        std::advance(it, n_nodes_boundaries);  // skip boundary/holes nodes
        for (; it != dcel_.nodes_end(); ++it) {
            node_t* u = &(*it);
            insert_vertex_at_conflict_(u); 
        }   

        // reassign consecutive IDs to all cells and half-edges for consistency
        int cont = 0;
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it)
            it->set_id(cont++);
        cont = 0;
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it)
            it->set_id(cont++);
    }


//------------------------- methods to initialize the triangulation -------------------------------------------

    //function performing the first raw triangulation of the domain using the polygon.h class
    void initialize_triangulation_(const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& boundaries, std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>>& holes) {
        
        dcel_= dcel_t::make_polygon(boundaries[0], holes[0]);
       
        if(boundaries.size() == 1 ){
            polygon_t polygon(boundaries[0], holes[0]);
            auto triangulation = polygon.triangulation();
            dcel_.from_triangulation(triangulation, holes[0]);
        }
        else {  // regions
            cell_t* c_new_edges = &(*std::prev(dcel_.cells_end())); // cell to assign to the new emplaced halfedges
            for(int i=1; i< boundaries.size(); ++i){
                // creation of connection for internal regions
                for(int j=0; j< boundaries[i].rows(); ++j){
                        coords_t co= boundaries[i].row(j);  
                        if(dcel_.find_node(co)) continue; // node already added in dcel_
                        // create a new node and halfedge for the segment point (it's not on the boundary)
                        node_t* n1 = dcel_.insert_node(node_t(dcel_.n_nodes(), false, co)); 
                        halfedge_t* h1 = dcel_.emplace_halfedge(n1, true);  //is_segment=true
                        n1->set_halfedge(h1);
                        h1->set_cell(c_new_edges);
                } 
                cell_t* c_holes= nullptr;
                for (int j = 0; j < boundaries[i].rows(); ++j) {
                        coords_t co1= boundaries[i].row(j);  
                        coords_t co2= boundaries[i].row( (j+1) % boundaries[i].rows() ); 
                        node_t* n1= dcel_.find_node(co1);
                        node_t* n2= dcel_.find_node(co2);
                        halfedge_t* h_between = dcel_.find_halfedge_between(co1, co2);
                        if (!h_between) {
                            insert_collinear_chain_(n1, n2);
                            h_between = dcel_.find_halfedge_between(co1, co2);
                            if (h_between && !h_between->on_boundary()) {
                                c_new_edges = h_between->twin()->cell();
                            }
                        }
                        if (h_between) {
                            c_holes = h_between->cell();
                        }       
                } 
                if(holes.size() == i)
                    holes.push_back({});
                polygon_t polygon(boundaries[i], holes[i]);
                auto triangulation = polygon.triangulation();
                // update holes[i] edges to the cell of boundary i 
                for (int j = 0; j < holes[i].size(); ++j) {
                    coords_t co = holes[i][j].row(0);  
                    if (!dcel_.find_node(co)) continue;
                    node_t* n1 = dcel_.find_node(co);
                    halfedge_t* h_hole= n1->halfedge();   // make_polygon creates holes' nodes s.t. its own halfedge is defined 
                    h_hole->set_cell(c_holes);
                    halfedge_t* next= h_hole->next();
                    do{
                        next->set_cell(c_holes);
                        next = next->next();
                    }while(next!= h_hole);
                }
                dcel_.from_triangulation(triangulation, holes[i]);
                fix_collinear_edge_crossings_();
            }
        }
        // connect any nodes that might appear in only some regions and not in others (vertex in a region and collinear in another, e.g. stair)
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            cell_t* c = &(*it);
            halfedge_t* h = c->halfedge();
            halfedge_t* h_nn = h->next()->next();
            while(true){
                if(!fdapde::internals::collinear(h->node()->coords(), h->next()->node()->coords(), h_nn->node()->coords())){
                    dcel_.insert_edge(h, h_nn);
                    break;
                }
                h=h->next();
                h_nn= h_nn->next();
            }
        } 
    }

    //  function detecting and removing edges that improperly cross over collinear intermediate nodes (usually between 2 different regions' boundaries) 
    void fix_collinear_edge_crossings_(){
        std::unordered_set<halfedge_t*> edges_to_remove;

        // scan all cells edges, and identify those that span across other collinear nodes without passing through them
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            cell_t* c = &(*it);
            halfedge_t* h_start = c->halfedge();
            if (!h_start) continue;

            // loop through the edges of the current cell
            halfedge_t* h = h_start;
            do {
                node_t* n1 = h->node();
                node_t* n2 = h->twin()->node();
                if (!n1 || !n2) continue;

                coords_t A = n1->coords();
                coords_t B = n2->coords();
                coords_t AB = B - A;
                double ab2 = AB.squaredNorm();
                if (ab2 < 1e-12) continue;

                // check if any third node lies strictly between A and B (collinear and internal)
                for (auto nit = dcel_.nodes_begin(); nit != dcel_.nodes_end(); ++nit) {
                    node_t* P = &(*nit);
                    if (P == n1 || P == n2) continue;  // skip endpoints

                    coords_t p = P->coords();
                    if (!fdapde::internals::collinear(A, p, B)) continue;

                    coords_t AP = p - A;
                    double t = AB.dot(AP) / ab2;
                    if (t > 1e-6 && t < 1.0 - 1e-6) {
                        // P lies strictly between A and B, mark the edge for removal
                        if (!edges_to_remove.count(h) && !edges_to_remove.count(h->twin())) {
                            edges_to_remove.insert(h);
                        }
                        break;
                    }
                }

                h = h->next();
            } while (h && h != h_start);
        }

        // remove problematic edges (there are already edges underneath)
        for (halfedge_t* h : edges_to_remove) 
            dcel_.remove_edge(h);
    }

    // function inserting a chain of segments between two nodes A and B by detecting and connecting all intermediate nodes that lie collinearly along the segment AB
    // used to create the external boundary and internal regions' boundaries of the initial triangulation
    void insert_collinear_chain_(node_t* A, node_t* B) {
        coords_t pA = A->coords();
        coords_t pB = B->coords();
        coords_t AB = pB - pA;
        double ab_norm2 = AB.squaredNorm();

        // vector of intermediate nodes that lie collinearly between A and B, where each element is a pair (position, node)
        std::vector<std::pair<double, node_t*>> intermediate;

        //we need to iterate over all nodes in the DCEL since nodes are not necessarily ordered
        for (auto it = dcel_.nodes_begin(); it != dcel_.nodes_end(); ++it) {
            node_t* P = &(*it);
            if (P == A || P == B) continue;
            coords_t p = P->coords();
            if (!fdapde::internals::collinear(pA, p, pB)) continue;
            coords_t AP = p - pA;
            // calculate relative position of P along the segment AB
            double t = AB.dot(AP) / ab_norm2;  
            // only consider points that are strictly between A and B
            if (t > 1e-6 && t < 1.0 - 1e-6) 
                intermediate.emplace_back(t, P);
        }

        // order intermediate based on t (position of P on AB)
        std::sort(intermediate.begin(), intermediate.end(),[](const std::pair<double, node_t*>& a, const std::pair<double, node_t*>& b) {
            return a.first < b.first;
        });

        // construct chain A → P1 → ... → Pk → B
        node_t* prev = A;
        for (auto& [_, curr] : intermediate) {
            halfedge_t* h1 = prev->halfedge();  // since prev is a boundary/segment/hole node, make_polygon assigns the right halfedge
            halfedge_t* h2 = curr->halfedge();  // since curr is a boundary/segment/hole node, make_polygon assigns the right halfedge
            if (h1 && h2) {
                halfedge_t* h_new = dcel_.insert_edge(h1, h2);
                h_new->set_segment(true);
                h_new->twin()->set_segment(true);
            }
            prev = curr;
        }
        // connect last point to B
        halfedge_t* h1 = prev->halfedge();
        halfedge_t* h2 = dcel_.find_halfedge(B, h1->cell(), true);  
        if (h1 && h2) {
            halfedge_t* h_new = dcel_.insert_edge(h1, h2);
            h_new->set_segment(true);
            h_new->twin()->set_segment(true);
        }
    }


    // function dividing boundary points into vertices (points where the edge changes) and collinear points, returning only the vertices
    Eigen::Matrix<double, Eigen::Dynamic, 2>  split_boundary_points_(const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary) {
        
        std::list<Eigen::Matrix<double, 1, 2>> vertex_points_list;
        int n = boundary.rows();

        // iterate through the boundary points
        for (int i = 0; i < n; ++i) {
            const auto& point = boundary.row(i);
            const auto& point_prev = boundary.row((i-1+n)%n);
            const auto& point_next = boundary.row((i+1)%n);
            if (!fdapde::internals::collinear(point_prev, point, point_next)) 
                vertex_points_list.push_back(point);
        }

        Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_points(vertex_points_list.size(), 2);
        int i=0;
        for (const auto & point : vertex_points_list) {
            vertex_points.row(i++) = point;
        }
        
        return vertex_points;
    }

    // function dividing boundary points into vertices (points where the edge changes) and collinear points, returning only the vertices
    // function dividing boundary points into vertices (points where the edge changes) and collinear points, returning only the vertices
    // this function also includes the points that are attachments to the boundary for the internal regions, even if they are collinear
    Eigen::Matrix<double, Eigen::Dynamic, 2> split_boundary_points_with_attachments_(const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& boundary_vertices) 
    {
        std::list<Eigen::Matrix<double, 1, 2>> vertex_points_list;

        // lambda function to compare two 2D row vectors, since std::set requires a strict weak ordering
        auto row_matrix_less = [](const Eigen::Matrix<double, 1, 2>& a,const Eigen::Matrix<double, 1, 2>& b) -> bool 
        {
            if (std::abs(a(0) - b(0)) > 1e-10) return a(0) < b(0);
            return a(1) < b(1) - 1e-10;
        };

        // build the set storing points defining regions, using row_matrix_less as comparison function
        // not using std::unordered_set because boundary_vertices doesn't have millions of points generally, 
        // and because comparing double values with numerical tolerance is incompatible with hash-based lookup, which requires strict equality
        std::set<Eigen::Matrix<double, 1, 2>, decltype(row_matrix_less)> regions_points(row_matrix_less);
        for (int j=1; j< boundary_vertices.size(); ++j)  // j=0 is the external boundary, which is not considered here
            for (int i = 0; i < boundary_vertices[j].rows(); ++i) 
                regions_points.insert(boundary_vertices[j].row(i));

        int n = boundary_vertices[0].rows();  // number of points in the external boundary
        for (int i = 0; i < n; ++i) {
            const auto& point = boundary_vertices[0].row(i);
            const auto& point_prev = boundary_vertices[0].row((i - 1 + n) % n);
            const auto& point_next = boundary_vertices[0].row((i + 1) % n);
            // assuming in regions_points we have only points of regions that are not collinear, we can check if the point is a vertex or an attachment
            bool is_vertex = !fdapde::internals::collinear(point_prev, point, point_next);
            bool is_attachment = regions_points.find(point) != regions_points.end();
            if (is_vertex || is_attachment)
                vertex_points_list.push_back(point);
        }

        Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_points(vertex_points_list.size(), 2);
        int i = 0;
        for (const auto& point : vertex_points_list) {
            vertex_points.row(i++) = point;
        }
        return vertex_points;
    }

    // function completing the boundary by inserting collinear points between existing vertices, ensuring the boundary is complete
    void complete_boundary_(const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary, const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary_vertices ) {
        //if the boundary is already complete we do not need to do anything
        if(boundary.rows()==boundary_vertices.rows()) return; 

        auto row_matrix_less = [](const Eigen::Matrix<double, 1, embed_dim>& a, const Eigen::Matrix<double, 1, embed_dim>& b) -> bool {
            if ((a - b).norm() < 1e-10) return false;
            if (std::abs(a(0) - b(0)) > 1e-10) return a(0) < b(0);
            return a(1) < b(1) - 1e-10;
        };
        // set of vertices
        std::set<Eigen::Matrix<double, 1, embed_dim>, decltype(row_matrix_less)> vertices_set(row_matrix_less);
        for (int i = 0; i < boundary_vertices.rows(); ++i)
            vertices_set.insert(boundary_vertices.row(i));
            
        int j = 0;
        int k = 0;
        halfedge_t* e = nullptr; // edge to find between two existing boundary points
        coords_t last_coords = boundary_vertices.row((boundary_vertices.rows() - 1)); 

        for (int i = 0; i < boundary.rows(); i = j) {
            if (vertices_set.find(boundary.row(i)) == vertices_set.end()) { // faster than scanning all DCEL nodes
                j = i;

                // p = point to insert
                coords_t p = boundary.row(j); 
                // if p is already in dcel_, move to the next point
                if (dcel_.find_node(p)) {
                    last_coords = p;
                    ++j;
                    continue;  
                }

                // determine the two adjacent boundary points surrounding p 
                coords_t n1 = last_coords;
                coords_t n2 = boundary_vertices.row(k % boundary_vertices.rows());  

                // find and remove edge between n1 and n2, to add p
                if (!e) e = dcel_.find_halfedge_between(n1, n2);
                if (!e) e = dcel_.find_halfedge_between(n1, boundary.row(0)); 
                if (!e) continue;

                node_t* m = dcel_.insert_node(node_t(dcel_.n_nodes(), e->on_boundary(), p));
                halfedge_t* prev = e->prev();
                halfedge_t* next = e->next();
                halfedge_t* twin_prev = e->twin()->prev();

                // insert new edges
                halfedge_t* h1 = dcel_.emplace_halfedge(m, true);
                h1->set_cell(next->cell());
                dcel_.insert_edge(next, h1);
                h1->twin()->set_segment(true);

                halfedge_t* h2 = dcel_.insert_edge(prev->next(), h1);
                h2->set_segment(true);
                h2->twin()->set_segment(true);

                dcel_.insert_edge(prev, h1);
                dcel_.remove_edge(e);

                if (!h1->on_boundary()) {  // an internal segment
                    dcel_.insert_edge(twin_prev, h2->twin());
                }

                last_coords = p; 
                ++j;
                e=h1;

            } else { // move to the next boundary points to insert
                // update last_coords (new n1)
                last_coords = boundary.row(i);  
                ++j;
                ++k;
                e = nullptr;  // reset edge to find next time
            }
        }
    }

//----------------------- end of methods to initialize the triangulation -------------------------------------

//----------------------- methods to perform the Costrained Delaunay Triangulation (CDT) ---------------------

    // function inserting a vertex and handling conflicts
    void insert_vertex_at_conflict_(node_t* u) {
        // retrieve the triangle in conflict with u 
        cell_t* t = u->conflict(); 
        // vector storing the halfedge refering to the cells that need to be deleted(D) and created(C)
        std::list<halfedge_t*> D;
        std::list<halfedge_t*> C;

        // remove the edge if the point falls on it 
        // remove the edge if the point falls on it 
        const coords_t& t1 = t->halfedge()->node()->coords();
        const coords_t& t2 = t->halfedge()->next()->node()->coords();
        const coords_t& t3 = t->halfedge()->prev()->node()->coords();
        
        bool found_on_edge = false;

        //if u lies on edge t1-t2, cavity is expanded on both sides of t1-t2
        if (fdapde::internals::contains(u->coords(), t1, t2)) {
            D.push_back(t->halfedge());
                // work on the remaining cavity edges to correctly activate the algorithm 
                mark_cavity_(u, t->halfedge()->next(), D, C);
                mark_cavity_(u, t->halfedge()->prev(), D, C);
                mark_cavity_(u, t->halfedge()->twin()->next(), D, C);
                mark_cavity_(u, t->halfedge()->twin()->prev(), D, C);
            found_on_edge = true;
        }
    
        else if (fdapde::internals::contains(u->coords(), t2, t3)) {
            D.push_back(t->halfedge()->next());
                mark_cavity_(u, t->halfedge(), D, C);
                mark_cavity_(u, t->halfedge()->prev(), D, C);
                mark_cavity_(u, t->halfedge()->next()->twin()->next(), D, C);
                mark_cavity_(u, t->halfedge()->next()->twin()->prev(), D, C);
            found_on_edge = true;
        }
    
        else if (fdapde::internals::contains(u->coords(), t3, t1)) {
            D.push_back(t->halfedge()->prev());
                mark_cavity_(u, t->halfedge()->next(), D, C);
                mark_cavity_(u, t->halfedge(), D, C);
                mark_cavity_(u, t->halfedge()->prev()->twin()->next(), D, C);
                mark_cavity_(u, t->halfedge()->prev()->twin()->prev(), D, C);
            found_on_edge = true;
        }
        // standard case: try to expand the cavity from the edges of the triangle 
        if (!found_on_edge) {
            mark_cavity_(u, t->halfedge(), D, C);
            mark_cavity_(u, t->halfedge()->next(), D, C);
            mark_cavity_(u, t->halfedge()->prev(), D, C);
        }
    
        // invalidate the conflicts node->cell for the points of the cavity
        // invalidate the conflicts node->cell for the points of the cavity
        std::unordered_set<node_t*> invalidated_nodes;
        for (halfedge_t* h : D) {
            cell_t* current_cell = h->cell();
            if (current_cell) {
                for (node_t* point : current_cell->conflicting_points()) {
                    if (point != u) {
                        point->set_conflict(nullptr);  
                        invalidated_nodes.insert(point);
                    }
                }
                current_cell->clear_conflicts();
            }
            // do the same for halfedge twin's cell
            cell_t* twin_cell = h->twin()->cell();
            if (twin_cell) {
                for (node_t* point : twin_cell->conflicting_points()) {
                    if (point != u) {
                        point->set_conflict(nullptr);  
                        invalidated_nodes.insert(point);
                    }
                }
                twin_cell->clear_conflicts();
            }
        }
        // if all the new traingles are Delaunay: do not expand the cavity 
        if(D.empty()){
            cell_t* current_cell = u->conflict();
            if (current_cell) {
                for (node_t* point : current_cell->conflicting_points()) {
                    if (point != u) {
                        point->set_conflict(nullptr);  
                        invalidated_nodes.insert(point);
                    }
                }
                current_cell->clear_conflicts();
            }
        }
        u->remove_conflict();
        
        // remove cavity contour edges
        for (halfedge_t* h : D) { 
            dcel_.remove_edge(h);
        }
        // create new cells 
        for (halfedge_t* h : C) { 
            add_triangle_(h, std::vector<node_t*> {u});
        }
        
        // reassign the conflicts to the new cells  
        for (node_t* y : invalidated_nodes) {
            detect_conflicts_(y, C);
        }
    } 

    // function for conflict detection in conflict graph algorithm 
    void detect_conflicts_(node_t* n, const std::list<halfedge_t*>& cells_to_check = {}) {
        bool found = false;
        // initialization case: scan all the cells
        if (cells_to_check.empty()) {  
            for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
                 
                cell_t* t = &(*it);
                const coords_t& t1 = t->halfedge()->prev()->node()->coords();
                const coords_t& t2 = t->halfedge()->node()->coords();
                const coords_t& t3 = t->halfedge()->next()->node()->coords();

                bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(t1, t2, t3);

                // verifying if the point is inside the triangle
                if (!found) {
                    bool inside_triangle = ccw ? fdapde::internals::point_in_2d_tri(n->coords(), t1, t2, t3)
                                            : fdapde::internals::point_in_2d_tri(n->coords(), t3, t2, t1);
                    if (inside_triangle) {
                        // the conflict is from point to triangle and viceversa
                        // the conflict is from point to triangle and viceversa
                        n->set_conflict(t); 
                        t->add_conflict(n);
                        found = true;
                    }
                }
            }
        } else {  // standard case: scan the cavity
        } else {  // standard case: scan the cavity
            for (halfedge_t* h : cells_to_check) {
                cell_t* t = h->cell();
                if (!t) continue;

                const coords_t& t1 = t->halfedge()->prev()->node()->coords();
                const coords_t& t2 = t->halfedge()->node()->coords();
                const coords_t& t3 = t->halfedge()->next()->node()->coords();

                bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(t1, t2, t3);

                if (!found) {
                    bool inside_triangle = ccw ? fdapde::internals::point_in_2d_tri(n->coords(), t1, t2, t3)
                                            : fdapde::internals::point_in_2d_tri(n->coords(), t3, t2, t1);
                    if (inside_triangle) {
                        n->set_conflict(t); 
                        t->add_conflict(n);
                        found = true;
                    }
                }
            }
        }
    }

    // function marking the cavity during insertion
    void mark_cavity_(node_t* u, halfedge_t* h, std::list<halfedge_t*>& D, std::list<halfedge_t*>& C) {
        // if boundary is reached then stop cavity expansion (create triangle with edge h)
        if(h->is_segment()){
            C.push_back(h);  
            return;
        }

        node_t* x = dcel_.adjacent(h);
        if (!x) {
            return;
        } 
        bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(u->coords(), h->node()->coords(), h->twin()->node()->coords());
        // test of circumcircle   
        // test of circumcircle   
        bool inside;
        if (ccw) {
            inside = fdapde::internals::in_circle(u->coords(), h->node()->coords(), h->twin()->node()->coords(), x->coords());
        } else {
            inside = fdapde::internals::in_circle(u->coords(), h->twin()->node()->coords(), h->node()->coords(), x->coords());
        }
        if (inside) {    // test fails so append vw to D and expand the cavity 
        if (inside) {    // test fails so append vw to D and expand the cavity 
            D.push_back(h);
            mark_cavity_(u, h->twin()->prev(), D, C);
            mark_cavity_(u, h->twin()->next(), D, C);
            return;
        } else {
            C.push_back(h);  // new triangle with edge h and new inserted vertex will be created
            return;
        }
    }

    halfedge_t* add_triangle_(halfedge_t* v,const std::vector<node_t*>& node){
        return dcel_.add_polygon(v ,node);
    }


//--------------------- end of methods to perform the Constrained Delaunay Triangulation (CDT) ----------------

//--------------------- support methods to refine the CDT -----------------------------------------------------

    // function checking if a given edge is encroached
    // the edge is considered encroached if the angle opposite to it is >= 90 degrees
    bool check_encroachment_(halfedge_t* e) const{
        coords_t A = e->node()->coords();
        coords_t B = e->twin()->node()->coords();
        coords_t C = e->prev()->node()->coords();
        coords_t C = e->prev()->node()->coords();
        double angle = fdapde::internals::angle_between(B, C, A);
        return angle >= 90.0;
    }

    // overloaded version testing encroachment using a custom point (e.g., circumcenter for refinement algorithm) instead of C
    bool check_encroachment_(halfedge_t* e, const coords_t& c) const{
        coords_t A = e->node()->coords();
        coords_t B = e->twin()->node()->coords();
        coords_t B = e->twin()->node()->coords();
        double angle = fdapde::internals::angle_between(B, c, A);
        return angle >= 90.0;
    }


    // function that checks if edge defined by halfedge h is seditious (definition from Delaunay Mesh Generation by Shewchuk)
    // if it is, it is the triangle's shortest edge since its opposing angle is < 60 degrees and the triangle is isosceles
    bool is_edge_seditious_(halfedge_t* h) const{
        if (h->is_segment()) return false;  // segments can't be seditious
        
        if (!(h->prev()->is_segment() && h->next()->is_segment())) return false;  
        coords_t a = h->node()->coords();
        coords_t b = h->next()->node()->coords();
        coords_t c = h->prev()->node()->coords();
        // angle in c^ is angle of interest
        if (fdapde::internals::angle_between(b,c,a) >= 60) return false; // angle is not too small
        
        // 2 edges of h's cell need to have same length and to be midpoints of another segment 
        if (fdapde::internals::segment_length(c,a) != fdapde::internals::segment_length(c,b) )  return false;
        coords_t d = h->prev()->twin()->prev()->node()->coords();  
        coords_t e = h->next()->twin()->next()->next()->node()->coords();
        if ( !( fdapde::internals::collinear(c, a, d) && 
                fdapde::internals::collinear(c, b, e) ) )
                return false;
        
        std::cout << "SEDITIOUS EDGE: " << h->id() << std::endl;
        return true;
    }   

    // function splitting the first encroached segment that is not adjacent to a seditious edge
    bool split_first_encroached_segment_(std::unordered_set<halfedge_t*>& segments,
                                        std::unordered_set<halfedge_t*>& encroached_segments,
                                        std::multimap<double, cell_t*>& bad_triangles,
                                        double min_angle, double max_area) {
        auto it = encroached_segments.begin();
        while (it != encroached_segments.end()) {
            halfedge_t* e = *it;

            if (!e || !e->twin()) {
                it = encroached_segments.erase(it);
                continue;
            }
            ++it;
            encroached_segments.erase(e);
            encroached_segments.erase(e->twin());

            // check if adjacent edges to e are seditious; no split is performed if one of them is
            if (is_edge_seditious_(e->next()) || is_edge_seditious_(e->prev())) {
                continue;
            }

            split_segment_(e, segments, encroached_segments, bad_triangles, min_angle, max_area);
            return true;
        }
        return false;
    }




    // utility function removing all entries of a given cell from a multimap
    void remove_from_multimap_(std::multimap<double, cell_t*>& mmap, cell_t* target) const{
        for (auto it = mmap.begin(); it != mmap.end(); ) {
            if (it->second == target) {
                it = mmap.erase(it);
            } else {
                ++it;
            }
        }
    }

    // function that splits an encroached segment into two by inserting a new node (either the middle point or a point projected onto the segment)
    void split_segment_(halfedge_t* e, std::unordered_set<halfedge_t*>& segments,
        std::unordered_set<halfedge_t*>& encroached_segments, std::multimap<double, cell_t*>& bad_triangles,
        double min_angle, double max_area) {

        // remove e and its twin from the set of segments, since removing e implies removing its twin
        segments.erase(e);
        segments.erase(e->twin());
        remove_from_multimap_(bad_triangles, e->cell());
        remove_from_multimap_(bad_triangles, e->twin()->cell());

        node_t* a = e->node();
        node_t* b = e->twin()->node();
        node_t* c = e->prev()->node();

        // coordinates of the point to insert to split e
        coords_t split_pt; 

        // if ab^c is <= 45° and ab is not too close in length to bc, create a circular crown to ensure at next sweep e is not encroached anymore
        if (e->next()->is_segment() &&
            (fdapde::internals::segment_length(a->coords(), b->coords()) - fdapde::internals::segment_length(b->coords(), c->coords()) ) / fdapde::internals::segment_length(a->coords(), b->coords()) > 0.1 &&
            fdapde::internals::angle_between(a->coords(), b->coords(), c->coords()) <=45) {
            // project c onto the line ab s.t. b_split = bc
            coords_t split_pt_ref = c->coords();
            double r = (split_pt_ref - b->coords()).norm();
            double r = (split_pt_ref - b->coords()).norm();
            coords_t ab = a->coords() - b->coords();
            double L = ab.norm();
            double t = r / L;
            split_pt = b->coords() + t * ab;
        } else {  // split e in the middle
            split_pt = 0.5 * (a->coords() + b->coords());
        }

        // insert the split point in the dcel_
        node_t* m = dcel_.insert_node(node_t(dcel_.n_nodes(), e->on_boundary(), split_pt));

        halfedge_t* prev = e->prev(); //ca
        halfedge_t* twin_prev = e->twin()->prev();   
        halfedge_t* next = e->next();  //bc

        // create new connections between am, mb, bc before removing e
        halfedge_t* h1 = dcel_.emplace_halfedge(m, true);
        h1->set_cell(next->cell());
        dcel_.insert_edge(next, h1); //mb
        h1->twin()->set_segment(true);
        halfedge_t* h2 = dcel_.insert_edge(prev->next(), h1);  //am
        h2->set_segment(true);
        h2->twin()->set_segment(true);
        dcel_.insert_edge(prev, h1);  //mc
        dcel_.remove_edge(e);

        // check whether the 2 new triangles are badly-shaped
        auto already_present = [&](cell_t* c) {return std::find_if(bad_triangles.begin(), bad_triangles.end(),[c](const auto& entry) { return entry.second == c; }) != bad_triangles.end();};
        // check whether the 2 new triangles are badly-shaped
        double p1 = is_bad_triangle_(h1->cell(), min_angle, max_area);
        if (p1 >= 0.0 && !already_present(h1->cell()) ) 
            bad_triangles.insert({p1,h1->cell()});
        double p2 = is_bad_triangle_(h2->cell(), min_angle, max_area);
        if (p2 >= 0.0 && !already_present(h2->cell()) ) 
            bad_triangles.insert({p2,h2->cell()});
        
        // if h1 is not a boundary edge, insert an edge that connects m with the opposite node, in order to ensure the mesh is conforming
        if(!h1->on_boundary()) {
            dcel_.insert_edge(twin_prev, h2->twin());
            // check whether the two new triangles are badly-shaped
            double p3 = is_bad_triangle_(h1->twin()->cell(), min_angle, max_area);
            if (p3 >= 0.0 && !already_present(h1->twin()->cell()) ) 
                bad_triangles.insert({p3, h1->twin()->cell()});
            double p4 = is_bad_triangle_(h2->twin()->cell(), min_angle, max_area);
            if (p4 >= 0.0 && !already_present(h2->twin()->cell()) ) 
                bad_triangles.insert({p4, h2->twin()->cell()});
        }

        if (fdapde::internals::in_circle(h1->node()->coords(), next->node()->coords(), prev->node()->coords(), next->twin()->prev()->node()->coords()) || 
            fdapde::internals::in_circle(next->twin()->prev()->node()->coords(), h1->prev()->node()->coords(), next->node()->coords(), h1->node()->coords()) || 
            fdapde::internals::in_circle(h1->node()->coords(), prev->node()->coords(), h2->node()->coords(), prev->twin()->prev()->node()->coords()) || 
            fdapde::internals::in_circle(prev->twin()->prev()->node()->coords(), h2->node()->coords(), prev->node()->coords(), h1->node()->coords()) ) {
            // perform local flips if necessary to maintain Delaunay property
            flip_refinement_(encroached_segments, bad_triangles, min_angle, max_area);
        }
        else if(!h1->on_boundary()){
            halfedge_t* h3= h1->twin();
            halfedge_t* h4= h2->twin();
            if (fdapde::internals::in_circle(h4->node()->coords(), h3->prev()->node()->coords(), h3->node()->coords(), h3->prev()->twin()->prev()->node()->coords()) ||
                fdapde::internals::in_circle(h3->prev()->twin()->prev()->node()->coords(), h3->node()->coords(), h3->prev()->node()->coords(), h4->node()->coords()) ||
                fdapde::internals::in_circle(h4->node()->coords(), h2->node()->coords(), h4->prev()->node()->coords(), h4->next()->twin()->prev()->node()->coords()) ||
                fdapde::internals::in_circle(h4->next()->twin()->prev()->node()->coords(), h4->prev()->node()->coords(), h2->node()->coords(), h4->node()->coords())){
                    flip_refinement_(encroached_segments, bad_triangles, min_angle, max_area);
                }
        }
        // perform local flips if necessary to maintain Delaunay property
        //flip_refinement_(encroached_segments, bad_triangles, min_angle, max_area);

        // recheck the two new segments for possible encroachment
        segments.insert(h1);
        segments.insert(h2);
        // if they are not boundary segments, insert also the twins (internal regions case)
        if (!h1->on_boundary())  {
            segments.insert(h1->twin());
        }
        if (!h2->on_boundary()) {
            segments.insert(h2->twin());
        }
        
        // check in segments if new encorachments are present
        for(auto it = segments.begin(); it != segments.end(); ++it) {
            halfedge_t* h = *it;
            if (check_encroachment_(h)) {
                encroached_segments.insert(h);
            }
        }
    }

    // function locating the triangle containing a given point P, starting from an initial triangle and expanding locally.
    // designed for refinement routines where the point P is typically the circumcenter of a triangle t, making it natural to start the search from t.
    cell_t* find_triangle_local_(const coords_t& P, cell_t* start) const{
        std::unordered_set<cell_t*> visited; // track visited triangles to avoid revisiting them
        std::queue<cell_t*> queue;           // queue storing triangles to be explored during the local search

        queue.push(start);
        visited.insert(start);

        while (!queue.empty()) {
            cell_t* current = queue.front();
            queue.pop();

            // retrieve the coordinates of the current triangle's vertices
            // retrieve the coordinates of the current triangle's vertices
            const coords_t& A = current->halfedge()->node()->coords();
            const coords_t& B = current->halfedge()->next()->node()->coords();
            const coords_t& C = current->halfedge()->prev()->node()->coords();

            // check if point P lies inside the current triangle
            // check if point P lies inside the current triangle
            if (fdapde::internals::point_in_2d_tri(P, A, B, C)) {
                return current;  // found the containing triangle
                return current;  // found the containing triangle
            }

            // explore neighboring triangles across the three edges
            // explore neighboring triangles across the three edges
            for (int i = 0; i < 3; ++i) {
                halfedge_t* e = current->halfedge();
                for (int j = 0; j < i; ++j) e = e->next();  

                cell_t* neighbor = e->twin()->cell();
                if (neighbor && visited.count(neighbor) == 0) {
                for (int j = 0; j < i; ++j) e = e->next();  

                cell_t* neighbor = e->twin()->cell();
                if (neighbor && visited.count(neighbor) == 0) {
                    queue.push(neighbor);
                    visited.insert(neighbor);
                }
            }
        }

        return nullptr;  // fallback if no containing triangle is found (should not happen in practice)
    }

    // function computing a priority value in the range [0, 4095] based on the squared length of the shortest edge
    // used to order bad triangles using the key of the associated multimap, ensuring that triangles with small angles have the priority on the ones with large area
    int triangle_priority_(double min_edge2) const {
        const double SQR2 = std::sqrt(2.0);  // threshold used to refine fractional exponent
        double length = 0.0;
        int exponent = 0;
        int posexponent = 0;

        // normalize the input so that min_edge2 ≥ 1.0
        // normalize the input so that min_edge2 ≥ 1.0
        if (min_edge2 >= 1.0) {
            length = min_edge2;
            posexponent = 1;  // mark as originally ≥ 1
            posexponent = 1;  // mark as originally ≥ 1
        } else {
            length = 1.0 / min_edge2;  // invert to bring into ≥ 1 range
            length = 1.0 / min_edge2;  // invert to bring into ≥ 1 range
            posexponent = 0;
        }

        // approximate log2(length) using repeated squaring and multiplication
        // approximate log2(length) using repeated squaring and multiplication
        while (length > 2.0) {
            int expincrement = 1;
            double multiplier = 0.5;

            // find the largest multiplier that keeps length * multiplier^2 > 1

            // find the largest multiplier that keeps length * multiplier^2 > 1
            while (length * multiplier * multiplier > 1.0) {
                expincrement *= 2;
                multiplier *= multiplier;
            }


            exponent += expincrement;
            length *= multiplier;  // reduce length accordingly
            length *= multiplier;  // reduce length accordingly
        }

        // final adjustment: multiply by 2 and add 1 if still greater than sqrt(2)
        // final adjustment: multiply by 2 and add 1 if still greater than sqrt(2)
        exponent = 2 * exponent + (length > SQR2 ? 1 : 0);

        // map to integer in the range [0, 4095]
        // map to integer in the range [0, 4095]
        int queuenumber;
        if (posexponent) {
            queuenumber = 2047 - exponent;  // for original values ≥ 1
            queuenumber = 2047 - exponent;  // for original values ≥ 1
        } else {
            queuenumber = 2048 + exponent;  // for original values < 1
            queuenumber = 2048 + exponent;  // for original values < 1
        }

        return queuenumber;  // priority value in [0, 4095]
        return queuenumber;  // priority value in [0, 4095]
    }

    // function evaluating whether a given triangle t should be marked as "bad" for refinement, based on a minimum angle and a maximum area constraint
    // if the triangle is considered bad, it returns a priority value for insertion into the bad triangles multimap;
    // otherwise, it returns -1.0, indicating that the triangle meets the quality criteria.
    double is_bad_triangle_(cell_t* t, double min_angle, double max_area) const{
        // extract triangle vertex coordinates
        coords_t A = t->halfedge()->prev()->node()->coords();
        coords_t B = t->halfedge()->node()->coords();
        coords_t C = t->halfedge()->next()->node()->coords();

        // compute squared edge lengths
        double a2 = (B - C).squaredNorm();  // edge opposite vertex A
        double b2 = (A - C).squaredNorm();  // edge opposite vertex B
        double c2 = (A - B).squaredNorm();  // edge opposite vertex C
        // compute squared edge lengths
        double a2 = (B - C).squaredNorm();  // edge opposite vertex A
        double b2 = (A - C).squaredNorm();  // edge opposite vertex B
        double c2 = (A - B).squaredNorm();  // edge opposite vertex C

        double angle_deg;

        // determine the smallest angle by identifying the shortest side
        // and computing the angle opposite to it
        double angle_deg;

        // determine the smallest angle by identifying the shortest side
        // and computing the angle opposite to it
        if (a2 <= b2 && a2 <= c2) {
            angle_deg = fdapde::internals::angle_between(C, A, B); // ∠CAB
            angle_deg = fdapde::internals::angle_between(C, A, B); // ∠CAB
        } else if (b2 <= c2) {
            angle_deg = fdapde::internals::angle_between(A, B, C); // ∠ABC
            angle_deg = fdapde::internals::angle_between(A, B, C); // ∠ABC
        } else {
            angle_deg = fdapde::internals::angle_between(B, C, A); // ∠BCA
            angle_deg = fdapde::internals::angle_between(B, C, A); // ∠BCA
        }

        // convert angle to cos² form for robust comparison
        double angle_cos2 = std::pow(std::cos(angle_deg * M_PI / 180.0), 2.0);
        double min_angle_cos2   = std::pow(std::cos(min_angle * M_PI / 180.0), 2.0);
        // convert angle to cos² form for robust comparison
        double angle_cos2 = std::pow(std::cos(angle_deg * M_PI / 180.0), 2.0);
        double min_angle_cos2   = std::pow(std::cos(min_angle * M_PI / 180.0), 2.0);

        // compute area of triangle
        // compute area of triangle
        double area = fdapde::internals::measure_2d_tri(A, B, C);

        // if the triangle violates either the minimum angle or maximum area constraint
        if (angle_cos2 > min_angle_cos2 || area > max_area) {

            double AB2 = (B - A).squaredNorm();
            double BC2 = (C - B).squaredNorm();
            double CA2 = (A - C).squaredNorm();
            double min_edge2 = std::min({AB2, BC2, CA2});

            // compute triangle priority based on shortest edge
            int priority = triangle_priority_(min_edge2);
            return static_cast<double>(priority);  // triangle is bad: return priority
        } else {
            return -1.0;  // triangle is good: ignore
        }
    }


    // function attempting to split the worst triangle with highest priority (small angle) in the bad triangle queue
    bool split_first_bad_triangle_(std::unordered_set<halfedge_t*>& segments,
                                std::unordered_set<halfedge_t*>& encroached_segments,
                                std::multimap<double, cell_t*>& bad_triangles,
                                double min_angle, double max_area) {
        
        // iterate over bad triangles in reverse priority order (worst triangle first)
        for (auto it = bad_triangles.rbegin(); it != bad_triangles.rend(); ) {


            cell_t* t = it->second;
            auto erase_it = std::prev(it.base());
            ++it; 
            bad_triangles.erase(erase_it);  // remove the current triangle from the multimap
            ++it; 
            bad_triangles.erase(erase_it);  // remove the current triangle from the multimap

            if (!t || !t->halfedge()) continue;

            // attempt to split the triangle (may fail if encroachment is detected)
            bool split = split_triangle_(t, segments, encroached_segments, bad_triangles, min_angle, max_area);
            if (split) {
                return true;  
            } else {
                // triangle could not be split now (due to encroached segments), reinsert it for future attempts
                double p1 = is_bad_triangle_(t, min_angle, max_area);
                bad_triangles.insert({p1, t});
                return true;  
            }
        }

        return false;  
        return false;  
    }


    // function attempting to split a bad triangle by inserting its circumcenter; if the circumcenter encroaches a segment of the PLC, it splits that segment instead
    bool split_triangle_(cell_t* t, std::unordered_set<halfedge_t*>& segments, std::unordered_set<halfedge_t*>& encroached_segments,
        std::multimap<double, cell_t*>& bad_triangles, double min_angle, double max_area) {
        
        halfedge_t* h1 = t->halfedge()->prev();
        halfedge_t* h2 = t->halfedge();
        halfedge_t* h3 = t->halfedge()->next();

        coords_t A = t->halfedge()->prev()->node()->coords();
        coords_t B = t->halfedge()->node()->coords();
        coords_t C = t->halfedge()->next()->node()->coords();

        double area = fdapde::internals::measure_2d_tri(A, B, C);
        // handle the case of acute angles for which triangle do not pass the constraint of min_angle 
        // but passes the one of max_area, so do not split it 
        // handle the case of acute angles for which triangle do not pass the constraint of min_angle 
        // but passes the one of max_area, so do not split it 
        if(area <= max_area){
            for (halfedge_t* h : {h1, h2, h3}) {
                if(is_edge_seditious_(h) || is_edge_seditious_(h->twin()))  return true;
            }
        }

        // compute the circumcenter of triangle ABC
        // compute the circumcenter of triangle ABC
        coords_t c = fdapde::internals::circumcenter(A, B, C); 
        
        // understand where the circumcenter is actually falling 
        cell_t* cf = find_triangle_local_(c, t); 
        halfedge_t* e1 = cf->halfedge()->prev();
        halfedge_t* e2 = cf->halfedge();
        halfedge_t* e3 = cf->halfedge()->next();
        // check whether the circumcenter c encroaches any segment of the triangle cf
        bool flag = false;
        for (halfedge_t* e : {e1, e2, e3}) {
            if(e->is_segment() && e->cell() && check_encroachment_(e, c)) {
                    encroached_segments.insert(e);     
                    flag = true;  
            }
        }
        if (flag) return false;

        // if no encroachment is detected, insert the circumcenter into the mesh
        // if no encroachment is detected, insert the circumcenter into the mesh
        node_t* circ = dcel_.insert_node(node_t(dcel_.n_nodes(), false, c));
        // insert the new node into the triangulation finally splitting the triangle of interest 
        insert_vertex_(circ, cf, encroached_segments, bad_triangles, min_angle, max_area);
        return true;
    }

    // function doing the same task of flip() method, but necessary to keep track of encroached segments and bad triangles eventually created 
    void flip_refinement_(std::unordered_set<halfedge_t*>& encroached_segments,
        std::multimap<double, cell_t*>& bad_triangles, double min_angle, double max_area) {

        std::unordered_set<halfedge_t*> halfedges_to_check;
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it,++it) {
            if(!it->is_segment())
                halfedges_to_check.insert(&(*it));
        }

        while (!halfedges_to_check.empty()) {
            halfedge_t* edge = *(halfedges_to_check.begin());
            halfedges_to_check.erase(edge);
            halfedges_to_check.erase(edge->twin());

            cell_t* neighbor = edge->twin()->cell();

            coords_t A = edge->node()->coords();
            coords_t B = edge->twin()->node()->coords();
            coords_t C = edge->prev()->node()->coords();
            coords_t D = edge->twin()->prev()->node()->coords();
    
            if (fdapde::internals::in_circle(A, B, C, D) || fdapde::internals::in_circle(A, D, B, C)) {
                
                halfedge_t* e = edge;
                
                remove_from_multimap_(bad_triangles, edge->cell());
                remove_from_multimap_(bad_triangles, edge->twin()->cell());
                halfedge_t* prev= edge->prev();
                halfedge_t* twin_prev= edge->twin()->prev();
                
                dcel_.remove_edge(edge);
                halfedge_t* new_edge = dcel_.insert_edge(prev, twin_prev);
            
                if (new_edge) {
                    if(!new_edge->prev()->is_segment()){
                        halfedges_to_check.insert(new_edge->prev());
    
                    }
                    if(!new_edge->next()->is_segment()){
                        halfedges_to_check.insert(new_edge->next());
      
                    }
                    if(!new_edge->twin()->prev()->is_segment()){
                        halfedges_to_check.insert(new_edge->twin()->prev());
                   
                    }
                    if(!new_edge->twin()->next()->is_segment()){
                        halfedges_to_check.insert(new_edge->twin()->next());
            
                    }

                    auto already_present = [&](cell_t* c) {return std::find_if(bad_triangles.begin(), bad_triangles.end(),[c](const auto& entry) { return entry.second == c; }) != bad_triangles.end();
};
                    double p1 = is_bad_triangle_(new_edge->cell(), min_angle, max_area);
                    if (p1 >= 0.0 && !already_present(new_edge->cell())) 
                        bad_triangles.insert({p1, new_edge->cell()});
                    
                    if(new_edge->prev()->is_segment() && new_edge->prev()->cell() && check_encroachment_(new_edge->prev())){
                        encroached_segments.insert(new_edge->prev());
                    }
                    if(new_edge->next()->is_segment() && new_edge->next()->cell() && check_encroachment_(new_edge->next())){
                        encroached_segments.insert(new_edge->next());
                    }
                    double p2 = is_bad_triangle_(new_edge->twin()->cell(), min_angle, max_area);
                    if (p2 >= 0.0 && !already_present(new_edge->twin()->cell())) 
                        bad_triangles.insert({p2, new_edge->twin()->cell()});
                    
                   if(new_edge->twin()->prev()->is_segment() && new_edge->twin()->prev()->cell() && check_encroachment_(new_edge->twin()->prev())){
                        encroached_segments.insert(new_edge->twin()->prev());
                    }
                    if(new_edge->twin()->next()->is_segment() && new_edge->twin()->next()->cell() && check_encroachment_(new_edge->twin()->next())){
                        encroached_segments.insert(new_edge->twin()->next());
                    }
                }
            }
        }
    }
    
//--------------------- methods to implement the Bowyer-Watson algorithm --------------------------------------
// these functions work like the Conflict-Graph algorithm for cavity expansion, but they are able to insert new points on the fly, as conflics are not detected
// necessary for the insertion of triangles' circumcenters in the refinement routine; for this reason they also keep track of encroached segments and bad triangles eventually created 

    // function that digs a cavity around a new node u, starting from halfedge vw
    void dig_cavity_(node_t* u, halfedge_t* vw, std::unordered_set<halfedge_t*>& encroached_segments,
        std::multimap<double, cell_t*>& bad_triangles, double min_angle, double max_area) { 

        auto already_present = [&](cell_t* c) {return std::find_if(bad_triangles.begin(), bad_triangles.end(),[c](const auto& entry) { return entry.second == c; }) != bad_triangles.end();};

        if(vw->is_segment()){ // vw cannot be removed, so directly create a new triangle from vw and u
            add_triangle_(vw,std::vector<node_t*> {u});
            cell_t* t = vw->cell();
            double p1 = is_bad_triangle_(t, min_angle, max_area);
            if (p1 >= 0.0 && !already_present(t)) 
                    bad_triangles.insert({p1, t});
            if (check_encroachment_(vw)) {
                encroached_segments.insert(vw);
            }
            return;
        }

        // 3rd node of vw triangle
        node_t* x = dcel_.adjacent(vw);
        if (!x) {
            return;
        }
        bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(u->coords(), vw->node()->coords(), vw->twin()->node()->coords());
          
          
        bool inside;
        if (ccw) {
            inside = fdapde::internals::in_circle(u->coords(), vw->node()->coords(), vw->twin()->node()->coords(), x->coords());
        } else {
            inside = fdapde::internals::in_circle(u->coords(), vw->twin()->node()->coords(), vw->node()->coords(), x->coords());
        }
        if (inside) { // u-vw is not Delaunay, cavity is expanded
            halfedge_t* wv = vw->twin();
            halfedge_t* vx = vw->twin()->next();
            halfedge_t* xw = vw->twin()->prev(); 
            remove_from_multimap_(bad_triangles, vw->cell());
            remove_from_multimap_(bad_triangles, vw->twin()->cell());
            dcel_.remove_edge(vw);  // remove vw since, after u insertion, it's not locally Delaunay
            dig_cavity_(u, vx, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, xw, encroached_segments, bad_triangles, min_angle, max_area);
        } else {  // create triangle u-vw since it's Delaunay
            add_triangle_(vw,std::vector<node_t*> {u});
            cell_t* t = vw->cell();
            double p2 = is_bad_triangle_(t, min_angle, max_area);
            if (p2 >= 0.0 && !already_present(t)) 
                bad_triangles.insert({p2, t});
                
            return;
        } 
    }
    
    // functions that inserts a new vertex u into the triangulation; triangle is the cell in which u falls
    void insert_vertex_(node_t* u, cell_t* triangle, std::unordered_set<halfedge_t*>& encroached_segments,
        std::multimap<double, cell_t*>& bad_triangles , double min_angle, double max_area) {

        halfedge_t* vw = triangle->halfedge();
        halfedge_t* wx = vw->next();
        halfedge_t* xv = vw->prev();
    
        const coords_t& v = vw->node()->coords();
        const coords_t& w = wx->node()->coords();
        const coords_t& x = xv->node()->coords();
        bool found_on_edge = false;
    
        // if u falls on vw, cavity is expanded on both sides of vw
        if (fdapde::internals::contains(u->coords(), v, w)) {
            halfedge_t* twin_next = vw->twin()->next();
            halfedge_t* twin_prev = vw->twin()->prev();
  
            remove_from_multimap_(bad_triangles, vw->cell());
            remove_from_multimap_(bad_triangles, vw->twin()->cell());

            dcel_.remove_edge(vw);
            dig_cavity_(u, wx, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, xv, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, twin_next, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, twin_prev, encroached_segments, bad_triangles, min_angle, max_area);
            found_on_edge = true;
        } else if (fdapde::internals::contains(u->coords(), w, x)) {
            halfedge_t* twin_next = wx->twin()->next();
            halfedge_t* twin_prev = wx->twin()->prev();
    
            remove_from_multimap_(bad_triangles, wx->cell());
            remove_from_multimap_(bad_triangles, wx->twin()->cell());

            dcel_.remove_edge(wx);
            dig_cavity_(u, vw, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, xv, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, twin_next, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, twin_prev, encroached_segments, bad_triangles, min_angle, max_area);
            found_on_edge = true;
        } else if (fdapde::internals::contains(u->coords(), x, v)) {
            halfedge_t* twin_next = xv->twin()->next();
            halfedge_t* twin_prev = xv->twin()->prev();

            remove_from_multimap_(bad_triangles, xv->cell());
            remove_from_multimap_(bad_triangles, xv->twin()->cell());

            dcel_.remove_edge(xv);
            dig_cavity_(u, vw, encroached_segments, bad_triangles, min_angle,max_area);
            dig_cavity_(u, wx, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, twin_next, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, twin_prev, encroached_segments, bad_triangles, min_angle, max_area);
            found_on_edge = true;
        }
    
        if (!found_on_edge) { // u does not fall on any edge, cavity is expanded starting from triangle edges
            dig_cavity_(u, vw, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, wx, encroached_segments, bad_triangles, min_angle, max_area);
            dig_cavity_(u, xv, encroached_segments, bad_triangles, min_angle, max_area);
        }
    }

//--------------------- end of methods to implement the Bowyer-Watson algorithm --------------------------------------

    // checks the global quality of the current triangulation.
    bool check_quality_(double min_angle, double max_area) const {
        bool all_ok = true;     // will remain true only if no violations are found
        int bad_count = 0;      // counter for triangles that fail the quality checks

        // loop over all triangles (cells) in the DCEL
        for (auto it = dcel_.cells_cbegin(); it != dcel_.cells_cend(); ++it) {
            const cell_t* t = &(*it);

            // retrieve coordinates of triangle vertices
            // retrieve coordinates of triangle vertices
            coords_t A = t->halfedge()->prev()->node()->coords();
            coords_t B = t->halfedge()->node()->coords();
            coords_t C = t->halfedge()->next()->node()->coords();

            // compute the three internal angles of triangle ABC
            // compute the three internal angles of triangle ABC
            double angleA = fdapde::internals::angle_between(C, A, B); // ∠CAB
            double angleB = fdapde::internals::angle_between(A, B, C); // ∠ABC
            double angleC = fdapde::internals::angle_between(B, C, A); // ∠BCA
            double angle = std::min({angleA, angleB, angleC});     // smallest internal angle

            // compute area of the triangle
            double area = fdapde::internals::measure_2d_tri(A, B, C);
            double angle = std::min({angleA, angleB, angleC});     // smallest internal angle

            // compute area of the triangle
            double area = fdapde::internals::measure_2d_tri(A, B, C);

            // check if the triangle violates any of the two constraints
            if (angle < min_angle || area > max_area) {
            // check if the triangle violates any of the two constraints
            if (angle < min_angle || area > max_area) {
                all_ok = false;
                bad_count++;
            }
        }

        // summary output for diagnostic purposes
        // summary output for diagnostic purposes
        if (all_ok) {
            std::cout << "All triangles satisfy the quality constraints." << std::endl;
        } else {
            std::cout << "Found " << bad_count << " triangles violating quality constraints." << std::endl;
        }

        return all_ok;
    }


    //------------------ end of support methods to refine the CDT ----------------------------------------------


    //------------------ end of support methods to refine the CDT ----------------------------------------------
    
};
  
}  // namespace fdapde

#endif // _DELAUNAY_H_