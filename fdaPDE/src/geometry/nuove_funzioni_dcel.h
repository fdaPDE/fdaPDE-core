      
        void set_on_boundary(bool b) {
            boundary_ = b;
        }

        halfedge_t* cell_on_boundary() const { 
            if (!h_) return nullptr;  
        
            halfedge_t* h1 = h_;
            do {
                if (h1->on_boundary()) return h1;  
                h1 = h1->next(); 
            } while (h1 != h_);  
        
            return nullptr;  
        }


           //////////////////CODICE DI PROVA /////////////////////////////
    //(deve rimuovere tutti i lati interni e creare una cavità)
    void remove_polygon(const cell_t* cell){    //DA CORREGGERE QUEI DO-WHILE
       
        halfedge_t* b = cell->on_boundary();
        if (b){ // only remove edges on boundary    DOVREBBE ARRIVARE FINO A CASO b. IN CUI RIMANE SOLO 1 LATO SUL BORDO
        do{
         b = remove_edge(b);
        }while(b->on_boundary())
        return; 
        }

        halfedge_t* h1 = cell->halfedge();
        halfedge_t* ending = h1->twin()->next();
        do{
        h1 = remove_edge(h1);
        }while(h1!=ending)

    // remove cells --> already done by remove_edge  IN TEORIA, RICONTROLLARE
    }
    /////////////////FINE CODICE DI PROVA //////////////////

        ////////////////////CODICE DI PROVA ////////////////////////
        bool remove_node(const node_t& node) {
            auto it = std::find(nodes_begin(), nodes_end(), node);
            if (it == nodes_end()) return false; 
            
            nodes_.remove(node);
            n_nodes_--;
            return true;
        }
        ///////////////////FINE CODICE DI PROVA ////////////////////

            ///////////////////////////// CODICE DI PROVA///////////////////////////////////////7
    halfedge_t* remove_edge(halfedge_t* v1){
        if (v1->on_boundary()) { // v1 is on the boundary
        //2 cases: 
        //a. v1 and its next/prev are on boundary --> elongate v1->next or prev (SE NO NON PERMETTERE)
        if(v1->next->on_boundary() || v1->prev()->on_boundary()){
            v1->next()->set_prev(v1->prev());
            v1->prev()->set_next(v1->next());
            v1->next()->set_node(v1->node());     
            halfedge_t* v2= v1->twin();
            v2->next()->set_prev(v2->prev());
            v2->prev()->set_next(v2->next());
            v2->next()->set_node(v2->node());
            // remove edges, cell is the same so it doesn't need to be removed
            halfedge_t* next= v1->next();
            n_halfedges_ -= 2;
            auto it1 = std::find(halfedges_begin(), halfedges_end(), v1);
            auto it2 = std::find(halfedges_begin(), halfedges_end(), v2);
            halfedges_.erase(it1);
            halfedges_.erase(it2); 
            return next;
        }
        //b. only v1 is on boundary and its next/prev isn't
        else{
            halfedge_t* v2= v1->twin();
            v1->next()->set_prev(v2->prev());
            v2->prev()->set_next(v1->next());    
            v1->prev()->set_next(v2->next());
            v2->next()->set_prev(v1_prev());
            // remove edges and cell
            n_cells_--;
            auto it = std::find(cells_begin(), cells_end(), v1->cell());
            halfedge_t* end = v1, begin = v1->next();
            do {
               begin->set_cell(nullptr);
               begin->node()->set_on_boundary(true);
               begin = begin->next();
            } while (begin!=end)
            cells_.erase(it);
            halfedge_t* next= v1->next();
            n_halfedges_ -= 2;
            auto it1 = std::find(halfedges_begin(), halfedges_end(), v1);
            auto it2 = std::find(halfedges_begin(), halfedges_end(), v2);
            halfedges_.erase(it1);
            halfedges_.erase(it2); 
            return next; 
        }
    } 

    halfedge_t* v2 = v1->twin();
    // set next of v1_prev to v2_next
    v1->prev()->set_next(v2->next());      
    v2->next()->set_prev(v1->prev());
    // set next of v2_prev to v1_next
    v2->prev()->set_next(v1->next());
    v1->next()->set_prev(v2->prev());
    // remove v2's cell
    n_cells_--;
    auto it = std::find(cells_begin(), cells_end(), v2->cell());
    cells_.erase(it);    
    // insert halfedges in v1's cell
    halfedge_t* end = v2, begin = v2->next();
    cell_t* c1 = v1->cell();
    c1->set_halfedge(begin); // either this or v1->prev()
    do {
        begin->set_cell(c1);
        begin = begin->next();
    } while (begin != end);
    // remove v1 and v2
    halfedge_t* next= v1->next();
    n_halfedges_ -= 2;
    auto it1 = std::find(halfedges_begin(), halfedges_end(), v1);
    auto it2 = std::find(halfedges_begin(), halfedges_end(), v2);
    halfedges_.erase(it1);
    halfedges_.erase(it2);  
    return next;
    }

    /////////////////////////////// FINE CODICE DI PROVA//////////////////////////


        ////////////////////////////// CODICE DI PROVA //////////////////////////////
        node_t* adjacent(halfedge_t* edge) const {
            return (edge->twin()) ? edge->twin()->prev()->node() : nullptr;  
        }
        ////////////////////////////// FINE CODICE DI PROVA //////////////////////////







//il find_triangle mi dice che u sta in tringle ---> concetto da esprimere in funzione che fa la triangolazione 
void insert_vertex(node_t* u,cell_t* triangle) {
 // si possono evitare i tre sotto e scrivere tutto esplicito
    halfedge_t* vw = triangle->halfedge();
    halfedge_t* wx = vw->next();
    halfedge_t* xv = vw->prev();
// da ricontrollare se il bordo è gestito correttamente: idea è se un lato è sul bordo poi il delete_triangle lo cancella 
// quindi non posso fare la dig_cavity li 
    bool vw_is_boundary = vw->on_boundary();
    bool wx_is_boundary = wx->on_boundary();
    bool xv_is_boundary = xv->on_boundary();

    delete_triangle(triangle);
    
    if (!vw_is_boundary) dig_cavity(u, vw);
    if (!wx_is_boundary) dig_cavity(u, wx);
    if (!xv_is_boundary) dig_cavity(u, xv);
}



    // dcel.remove_polygon(find_halfedge_from_id(20)->cell());
    /*
    DCEL<2, 2>::node_t node1(dcel.n_nodes(),nullptr,false,0.5,1.), node2(dcel.n_nodes()+1,nullptr,false,1.5,1.);
    dcel.insert_node(node1);
    dcel.insert_node(node2);
    DCEL<2, 2>::halfedge_t h_i1(dcel.n_halfedges()+1000,&node1);
    DCEL<2, 2>::halfedge_t h_i2(dcel.n_halfedges()+1001,&node2);
    h_i1.set_cell(nullptr);
    h_i2.set_cell(nullptr);  //SE NO BISOGNA MODIFICARE IL COSTRUTTORE
    node1.set_halfedge(&h_i1);
    node2.set_halfedge(&h_i2);
    DCEL<2, 2>::halfedge_t* he1 = find_halfedge_from_node(1);
    DCEL<2, 2>::halfedge_t* hi1=&h_i1;
    DCEL<2, 2>::halfedge_t* hi2=&h_i2;

    //TEST insert e remove edge
    DCEL<2, 2>::halfedge_t* h10= dcel.insert_edge(he1, hi1);
    DCEL<2, 2>::halfedge_t* h12= dcel.insert_edge(hi2, he1);
    DCEL<2, 2>::halfedge_t* h3=dcel.insert_edge(h10->twin(), h12);
    dcel.insert_edge(find_halfedge_from_id(3), find_halfedge_from_id(12));
    dcel.insert_edge(find_halfedge_from_id(14), find_halfedge_from_id(4));
    dcel.insert_edge(find_halfedge_from_id(14), find_halfedge_from_id(3));
    dcel.insert_edge(find_halfedge_from_id(0), find_halfedge_from_id(18));   
    dcel.remove_edge(find_halfedge_from_id(10));
    dcel.remove_edge(find_halfedge_from_id(13));
    dcel.remove_edge(find_halfedge_from_id(18));
    dcel.remove_edge(find_halfedge_from_id(21));
    dcel.remove_edge(find_halfedge_from_id(16));
    dcel.remove_edge(find_halfedge_from_id(14));
    */







    halfedge_t* remove_edge(halfedge_t* v1){
        if(!v1) return nullptr;
        /*
        if (v1->on_boundary()) { // v1 is on the boundary
        //2 cases: 
        //a. v1 and its next/prev are on boundary --> elongate v1->next or prev (SE NO NON PERMETTERE)
        if(v1->next()->on_boundary() || v1->prev()->on_boundary()){
            v1->next()->set_prev(v1->prev());
            v1->prev()->set_next(v1->next());
            v1->next()->set_node(v1->node());     
            halfedge_t* v2= v1->twin();
            v2->next()->set_prev(v2->prev());
            v2->prev()->set_next(v2->next());
            v2->next()->set_node(v2->node());
            // remove edges, cell is the same so it doesn't need to be removed
            halfedge_t* next= v1->next();
            n_halfedges_ -= 2;
            auto it1 = std::find(halfedges_begin(), halfedges_end(), v1);
            auto it2 = std::find(halfedges_begin(), halfedges_end(), v2);
            halfedges_.erase(it1);
            halfedges_.erase(it2); 
            return next;
        }
        //b. only v1 is on boundary and its next/prev isn't
        else{
            halfedge_t* v2= v1->twin();
            v1->next()->set_prev(v2->prev());
            v2->prev()->set_next(v1->next());    
            v1->prev()->set_next(v2->next());
            v2->next()->set_prev(v1->prev());
            // remove edges and cell
            n_cells_--;
            auto it = std::find(cells_begin(), cells_end(), v1->cell());
            halfedge_t* end = v1;
            halfedge_t* begin = v1->next();
            do {
               begin->set_cell(nullptr);
               begin->node()->set_on_boundary(true);
               begin = begin->next();
            } while (begin!=end);
            cells_.erase(it);
            halfedge_t* next= v1->next();
            n_halfedges_ -= 2;
            auto it1 = std::find(halfedges_begin(), halfedges_end(), v1);
            auto it2 = std::find(halfedges_begin(), halfedges_end(), v2);
            halfedges_.erase(it1);
            halfedges_.erase(it2); 
            return next; 
        }
        } */
        halfedge_t* v2 = v1->twin();
        // insert halfedges in v1's cell
        halfedge_t* end = v2;
        halfedge_t* begin = v2->next();
        cell_t* c1 = v1->cell();
        c1->set_halfedge(begin); // either this or v1->prev()
        do {
            begin->set_cell(c1);
            begin = begin->next();
        } while (begin != end);
        // set next of v1_prev to v2_next
        v1->prev()->set_next(v2->next());      
        v2->next()->set_prev(v1->prev());
        // set next of v2_prev to v1_next
        v2->prev()->set_next(v1->next());
        v1->next()->set_prev(v2->prev());
        // remove v2's cell
        n_cells_--;
        cell_t* c2=v2->cell();
        if (c2) {
            //auto it = std::find(cells_begin(), cells_end(), v2->cell());
            //auto it = std::find_if(cells_.begin(), cells_.end(), [=](const cell_t& c) { return c.id() == c2->id(); });
            cell_iterator it = cells_.begin();
            while (it != cells_.end()) {
                if (it->id() == c2->id()) 
                    break;  // Fermiamo il loop se troviamo la cella
                ++it;
            }
            if (it != cells_.end()) 
                cells_.erase(it);
        }
        // remove v1 and v2
        halfedge_t* next= v1->next();
        n_halfedges_ -= 2;
        if (v1 && v2) {
            //auto it1 = std::find(halfedges_begin(), halfedges_end(), v1);
            //auto it2 = std::find(halfedges_begin(), halfedges_end(), v2);
            //auto it1 = std::find_if(halfedges_.begin(), halfedges_.end(), [=](const halfedge_t& h) { return h.id() == some_halfedge_ptr1->id();});
            //auto it2 = std::find_if(halfedges_.begin(), halfedges_.end(), [=](const halfedge_t& h) { return h.id() == some_halfedge_ptr2->id();});
            halfedge_iterator it1 = halfedges_.begin();
            while (it1 != halfedges_.end()) {
                if (it1->id() == v1->id()) {
                    break;
                }
                ++it1;
            }
            halfedge_iterator it2 = halfedges_.begin();
            while (it2 != halfedges_.end()) {
                if (it2->id() == v2->id()) {
                    break;
                }
                ++it2;
            }
            if (it1 != halfedges_.end() && it2 != halfedges_.end()) {
                halfedges_.erase(it1);
                halfedges_.erase(it2);
            }
        }
        return next; 
    }

    halfedge_t* add_polygon(halfedge_t* v, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& nodes){
        // update n_halfedges_ and n_cells_
        // n_nodes_ is already updated by insert_node
        int nodes_polygon= nodes.rows();
        cell_t* c= v->cell();

        std::vector<halfedge_t*> ghost_halfedges(nodes_polygon +2 ); //O(n)
        ghost_halfedges[0] = v;
        ghost_halfedges[1] = v->next();

        // add nodes and create ghost halfedges
        for (int i = 0; i < nodes_polygon; ++i) {
            node_t* n = insert_node(node_t(n_nodes_, /* boundary = */ false, nodes.row(i)));
            halfedges_.emplace_back(n_halfedges_+ 1000 + i, n);
            halfedge_t* h = std::addressof(halfedges_.back());    //NON USARE emplace_halfedge PERCHè INCASINA GLI INDICI, così li posso controllare io!
            ghost_halfedges[i+2] = h;
        }
        // add edges
        int count=0;
        for (int i = 0; i < nodes_polygon+2 ; ++i) {
            halfedge_t* h1 = ghost_halfedges[i];
            halfedge_t* h2 = ghost_halfedges[(i + 1) % (nodes_polygon+2)];  
            ghost_halfedges[(i + 1) % (nodes_polygon+2)]= insert_edge(h1, h2)->next();  
            if(!h2->next()) {
                auto it = std::find_if(halfedges_.begin(), halfedges_.end(), [=](const halfedge_t& h) { return h.id() == h2->id();});
                if (it != halfedges_.end()){
                    halfedges_.erase(it);
                    count++;
                } 
            } 
        }

        return c->halfedge();
    }




    //   void remove_polygon(const cell_t* cell){    //PRENDE IN INGRESSO CELLA O HALFEDGE?
       
        /*halfedge_t* b = cell->on_boundary();
        if (b){ // only remove edges on boundary    DOVREBBE ARRIVARE FINO A CASO b. IN CUI RIMANE SOLO 1 LATO SUL BORDO
        do{
         b = remove_edge(b);
        }while(b->on_boundary())
        return; 
        }*/
/*
        halfedge_t* h1 = cell->halfedge();
        halfedge_t* ending = h1->twin()->next();
        do{
            h1 = remove_edge(h1);
        }while(h1!=ending);
    }
*/





void add_triangle(const coords_t& A, const coords_t& B, const coords_t& C) {
    std::cout << "🔺 Aggiungo triangolo con vertici: " << A.transpose() << ", " 
              << B.transpose() << ", " << C.transpose() << std::endl;
    // ⚠️ Controllo per evitare triangoli degeneri
    if (A == B || B == C || C == A) {
        std::cerr << "❌ Errore: triangolo degenere con vertici coincidenti!" << std::endl;
        return;
    }


    // 1️⃣ Cerchiamo i nodi esistenti
    node_t* nA = find_node(A);
    node_t* nB = find_node(B);
    node_t* nC = find_node(C);

    if (!nA) nA = insert_node(node_t(n_nodes_, false, A));
    if (!nB) nB = insert_node(node_t(n_nodes_, false, B));
    if (!nC) nC = insert_node(node_t(n_nodes_, false, C));

    std::cout << "Nodi: A=" << nA->id() << " B=" << nB->id() << " C=" << nC->id() << std::endl;

    // 2️⃣ Cerchiamo se gli half-edge esistono già
    halfedge_t* hAB = find_halfedge(nA, nB);
    halfedge_t* hBC = find_halfedge(nB, nC);
    halfedge_t* hCA = find_halfedge(nC, nA);

    std::cout << "Edge trovati: AB=" << (hAB ? hAB->id() : -1) 
              << " BC=" << (hBC ? hBC->id() : -1) 
              << " CA=" << (hCA ? hCA->id() : -1) << std::endl;

    // 3️⃣ Se non esistono, li creiamo
    if (!hAB) {
        hAB = emplace_halfedge_(nA);
        std::cout << "Creato half-edge AB con ID: " << hAB->id() << std::endl;
    }
    if (!hBC) {
        hBC = emplace_halfedge_(nB);
        std::cout << "Creato half-edge BC con ID: " << hBC->id() << std::endl;
    }
    if (!hCA) {
        hCA = emplace_halfedge_(nC);
        std::cout << "Creato half-edge CA con ID: " << hCA->id() << std::endl;
    }

    // **CHECK: Tutti gli half-edge devono esistere**
    if (!hAB || !hBC || !hCA) {
        std::cerr << "❌ Errore: uno degli half-edge è NULL dopo l'inserimento!" << std::endl;
        return;
    }

    // 4️⃣ Creiamo e colleghiamo i twin
    halfedge_t* hBA = find_halfedge(nB, nA);
    halfedge_t* hCB = find_halfedge(nC, nB);
    halfedge_t* hAC = find_halfedge(nA, nC);

    if (!hBA) hBA = emplace_halfedge_(nB);
    if (!hCB) hCB = emplace_halfedge_(nC);
    if (!hAC) hAC = emplace_halfedge_(nA);

    // 5️⃣ Impostiamo i twin
    hAB->set_twin(hBA); hBA->set_twin(hAB);
    hBC->set_twin(hCB); hCB->set_twin(hBC);
    hCA->set_twin(hAC); hAC->set_twin(hCA);

    // 6️⃣ Colleghiamo next e prev
    hAB->set_next(hBC); 
    hBC->set_next(hCA); 
    hCA->set_next(hAB);

    hBA->set_prev(hCB); 
    hCB->set_prev(hAC); 
    hAC->set_prev(hBA);

    hAB->set_prev(hCA); 
    hBC->set_prev(hAB); 
    hCA->set_prev(hBC);

    hBA->set_next(hAC); 
    hCB->set_next(hBA); 
    hAC->set_next(hCB);

    // 🔍 **Check extra per assicurarci che tutti i next siano assegnati correttamente**
    std::cout << "🔍 Debug connessioni next dopo assegnazione: " << std::endl;
    std::cout << "AB: " << hAB->id() << " -> " << (hAB->next() ? std::to_string(hAB->next()->id()) : "nullptr") << std::endl;
    std::cout << "BC: " << hBC->id() << " -> " << (hBC->next() ? std::to_string(hBC->next()->id()) : "nullptr") << std::endl;
    std::cout << "CA: " << hCA->id() << " -> " << (hCA->next() ? std::to_string(hCA->next()->id()) : "nullptr") << std::endl;

    if (!hAB->next() || !hBC->next() || !hCA->next()) {
        std::cerr << "❌ Errore: almeno un half-edge non ha next() assegnato correttamente!" << std::endl;
        return;
    }
    

    // 7️⃣ Creiamo la cella
    cells_.push_back(cell_t(n_cells_++));
    cell_t* triangle = std::addressof(cells_.back());

    // Assegniamo la cella agli edge
    hAB->set_cell(triangle);
    hBC->set_cell(triangle);
    hCA->set_cell(triangle);

    triangle->set_halfedge(hAB);

    std::cout << "✅ Triangolo aggiunto con successo: " << triangle->id() << std::endl;
}

halfedge_t* find_halfedge(node_t* from, node_t* to) {
    if (!from || !to) return nullptr; // Protezione extra

    for (auto it = halfedges_.begin(); it != halfedges_.end(); ++it) {
        if (it->node() == from && it->next() && it->next()->node() == to) {
            std::cout << "✅ Trovato half-edge esistente da " << from->id() << " a " << to->id() << ": " << it->id() << std::endl;
            return std::addressof(*it);
        }
    }

    std::cout << "❌ Nessun half-edge trovato da " << from->id() << " a " << to->id() << std::endl;
    return nullptr;
}



node_t* find_node(const coords_t& coords) {
    for (auto it = nodes_begin(); it != nodes_end(); ++it) {
        if ((it->coords() - coords).norm() < 1e-8) { // Tolleranza per errore numerico
            return std::addressof(*it);
        }
    }
    return nullptr; // Se non trova nulla, restituisce nullptr
}




void add_triangle(const coords_t& p1, const coords_t& p2, const coords_t& p3) {
    // Funzione per trovare un nodo esistente
    auto find_node = [&](const coords_t& coords) -> node_t* {
        for (auto it = nodes_begin(); it != nodes_end(); ++it) {
            if ((it->coords() - coords).norm() < 1e-10) { // Tolleranza per errore numerico
                return std::addressof(*it);
            }
        }
        return nullptr;
    };

    // Creazione o ricerca dei nodi
    node_t* n1 = find_node(p1);
    if (!n1) n1 = insert_node(node_t(n_nodes_, /* boundary = */ false, p1));
    
    node_t* n2 = find_node(p2);
    if (!n2) n2 = insert_node(node_t(n_nodes_, /* boundary = */ false, p2));
    
    node_t* n3 = find_node(p3);
    if (!n3) n3 = insert_node(node_t(n_nodes_, /* boundary = */ false, p3));

    // Funzione per trovare un half-edge esistente
    auto find_halfedge = [&](node_t* from, node_t* to) -> halfedge_t* {
        for (auto it = halfedges_begin(); it != halfedges_end(); ++it) {
            if (it->node() == from && it->next() && it->next()->node() == to) {
                return std::addressof(*it);
            }
        }
        return nullptr;
    };

    // Creazione o ricerca dei half-edge e delle twin
    halfedge_t* h1 = find_halfedge(n1, n2);
    if (!h1) h1 = emplace_halfedge_(n1);
    if (!h1) { std::cerr << "Errore: Creazione h1 fallita!\n"; return; }
    
    halfedge_t* h2 = find_halfedge(n2, n3);
    if (!h2) h2 = emplace_halfedge_(n2);
    if (!h2) { std::cerr << "Errore: Creazione h2 fallita!\n"; return; }
    
    halfedge_t* h3 = find_halfedge(n3, n1);
    if (!h3) h3 = emplace_halfedge_(n3);
    if (!h3) { std::cerr << "Errore: Creazione h3 fallita!\n"; return; }

    halfedge_t* h1_twin = find_halfedge(n2, n1);
    if (!h1_twin) h1_twin = emplace_halfedge_(n2);
    
    halfedge_t* h2_twin = find_halfedge(n3, n2);
    if (!h2_twin) h2_twin = emplace_halfedge_(n3);
    
    halfedge_t* h3_twin = find_halfedge(n1, n3);
    if (!h3_twin) h3_twin = emplace_halfedge_(n1);

    // Controlli sui twin
    if (h1 && h1_twin) { h1->set_twin(h1_twin); h1_twin->set_twin(h1); }
    if (h2 && h2_twin) { h2->set_twin(h2_twin); h2_twin->set_twin(h2); }
    if (h3 && h3_twin) { h3->set_twin(h3_twin); h3_twin->set_twin(h3); }

    // Collegamento delle strutture next-prev
    if (h1 && h2) { h1->set_next(h2); h2->set_prev(h1); }
    if (h2 && h3) { h2->set_next(h3); h3->set_prev(h2); }
    if (h3 && h1) { h3->set_next(h1); h1->set_prev(h3); }

    if (h1_twin && h3_twin) { h1_twin->set_next(h3_twin); h3_twin->set_prev(h1_twin); }
    if (h3_twin && h2_twin) { h3_twin->set_next(h2_twin); h2_twin->set_prev(h3_twin); }
    if (h2_twin && h1_twin) { h2_twin->set_next(h1_twin); h1_twin->set_prev(h2_twin); }

    // Creazione della cella per il triangolo
    cells_.push_back(cell_t(n_cells_++));
    cell_t* c = std::addressof(cells_.back());
    c->set_halfedge(h1);

    // Associazione cella agli half-edge
    if (h1) h1->set_cell(c);
    if (h2) h2->set_cell(c);
    if (h3) h3->set_cell(c);

    // Associa i nodi ai loro half-edge
    if (n1) n1->set_halfedge(h1);
    if (n2) n2->set_halfedge(h2);
    if (n3) n3->set_halfedge(h3);
}




void remove_triangle(const cell_t* triangle) {
    dcel_.remove_polygon(triangle);  // Riutilizziamo remove_polygon
}


halfedge_t* add_polygon(halfedge_t* v, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& nodes){
    // update n_halfedges_ and n_cells_
    // n_nodes_ is already updated by insert_node
    int nodes_polygon= nodes.rows();
    cell_t* c= v->cell();

    std::vector<halfedge_t*> ghost_halfedges(nodes_polygon +2 ); //O(n)
    ghost_halfedges[0] = v;
    ghost_halfedges[1] = v->next();

    // add nodes and create ghost halfedges
   for (int i = 0; i < nodes_polygon; ++i) {
       node_t* n = insert_node(node_t(n_nodes_, /* boundary = */ false, nodes.row(i)));
   // add nodes and create ghost halfedges

        halfedges_.emplace_back(n_halfedges_+ 1000 + i, n);
        halfedge_t* h = std::addressof(halfedges_.back());    //NON USARE emplace_halfedge PERCHè INCASINA GLI INDICI, così li posso controllare io!
        ghost_halfedges[i+2] = h;
    }
    // add edges
    int count=0;
    for (int i = 0; i < nodes_polygon+2 ; ++i) {
        halfedge_t* h1 = ghost_halfedges[i];
        halfedge_t* h2 = ghost_halfedges[(i + 1) % (nodes_polygon+2)];  
        ghost_halfedges[(i + 1) % (nodes_polygon+2)]= insert_edge(h1, h2)->next();  
        if(!h2->next()) {
            auto it = std::find_if(halfedges_.begin(), halfedges_.end(), [=](const halfedge_t& h) { return h.id() == h2->id();});
            if (it != halfedges_.end()){
                halfedges_.erase(it);
                count++;
            } 
        } 
    }

    return c->halfedge();
}

 /*   halfedge_t* insert_edge(halfedge_t* v1, halfedge_t* v2) {
        // AGGIUNGO CONTROLLO ULTERIORE  DENTRO ALL'IF per essere sicura che v1,v2 e v2->next() non siano null
        if (v1->cell() && v2-> cell() && v1->cell()!=v2->cell()){
            std::cout << "Errore: i due half-edge appartengono a celle diverse. Non è possibile inserire un edge tra i due" << std::endl;
            return nullptr;
        }
        if (v1 && v2 && v2->next() && v1->next() && ( v1->node() == v2->next()->node() || v2->node()==v1->next()->node()) ) {
            std::cout << "Halfedges consecutivi" << std::endl;
            return v1;   // v1 and v2 are next halfedges
        }
        // get exiting halfedges from n1 and n2
        node_t* n1 = v1->node();
        node_t* n2 = v2->node();
        // create a pair of twin half-edges
        halfedge_t* h1 = emplace_halfedge_(n1);
        halfedge_t* h2 = emplace_halfedge_(n2);
        h1->set_twin(h2);
        h2->set_twin(h1);
	    // insert halfedge h1 between v1 and v1->prev
        h2->set_next(v1);
        // AGGIUNGO CONTROLLO SU v1->prev()
        if (v1->prev()){
            v1->prev()->set_next(h2->twin());
	        h2->twin()->set_prev(v1->prev());
            v1->set_prev(h2);
        }
        else{
            h1->set_prev(h2);
            h2->set_next(h1);
        }
        h2->next()->set_prev(h2);
        h2->set_node(n2);
	    // insert halfedge h2 between v2 and v2->prev
        h1->set_next(v2);
        // AGGIUNGO CONTROLLO SU v2->prev()
        if (v2->prev()) {
            v2->prev()->set_next(h1->twin());
	        h1->twin()->set_prev(v2->prev());
            v2->set_prev(h1);
        }
        else{
            h2->set_prev(h1);
            h1->set_next(h2);
        }
        h1->next()->set_prev(h1);
        h1->set_node(n1);
	    // set cell pointers
        // create new cell
        // SOLO SE L'EDGE NUOVO HA PREV E NEXT DIVERSI DA Sè STESSO
        if(h1->next() != h2 && h1->prev() != h2){
            //h1->set_cell(v1->cell());  //POTENZIALI PROBLEMI
            h1->set_cell(h1->prev()->cell());
            h1->cell()->set_halfedge(h1);
            cells_.push_back(cell_t(n_cells_++));
            cell_t* c1 = std::addressof(cells_.back());
            c1->set_halfedge(h2);
            halfedge_t* end = h2;
            do {         
              h2->set_cell(c1);   
              h2 = h2->next();
            } while (h2 != end );
        }
        else {
            if(h1->next()!=h2)
                h1->set_cell(h1->next()->cell());
            else
                h1->set_cell(h1->prev()->cell());
            h2->set_cell(h1->cell());
        }
	    return h1;
    }
    */


    /*
   // delaunay.print_dcel();
    delaunay.dcel().add_polygon(find_halfedge_from_id(0),internal.row(0));
    
    delaunay.dcel().add_polygon(find_halfedge_from_id(1),internal.row(0));
    //delaunay.print_dcel();
    delaunay.dcel().add_polygon(find_halfedge_from_id(2),internal.row(0));
    //delaunay.print_dcel();
    delaunay.dcel().add_polygon(find_halfedge_from_id(3),internal.row(0));

    Eigen::Matrix<double, 2, 2> prova;
    prova << 1.5, 0.5,
             0.5, 0.5;
    delaunay.dcel().add_polygon(find_halfedge_from_id(3),prova);
  
    delaunay.print_dcel();*/

        /*DCEL<2,2>::coords_t u = internal.row(1);
    std::cout << "🔍 Inserimento del punto interno: " << u.transpose()<< std::endl;
    
    const DCEL<2,2>::cell_t* triangle = delaunay.find_triangle(u);
    
    delaunay.insert_vertex(u,triangle);

    DCEL<2,2>::coords_t u_ = internal.row(2);
    std::cout << "🔍 Inserimento del punto interno: " << u_.transpose()<< std::endl;
    
    const DCEL<2,2>::cell_t* triangle_ = delaunay.find_triangle(u_);
    
    delaunay.insert_vertex(u_,triangle_);*/




    //REMOVE_EDGE  parte in più su boundary
    /*
        if (v1->on_boundary()) { // v1 is on the boundary
        //2 cases: 
        //a. v1 and its next/prev are on boundary --> elongate v1->next or prev (SE NO NON PERMETTERE)
        if(v1->next()->on_boundary() || v1->prev()->on_boundary()){
            v1->next()->set_prev(v1->prev());
            v1->prev()->set_next(v1->next());
            v1->next()->set_node(v1->node());     
            halfedge_t* v2= v1->twin();
            v2->next()->set_prev(v2->prev());
            v2->prev()->set_next(v2->next());
            v2->next()->set_node(v2->node());
            // remove edges, cell is the same so it doesn't need to be removed
            halfedge_t* next= v1->next();
            n_halfedges_ -= 2;
            auto it1 = std::find(halfedges_begin(), halfedges_end(), v1);
            auto it2 = std::find(halfedges_begin(), halfedges_end(), v2);
            halfedges_.erase(it1);
            halfedges_.erase(it2); 
            return next;
        }
        //b. only v1 is on boundary and its next/prev isn't
        else{
            halfedge_t* v2= v1->twin();
            v1->next()->set_prev(v2->prev());
            v2->prev()->set_next(v1->next());    
            v1->prev()->set_next(v2->next());
            v2->next()->set_prev(v1->prev());
            // remove edges and cell
            n_cells_--;
            auto it = std::find(cells_begin(), cells_end(), v1->cell());
            halfedge_t* end = v1;
            halfedge_t* begin = v1->next();
            do {
               begin->set_cell(nullptr);
               begin->node()->set_on_boundary(true);
               begin = begin->next();
            } while (begin!=end);
            cells_.erase(it);
            halfedge_t* next= v1->next();
            n_halfedges_ -= 2;
            auto it1 = std::find(halfedges_begin(), halfedges_end(), v1);
            auto it2 = std::find(halfedges_begin(), halfedges_end(), v2);
            halfedges_.erase(it1);
            halfedges_.erase(it2); 
            return next; 
        }
        } */



            

    bool is_point_inside_triangle(const coords_t& P, const coords_t& A, const coords_t& B, const coords_t& C) const{
        // baricenter test

        Eigen::Matrix<double, LocalDim, LocalDim> M;
        M << (B - A), (C - A); 
        coords_t lambda = M.inverse() * (P - A);
        double lambda1 = lambda.x();
        double lambda2 = lambda.y();
        double lambda3 = 1 - lambda1 - lambda2;
        return (lambda1 >= 0 && lambda1<=1 && lambda2 >= 0 && lambda2<=1 && lambda3 >= 0 && lambda3<=1);  
    }



    bool in_circle(const coords_t& A, const coords_t& B, const coords_t& C, const coords_t& D) const {
  
    
        // Costruzione della matrice 3x3 
        Eigen::Matrix3d M;
        M << (A.x() - D.x()), (A.y() - D.y()), (A.x() - D.x()) * (A.x() - D.x()) + (A.y() - D.y()) * (A.y() - D.y()),
             (B.x() - D.x()), (B.y() - D.y()), (B.x() - D.x()) * (B.x() - D.x()) + (B.y() - D.y()) * (B.y() - D.y()),
             (C.x() - D.x()), (C.y() - D.y()), (C.x() - D.x()) * (C.x() - D.x()) + (C.y() - D.y()) * (C.y() - D.y());
    
        double det = M.determinant();
        
        std::cout << "Determinante InCircle: " << det << " -> " << (det > 0 ? "Dentro" : "Fuori") << std::endl;
        
        return det > 0;
    }



    
    bool is_counterclockwise(const coords_t& a, const coords_t& b, const coords_t& c) {
        double det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        return det > 0;  // true se in ordine anticlockwise
    }
    
    

    bool is_point_on_edge(const coords_t& P, const coords_t& A, const coords_t& B, double tol) {
        double cross = (P.y() - A.y()) * (B.x() - A.x()) - (P.x() - A.x()) * (B.y() - A.y());
        if (std::abs(cross) > tol) return false; // Non è collineare
    
        double dot = (P.x() - A.x()) * (B.x() - A.x()) + (P.y() - A.y()) * (B.y() - A.y());
        if (dot < 0) return false; // Punto fuori dal segmento
    
        double len_sq = (B.x() - A.x()) * (B.x() - A.x()) + (B.y() - A.y()) * (B.y() - A.y());
        if (dot > len_sq) return false; // Punto fuori dal segmento
    
        return true; 
    }
    

     /*  
    void initialize_triangulation() {
        std::cout << "🔷 Inizializzazione della triangolazione...\n";
    
        if (internal_points_.rows() == 0) {
            std::cerr << "❌ Errore: Nessun punto interno disponibile per inizializzare la triangolazione!\n";
            return;
        }
    
        Eigen::Matrix<double, 1, embed_dim> first_internal = internal_points_.row(0);
        auto it = dcel_.halfedges_begin();
        for (int i = 0; i < boundary_points_.rows(); ++i, ++it) {
            halfedge_t* he = &(*it);
            dcel_.add_polygon(he, first_internal);
        }
   
    int node_offset = boundary_points_.rows(); 
    for (const auto& hole : hole_points_) {
        if (hole.rows() == 0) continue; 
        
        node_t* first_hole_node = std::addressof(*std::next(dcel_.nodes_begin(), node_offset)); 
        halfedge_t* first_hole_he = first_hole_node->halfedge();
        
        std::vector<halfedge_t*> hole_edges;
        halfedge_t* he = first_hole_he;
        do {
            hole_edges.push_back(he);
            he = he->next();
        } while (he != first_hole_he);  

        
        for (halfedge_t* he : hole_edges) {
            dcel_.add_polygon(he, first_internal);
        }

        node_offset += hole.rows(); 
    }

    std::cout << "✅ Triangolazione iniziale completata!\n";
}

*/


/*
    void build_triangulation() {
        std::cout << "🔷 Inizio costruzione della triangolazione...\n";
        
        for (int i = 1; i < internal_points_.rows(); ++i) {
            coords_t u = internal_points_.row(i);
            std::cout << "🔍 Inserimento del punto interno: " << u.transpose() << std::endl;
            
            const cell_t* triangle = find_triangle(u);
            
            if (!triangle) {
                std::cerr << "❌ Errore: Nessun triangolo trovato per il punto " << u.transpose() << "!" << std::endl;
                continue;
            }
            
            insert_vertex(u, triangle);
        }
        int cont = 0;
        for(auto it = dcel_.cells_begin();it!=dcel_.cells_end();++it){
            it->set_id(cont);
            cont++;
        }

        int cont_h = 0;
        for(auto it = dcel_.halfedges_begin();it!=dcel_.halfedges_end();++it){
            it->set_id(cont_h);
            cont_h++;
        }
       
        std::cout << "✅ Triangolazione completata con successo!\n";
    }
*/



void insert_vertex(const coords_t& u, const cell_t* triangle) {
        
    halfedge_t* vw = triangle->halfedge();
    halfedge_t* wx = vw->next();
    halfedge_t* xv = vw->prev();

    std::cout << "Half-edges del triangolo: " << std::endl;
    std::cout << " - vw: " << vw->id() << " | wx: " << wx->id() << " | xv: " << xv->id() << std::endl;
    
    // expanding cavity
    dig_cavity(u, vw);
    dig_cavity(u, wx);
    dig_cavity(u, xv);
}



void dig_cavity(const coords_t& u, halfedge_t* vw) { 
    if(vw->on_boundary()){
        std::cout<<"SONO AL BORDO CON : "<<vw->id()<<std::endl;
        dcel_.add_polygon(vw,u.transpose());
        return;
    }

    node_t* x = dcel_.adjacent(vw);
    if (!x) {
        std::cout << "Half-edge " << vw->id() << " non ha un nodo adiacente.\n";
        return;
    }
    std::cout << "Nodo adiacente a half-edge " << vw->id() << ": " << x->id() << "\n";
    
    bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(u, vw->node()->coords(), vw->twin()->node()->coords());

    bool inside;
    if (ccw) {
        inside = fdapde::internals::in_circle(u, vw->node()->coords(), vw->twin()->node()->coords(), x->coords());
    } else {
        inside = fdapde::internals::in_circle(u, vw->twin()->node()->coords(), vw->node()->coords(), x->coords());
    }

    if (inside) { 
        std::cout << "Punto " << x->id() << " è dentro il circumcerchio";
        
        halfedge_t* wv = vw->twin();
        halfedge_t* vx = vw->twin()->next();
        halfedge_t* xw = vw->twin()->prev(); 

        std::cout << "Half-edges del triangolo: " << std::endl;
        std::cout << " - wv: " << wv->id() << " | vx: " << vx->id() << " | xw: " << xw->id() << std::endl;

        dcel_.remove_edge(vw);
        dig_cavity(u, vx);
        dig_cavity(u, xw);
       
        std::cout<<"REMOVE: "<<vw->twin()->id()<<std::endl;
    } else {
        std::cout << "Punto " << x->id() << " NON è dentro il circumcerchio. Aggiungo nuovo triangolo.\n";
        dcel_.add_polygon(vw,u.transpose());
        return;
    } 
}



delaunay.add_first_triangle(find_halfedge_from_id(0), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(1), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(2), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(3), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(4), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(5), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(6), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(7), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(8), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(9), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(10), internal.row(0));
delaunay.add_first_triangle(find_halfedge_from_id(11), internal.row(0));



void print_dcel() {
    std::cout << "==============================" << std::endl;
    std::cout << "📌 STATO ATTUALE DELLA DCEL 📌" << std::endl;
    std::cout << "==============================" << std::endl;

    // 📍 Stampa tutti i nodi
    std::cout << "\n🟢 NODI: \n";
    for (auto it = dcel_.nodes_begin(); it != dcel_.nodes_end(); ++it) {
        std::cout << "ID: " << it->id() << " | Coords: (" << it->coords()(0) << ", " << it->coords()(1) << ")"
                  << (it->on_boundary() ? " [BOUNDARY]" : "") << std::endl;
    }

    // 🔗 Stampa tutti gli Half-Edges
    std::cout << "\n🔵 HALF-EDGES: \n";
    for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
        std::cout << "ID: " << it->id()
                  << " | Nodo Origine: " << (it->node() ? std::to_string(it->node()->id()) : "NULL")
                  << " | Twin: " << (it->twin() ? std::to_string(it->twin()->id()) : "NULL")
                  << " | Next: " << (it->next() ? std::to_string(it->next()->id()) : "NULL")
                  << " | Prev: " << (it->prev() ? std::to_string(it->prev()->id()) : "NULL")
                  << std::endl;
    }

    // 🔳 Stampa tutte le Celle
    std::cout << "\n🟠 CELLE: \n";
    for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
        std::cout << "Cella ID: " << it->id() << " | Half-edge di riferimento: "
                  << (it->halfedge() ? std::to_string(it->halfedge()->id()) : "NULL") << std::endl;
        if (it->halfedge()) {
            halfedge_t* h = it->halfedge();
            std::cout << "  🔗 Half-edges nella cella: ";
            halfedge_t* start = h;
            do {
                std::cout << h->id() << " ";
                h = h->next();
            } while (h && h != start);
            std::cout << std::endl;
        }
    }

    std::cout << "==============================\n" << std::endl;
}



void dig_cavity(const coords_t& u, halfedge_t* vw) { 
    //if we are on the boundary we add the triangle  
        if(vw->on_boundary()){
            add_triangle(vw,u.transpose());
            return;
        }
    //finding the point adjacent to vw
        node_t* x = dcel_.adjacent(vw);
        if (!x) {
            return;
        }
        bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(u, vw->node()->coords(), vw->twin()->node()->coords());
    //test of circumcircle   
        bool inside;
        if (ccw) {
            inside = fdapde::internals::in_circle(u, vw->node()->coords(), vw->twin()->node()->coords(), x->coords());
        } else {
            inside = fdapde::internals::in_circle(u, vw->twin()->node()->coords(), vw->node()->coords(), x->coords());
        }
        if (inside) { 
        //falied the test so remove the triangle and expand the cavity on the remaining edges
            halfedge_t* wv = vw->twin();
            halfedge_t* vx = vw->twin()->next();
            halfedge_t* xw = vw->twin()->prev(); 
            
            dcel_.remove_edge(vw);
            dig_cavity(u, vx);
            dig_cavity(u, xw);
        } else {
        //passed the test,adding the triangle 
            add_triangle(vw,u.transpose());
            return;
        } 
    }
    




    void insert_vertex(const coords_t& u, const cell_t* triangle) {
        
        halfedge_t* vw = triangle->halfedge();
        halfedge_t* wx = vw->next();
        halfedge_t* xv = vw->prev();
        // expanding cavity
        dig_cavity(u, vw);
        dig_cavity(u, wx);
        dig_cavity(u, xv);
    }


    //overloaded one if user wants to impose internal points manually 
//if one point exceeds the domain find_triangle return nullptr and does not enter in the triangulation
void build_triangulation() {

    if (internal_points_.empty()) {
        return;
    }
    //inserting fist node in the domain and creating all the triangles from the boundary edges
    coords_t first_internal = internal_points_.front();
    auto it = dcel_.halfedges_begin();
    for (int i = 0; i < boundary_points_.rows(); ++i, ++it) {
        halfedge_t* he = &(*it);
        add_first_triangle(he, first_internal.transpose());
    }
/*
    int node_offset = boundary_points_.rows(); 
    for (const auto& hole : hole_points_) {
        if (hole.rows() == 0) continue; 
        
        node_t* first_hole_node = std::addressof(*std::next(dcel_.nodes_begin(), node_offset)); 
        halfedge_t* first_hole_he = first_hole_node->halfedge();
        
        std::vector<halfedge_t*> hole_edges;
        halfedge_t* he = first_hole_he;
        do {
            hole_edges.push_back(he);
            he = he->next();
        } while (he != first_hole_he);  

        for (halfedge_t* he : hole_edges) {
            add_first_triangle(he, first_internal);
        }

        node_offset += hole.rows(); 
    }*/

// flip the initial trinagulation if not Delaunay 
   flip();
//inserting the num_points-1 inner points in the domain
    for (size_t i = 1; i < internal_points_.size(); ++i) {
        coords_t u = internal_points_[i];
        const cell_t* triangle = find_triangle(u);

        if (!triangle) {
            continue;
        }
        insert_vertex(u, triangle);
    }
//reordering id of cells and halfedges to cover some jumps between ids after removing
    int cont = 0;
    for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
        it->set_id(cont);
        cont++;
    }

    dcel_.set_n_cells_(cont);

    int cont_h = 0;
    for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
        it->set_id(cont_h);
        cont_h++;
    }

    Triangulation<local_dim, embed_dim> triangulation = DCEL_to_Triangulation();

    std::string filename = "mesh_output.txt";
    export_triangulation_to_txt(triangulation, filename);

    std::string command = "python3 fdaPDE/src/plot_mesh.py";
    std::system(command.c_str()); 
}



void dig_cavity(const coords_t& u, halfedge_t* vw) { 
    //if we are on the boundary we add the triangle  
        if(vw->on_boundary()){
            add_triangle(vw,u.transpose());
            return;
        }
    //finding the point adjacent to vw
        node_t* x = dcel_.adjacent(vw);
        if (!x) {
            return;
        }
        bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(u, vw->node()->coords(), vw->twin()->node()->coords());
    //test of circumcircle   
        bool inside;
        if (ccw) {
            inside = fdapde::internals::in_circle(u, vw->node()->coords(), vw->twin()->node()->coords(), x->coords());
        } else {
            inside = fdapde::internals::in_circle(u, vw->twin()->node()->coords(), vw->node()->coords(), x->coords());
        }
        if (inside) { 
        //falied the test so remove the triangle and expand the cavity on the remaining edges
            halfedge_t* wv = vw->twin();
            halfedge_t* vx = vw->twin()->next();
            halfedge_t* xw = vw->twin()->prev(); 
            
            dcel_.remove_edge(vw);
            dig_cavity(u, vx);
            dig_cavity(u, xw);
        } else {
        //passed the test,adding the triangle 
            add_triangle(vw,u.transpose());
            return;
        } 
    }






    const cell_t* find_triangle(const coords_t& P) {
        
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            cell_t* cell = &(*it);  
    
            const coords_t& A = cell->halfedge()->node()->coords();
            const coords_t& B = cell->halfedge()->next()->node()->coords();
            const coords_t& C = cell->halfedge()->prev()->node()->coords();

            // Checking if one point is on the edge
            if (fdapde::internals::contains(P, A, B) || 
            fdapde::internals::contains(P, B, C) || 
            fdapde::internals::contains(P, C, A)) {
            //I remove from internal_points vector
                auto it = std::remove_if(internal_points_.begin(), internal_points_.end(), 
                [&](const coords_t& point) { return point.isApprox(P); });
                internal_points_.erase(it, internal_points_.end());

                return nullptr;
            }
    
            if (fdapde::internals::point_in_2d_tri(P, A, B, C)) {
                return cell;
            }
        }

    return nullptr;
    }




    void build_triangulation(int num_points) {

        double min_x = boundary_points_.col(0).minCoeff();
        double max_x = boundary_points_.col(0).maxCoeff();
        double min_y = boundary_points_.col(1).minCoeff();
        double max_y = boundary_points_.col(1).maxCoeff();
    
        //creating the generator of casual points for the internal_points
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist_x(min_x, max_x);
        std::uniform_real_distribution<double> dist_y(min_y, max_y);

        //generating internal_points 
        internal_points_.clear();
        internal_points_.reserve(num_points);

        while (static_cast<int>(internal_points_.size()) < num_points) {
            coords_t u;
            u << dist_x(gen), dist_y(gen);
            std::cout << "nuovo punto interno: " << u.transpose() << std::endl;
            //verifies if the point is inside the domain (in order to control the concavities)
            //and discard the points falling on the boundary of the domain
            if (fdapde::internals::point_in_polygon(boundary_points_, u)){
            /* {
                bool in_hole = false;
                for (const auto& hole : hole_points_) {
                    if (fdapde::internals::point_in_polygon(hole, u)) {
                        in_hole = true;
                        break;
                    }
                }
                if (!in_hole) {*/
                std::cout << "nuovo punto inserito: " << u.transpose() << std::endl;
                    internal_points_.push_back(u);}
              //  }
          //  }
        }
    
        if (internal_points_.empty()) {
            return;
        }
    
        //inserting fist node in the domain and creating all the triangles from the boundary edges
        coords_t first_internal = internal_points_.front();
        
        auto it = dcel_.halfedges_begin();
        for (int i = 0; i < boundary_points_.rows(); ++i, ++it) {
            halfedge_t* he = &(*it);
            add_first_triangle(he, first_internal.transpose());
        }
/*
        int node_offset = boundary_points_.rows(); 
        for (const auto& hole : hole_points_) {
            if (hole.rows() == 0) continue; 
            
            node_t* first_hole_node = std::addressof(*std::next(dcel_.nodes_begin(), node_offset)); 
            halfedge_t* first_hole_he = first_hole_node->halfedge();
            
            std::vector<halfedge_t*> hole_edges;
            halfedge_t* he = first_hole_he;
            do {
                hole_edges.push_back(he);
                he = he->next();
            } while (he != first_hole_he);  

            for (halfedge_t* he : hole_edges) {
                add_first_triangle(he, first_internal);
            }

            node_offset += hole.rows(); 
        }
*/
        flip();
    //inserting the num_points-1 inner points in the domain
        for (size_t i = 1; i < internal_points_.size(); ++i) {
            coords_t u = internal_points_[i];
            const cell_t* triangle = find_triangle(u);

            if (!triangle) {
                continue;
            }
            insert_vertex(u, triangle);
        }
    //reordering id of cells and halfedges to cover some jumps between ids after removing
        int cont = 0;
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            it->set_id(cont);
            cont++;
        }
        dcel_.set_n_cells_(cont);

        int cont_h = 0;
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
            it->set_id(cont_h);
            cont_h++;
        }

        Triangulation<local_dim, embed_dim> triangulation = DCEL_to_Triangulation();

        std::string filename = "mesh_output.txt";
        export_triangulation_to_txt(triangulation, filename);

        std::string command = "python3 fdaPDE/src/plot_mesh.py";
        std::system(command.c_str()); 
    }



    //overloaded one if user wants to impose internal points manually 
//if one point exceeds the domain find_triangle return nullptr and does not enter in the triangulation
void build_triangulation() {

    if (internal_points_.empty()) {
        return;
    }
    //inserting fist node in the domain and creating all the triangles from the boundary edges
    coords_t first_internal = internal_points_.front();
    auto it = dcel_.halfedges_begin();
    add_first_triangle(first_internal.transpose());
    
/*
    int node_offset = boundary_points_.rows(); 
    for (const auto& hole : hole_points_) {
        if (hole.rows() == 0) continue; 
        
        node_t* first_hole_node = std::addressof(*std::next(dcel_.nodes_begin(), node_offset)); 
        halfedge_t* first_hole_he = first_hole_node->halfedge();
        
        std::vector<halfedge_t*> hole_edges;
        halfedge_t* he = first_hole_he;
        do {
            hole_edges.push_back(he);
            he = he->next();
        } while (he != first_hole_he);  

        for (halfedge_t* he : hole_edges) {
            add_first_triangle(he, first_internal);
        }

        node_offset += hole.rows(); 
    }*/

// flip the initial trinagulation if not Delaunay 
   //flip();
//inserting the num_points-1 inner points in the domain
    for (size_t i = 1; i < internal_points_.size(); ++i) {
        coords_t u = internal_points_[i];
        const cell_t* triangle = find_triangle(u);

        if (!triangle) {
            continue;
        }
        insert_vertex(u, triangle);
    }
//reordering id of cells and halfedges to cover some jumps between ids after removing
    int cont = 0;
    for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
        it->set_id(cont);
        cont++;
    }

    dcel_.set_n_cells_(cont);

    int cont_h = 0;
    for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
        it->set_id(cont_h);
        cont_h++;
    }

    Triangulation<local_dim, embed_dim> triangulation = DCEL_to_Triangulation();

    std::string filename = "mesh_output.txt";
    export_triangulation_to_txt(triangulation, filename);

    std::string command = "python3 fdaPDE/src/plot_mesh.py";
    std::system(command.c_str()); 
}




void print_dcel() {
    std::cout << "==============================" << std::endl;
    std::cout << "📌 STATO ATTUALE DELLA DCEL 📌" << std::endl;
    std::cout << "==============================" << std::endl;

    // 📍 Stampa tutti i nodi
    std::cout << "\n🟢 NODI: \n";
    for (auto it = dcel_.nodes_begin(); it != dcel_.nodes_end(); ++it) {
        std::cout << "ID: " << it->id() << " | Coords: (" << it->coords()(0) << ", " << it->coords()(1) << ")"
                  << (it->on_boundary() ? " [BOUNDARY]" : "") << std::endl;
    }

    // 🔗 Stampa tutti gli Half-Edges
    std::cout << "\n🔵 HALF-EDGES: \n";
    for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
        std::cout << "ID: " << it->id()
                  << " | Nodo Origine: " << (it->node() ? std::to_string(it->node()->id()) : "NULL")
                  << " | Twin: " << (it->twin() ? std::to_string(it->twin()->id()) : "NULL")
                  << " | Next: " << (it->next() ? std::to_string(it->next()->id()) : "NULL")
                  << " | Prev: " << (it->prev() ? std::to_string(it->prev()->id()) : "NULL")
                  << std::endl;
    }

    // 🔳 Stampa tutte le Celle
    std::cout << "\n🟠 CELLE: \n";
    for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
        std::cout << "Cella ID: " << it->id() << " | Half-edge di riferimento: "
                  << (it->halfedge() ? std::to_string(it->halfedge()->id()) : "NULL") << std::endl;
        if (it->halfedge()) {
            halfedge_t* h = it->halfedge();
            std::cout << "  🔗 Half-edges nella cella: ";
            halfedge_t* start = h;
            do {
                std::cout << h->id() << " ";
                h = h->next();
            } while (h && h != start);
            std::cout << std::endl;
        }
    }

    std::cout << "==============================\n" << std::endl;
}




    const cell_t* find_triangle(const coords_t& P) {
        
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            cell_t* cell = &(*it);  
    
            const coords_t& A = cell->halfedge()->node()->coords();
            const coords_t& B = cell->halfedge()->next()->node()->coords();
            const coords_t& C = cell->halfedge()->prev()->node()->coords();

            // Checking if one point is on the edge
            if (fdapde::internals::contains(P, A, B) || 
            fdapde::internals::contains(P, B, C) || 
            fdapde::internals::contains(P, C, A)) {
            //I remove from internal_points vector
                auto it = std::remove_if(internal_points_.begin(), internal_points_.end(), 
                [&](const coords_t& point) { return point.isApprox(P); });
                internal_points_.erase(it, internal_points_.end());

                return nullptr;
            }
    
            if (fdapde::internals::point_in_2d_tri(P, A, B, C)) {
                return cell;
            }
        }

    return nullptr;
    }

    void dig_cavity(const coords_t& u, halfedge_t* vw) { 
    //if we are on the boundary we add the triangle  
        if(vw->on_boundary()){
            add_triangle(vw,u.transpose());
            return;
        }
    //finding the point adjacent to vw
        node_t* x = dcel_.adjacent(vw);
        if (!x) {
            return;
        }
        bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(u, vw->node()->coords(), vw->twin()->node()->coords());
    //test of circumcircle   
        bool inside;
        if (ccw) {
            inside = fdapde::internals::in_circle(u, vw->node()->coords(), vw->twin()->node()->coords(), x->coords());
        } else {
            inside = fdapde::internals::in_circle(u, vw->twin()->node()->coords(), vw->node()->coords(), x->coords());
        }
        if (inside) { 
        //falied the test so remove the triangle and expand the cavity on the remaining edges
            halfedge_t* wv = vw->twin();
            halfedge_t* vx = vw->twin()->next();
            halfedge_t* xw = vw->twin()->prev(); 
            
            dcel_.remove_edge(vw);
            dig_cavity(u, vx);
            dig_cavity(u, xw);
        } else {
        //passed the test,adding the triangle 
            add_triangle(vw,u.transpose());
            return;
        } 
    }
    
    
    void insert_vertex(const coords_t& u, const cell_t* triangle) {
        
        halfedge_t* vw = triangle->halfedge();
        halfedge_t* wx = vw->next();
        halfedge_t* xv = vw->prev();
        // expanding cavity
        dig_cavity(u, vw);
        dig_cavity(u, wx);
        dig_cavity(u, xv);
    }


    void build_triangulation(int num_points) {

        double min_x = boundary_points_.col(0).minCoeff();
        double max_x = boundary_points_.col(0).maxCoeff();
        double min_y = boundary_points_.col(1).minCoeff();
        double max_y = boundary_points_.col(1).maxCoeff();
    
        //creating the generator of casual points for the internal_points
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist_x(min_x, max_x);
        std::uniform_real_distribution<double> dist_y(min_y, max_y);

        //generating internal_points 
        internal_points_.clear();
        internal_points_.reserve(num_points);

        while (static_cast<int>(internal_points_.size()) < num_points) {
            coords_t u;
            u << dist_x(gen), dist_y(gen);
            std::cout << "nuovo punto interno: " << u.transpose() << std::endl;
            //verifies if the point is inside the domain (in order to control the concavities)
            //and discard the points falling on the boundary of the domain
            if (fdapde::internals::point_in_polygon(boundary_points_, u)){
            /* {
                bool in_hole = false;
                for (const auto& hole : hole_points_) {
                    if (fdapde::internals::point_in_polygon(hole, u)) {
                        in_hole = true;
                        break;
                    }
                }
                if (!in_hole) {*/
                std::cout << "nuovo punto inserito: " << u.transpose() << std::endl;
                    internal_points_.push_back(u);}
              //  }
          //  }
        }
    
        if (internal_points_.empty()) {
            return;
        }
    
        //inserting fist node in the domain and creating all the triangles from the boundary edges
        coords_t first_internal = internal_points_.front();
        
        auto it = dcel_.halfedges_begin();
        add_first_triangle( first_internal.transpose());
/*
        int node_offset = boundary_points_.rows(); 
        for (const auto& hole : hole_points_) {
            if (hole.rows() == 0) continue; 
            
            node_t* first_hole_node = std::addressof(*std::next(dcel_.nodes_begin(), node_offset)); 
            halfedge_t* first_hole_he = first_hole_node->halfedge();
            
            std::vector<halfedge_t*> hole_edges;
            halfedge_t* he = first_hole_he;
            do {
                hole_edges.push_back(he);
                he = he->next();
            } while (he != first_hole_he);  

            for (halfedge_t* he : hole_edges) {
                add_first_triangle(he, first_internal);
            }

            node_offset += hole.rows(); 
        }
*/
        flip();
    //inserting the num_points-1 inner points in the domain
        for (size_t i = 1; i < internal_points_.size(); ++i) {
            coords_t u = internal_points_[i];
            const cell_t* triangle = find_triangle(u);

            if (!triangle) {
                continue;
            }
            insert_vertex(u, triangle);
        }
    //reordering id of cells and halfedges to cover some jumps between ids after removing
        int cont = 0;
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            it->set_id(cont);
            cont++;
        }
        dcel_.set_n_cells_(cont);

        int cont_h = 0;
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
            it->set_id(cont_h);
            cont_h++;
        }

        Triangulation<local_dim, embed_dim> triangulation = DCEL_to_Triangulation();

        std::string filename = "mesh_output.txt";
        export_triangulation_to_txt(triangulation, filename);

        std::string command = "python3 fdaPDE/src/plot_mesh.py";
        std::system(command.c_str()); 
    }






    
                // Checking if one point is on the edge
                if (fdapde::internals::contains(y, t1, t2) || 
                fdapde::internals::contains(y, t2, t3) || 
                fdapde::internals::contains(y, t3, t1)) {
                    //I remove from internal_points vector
                    auto it = std::remove_if(internal_points_.begin(), internal_points_.end(), 
                    [&](const coords_t& point) { return point.isApprox(y); });
                    internal_points_.erase(it, internal_points_.end());
                    continue;
                }



                            // Debug: Verifica dei conflitti triangolo -> nodi
        std::cout << "=== Conflitti Triangolo -> Nodi ===" << std::endl;
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            cell_t* t = &(*it);
            if (!t) continue;
            auto& conflict_list = t->conflicting_points();
            std::cout << "Triangolo " << t->id() << " ha conflitti con nodi: ";
            for (node_t* point : conflict_list) {
                std::cout << point->id() << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "=== Conflitti Nodi -> Triangoli ===" << std::endl;
        for (auto it = dcel_.nodes_begin(); it != dcel_.nodes_end(); ++it) {
            node_t* n = &(*it);
            if (!n) continue;
            if(!n->on_boundary() && n->conflict()){
            cell_t* conflict = n->conflict();
            std::cout << "Nodo " << n->id() << " appartiene a : "<<conflict->id() << std::endl;}
        //  std::cout << "Nodo " << n->id() << "   "<<n->is_valid_conflict()<<std::endl;
        }



            // Function to insert a vertex handling conflicts
    void insert_vertex_at_conflict(node_t* u) {
        // Retrieve the triangle in conflict with u and we marked as visited 
        cell_t* t = u->conflict(); 
        std::vector<halfedge_t*> D;
        std::vector<halfedge_t*> C;

        mark_cavity(u, t->halfedge(), D, C);
        mark_cavity(u, t->halfedge()->prev(), D, C);
        mark_cavity(u, t->halfedge()->next(), D, C);
        
        //passing the conflicts of cthe cavity to a temporary vector 
        std::vector<node_t*> conflict_points_temp; 
        for (halfedge_t* h : D) { 
            cell_t* current_cell = h->cell();
            if (current_cell) {
                auto& conflict_list = current_cell->conflicting_points();
                for (node_t* point : conflict_list) {
                    if (point != u) { 
                        conflict_points_temp.push_back(point);
                        // checking if the point is in the cavity or is in a cell that does not belong to the cavity 
                        if (point->conflict() == current_cell) {
                            point->set_valid_conflict(false); // Invalidating the conflict 
                        }
                    }
                }
                current_cell->clear_conflicts();
            }


            cell_t* twin_cell = h->twin()->cell();
            if (twin_cell) {
                auto& conflict_list = twin_cell->conflicting_points();
                for (node_t* point : conflict_list) {
                    if (point != u) { 
                        conflict_points_temp.push_back(point);
                        // checking if the point is in the cavity or is in a cell that does not belong to the cavity 
                        if (point->conflict() == twin_cell) {
                            point->set_valid_conflict(false); // Invalidating the conflict 
                        }
                    }
                }
                twin_cell->clear_conflicts();
            }
        }
        if(D.empty()){
            cell_t* current_cell = u->conflict();
            if (current_cell) {
                auto& conflict_list = current_cell->conflicting_points();
                for (node_t* point : conflict_list) {
                    if (point != u) { 
                        conflict_points_temp.push_back(point);
                        // checking if the point is in the cavity or is in a cell that does not belong to the cavity 
                        if (point->conflict() == current_cell) {
                            point->set_valid_conflict(false); // Invalidating the conflict 
                        }
                    }
                }
                current_cell->clear_conflicts();
            }
        }
        u->remove_conflict();
        
        // removing cells of the cavity 
        for (halfedge_t* h : D) { 
            //std::cout <<"CIAO"<< h->id() << std::endl;
            dcel_.remove_edge(h);
        }
        // creating the new cells 
        for (halfedge_t* h : C) { 
          //  std::cout << h->id() << std::endl;
            add_triangle(h, u->coords().transpose());
        }
        
        // reassigning the conflicts to the new cells 
        for (node_t* y : conflict_points_temp) {
            bool found = false;
            for (halfedge_t* h : C) { // Ciclyng on the new cells
                cell_t* t = h->cell();
           // std::cout<<"TRIANGOLO CON ID: "<<h->cell()->id()<<std::endl;
                if (!t) continue;
                const coords_t& t1 = t->halfedge()->prev()->node()->coords();
                const coords_t& t2 = t->halfedge()->node()->coords();
                const coords_t& t3 = t->halfedge()->next()->node()->coords();
    
                bool ccw = fdapde::internals::are_2d_counterclockwise_sorted(t1, t2, t3);
                // Test 1: Verifing if the point is inside the triangle
                if (!y->is_valid_conflict() && !found){
                    bool inside_triangle = ccw ? fdapde::internals::point_in_2d_tri(y->coords(), t1, t2, t3)
                                            : fdapde::internals::point_in_2d_tri(y->coords(), t3, t2, t1);
                    // Assigning principal conflict                            
                    if (inside_triangle) {
                    y->set_conflict(t);
                    y->set_valid_conflict(true);
                    found = true;
                    } 
                } 
                // Test 2: Verifing if the point is in the circumcircle 
                bool inside_circumcircle = ccw ? fdapde::internals::in_circle(t1, t2, t3, y->coords())
                                                : fdapde::internals::in_circle(t3, t2, t1, y->coords());
                // adding n to list of conflict with t
                if (inside_circumcircle) t->add_conflict(y);  
            }
        }
    }  









    
    void build_triangulation(int N, const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary) {

        // Computing the bounding box
        double min_x = boundary.col(0).minCoeff();
        double max_x = boundary.col(0).maxCoeff();
        double min_y = boundary.col(1).minCoeff();
        double max_y = boundary.col(1).maxCoeff();
    
        // Creating the generator of causal numbers
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist_x(min_x, max_x);
        std::uniform_real_distribution<double> dist_y(min_y, max_y);
   
        int first_internal_id = -1;
        int generated_points = 0;

        while (generated_points < N) {
            coords_t u;
            u << dist_x(gen), dist_y(gen);
            // Verifiyng if the point is inside the polygon, in order to coorecty dale with concavities
            if (!fdapde::internals::point_in_polygon(boundary, u)) 
                continue; 
            if (generated_points == 0) {
                // Initialize triangulation with the first valid point
                add_first_triangle(u.transpose(), boundary);
                first_internal_id = dcel_.n_nodes() - 1;
            } else {
                // Create the node and detect the conflicts with existing cells 
                node_t* n = dcel_.insert_node(node_t(dcel_.n_nodes(), false, u));
                detect_conflicts(n); 
            }
            ++generated_points;
        }
        // inserting the remaining nodes in the domain 
        for (auto it = dcel_.nodes_begin(); it != dcel_.nodes_end(); ++it) {
            node_t* u = &(*it);
            if (!u->on_boundary() && u->id()!=first_internal_id)   
                insert_vertex_at_conflict(u); 
        }
         /*   
        //reordering id of cells and halfedges to cover some jumps between ids after removing
        int cont = 0;
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            it->set_id(cont);
            cont++;
        }

        dcel_.set_n_cells_(cont);

        int cont_h = 0;
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
            it->set_id(cont_h);
            cont_h++;
        }
        */
        /*mesh generation not included in order to test the efficiency of the triangulation
        Triangulation<local_dim, embed_dim> triangulation = DCEL_to_Triangulation();

        std::string filename = "mesh_output.txt";
        export_triangulation_to_txt(triangulation, filename);

        std::string command = "python3 fdaPDE/src/plot_mesh.py";
        std::system(command.c_str()); 
        */
    }




        // Function to insert a vertex handling conflicts
        void insert_vertex_at_conflict(node_t* u) {
            // Retrieve the triangle in conflict with u and we marked as visited 
            cell_t* t = u->conflict(); 
            std::vector<halfedge_t*> D;
            std::vector<halfedge_t*> C;
    
            mark_cavity(u, t->halfedge(), D, C);
            mark_cavity(u, t->halfedge()->prev(), D, C);
            mark_cavity(u, t->halfedge()->next(), D, C);
      
            //invalidating the conflicts node->cell for the point of the cavity
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
            //if all the new traingles are Delaunay and i am not expanding the cavity 
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
            
            // removing cells of the cavity 
            for (halfedge_t* h : D) { 
                dcel_.remove_edge(h);
            }
            // creating the new cells 
            for (halfedge_t* h : C) { 
                add_triangle(h, u->coords().transpose());
            }
            
            // Reassigning the conflicts to the new cells  
            for (node_t* y : invalidated_nodes) {
                detect_conflicts(y, C);
            }
        }  
    



        /*
    void insert_vertex_at_conflict(node_t* u) {
        auto t0 = high_resolution_clock::now(); // Start totale
    
        cell_t* t = u->conflict(); 
        std::vector<halfedge_t*> D;
        std::vector<halfedge_t*> C;
    
        auto t1 = high_resolution_clock::now(); // Start mark_cavity
    
        mark_cavity(u, t->halfedge(), D, C);
        mark_cavity(u, t->halfedge()->prev(), D, C);
        mark_cavity(u, t->halfedge()->next(), D, C);
    
        auto t2 = high_resolution_clock::now(); // End mark_cavity - start invalidazione
    
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
    
        if (D.empty()) {
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
    
        auto t3 = high_resolution_clock::now(); // End invalidazione - start rimozione
    
        for (halfedge_t* h : D) { 
            dcel_.remove_edge(h);
        }
    
        auto t4 = high_resolution_clock::now(); // End rimozione - start aggiunta
    
        for (halfedge_t* h : C) { 
            add_triangle(h, std::vector<node_t*> {u});
        }
    
        auto t5 = high_resolution_clock::now(); // End aggiunta - start ridistribuzione
    
        for (node_t* y : invalidated_nodes) {
            detect_conflicts(y, C);
        }
    
        auto t6 = high_resolution_clock::now(); // Fine totale
    
        // Timing breakdown
        auto cavity_time = duration_cast<microseconds>(t2 - t1).count();
        auto invalidate_time = duration_cast<microseconds>(t3 - t2).count();
        auto remove_time = duration_cast<microseconds>(t4 - t3).count();
        auto add_time = duration_cast<microseconds>(t5 - t4).count();
        auto redistribute_time = duration_cast<microseconds>(t6 - t5).count();
        auto total_time = duration_cast<microseconds>(t6 - t0).count();
    
        std::cout << "----- insert_vertex_at_conflict() breakdown -----\n";
        std::cout << "Mark cavity:        " << cavity_time        << " µs\n";
        std::cout << "Invalidate conf:    " << invalidate_time    << " µs\n";
        std::cout << "Remove triangles:   " << remove_time        << " µs\n";
        std::cout << "Add triangles:      " << add_time           << " µs\n";
        std::cout << "Redistribute conf:  " << redistribute_time  << " µs\n";
        std::cout << "TOTAL:              " << total_time         << " µs\n";
    }
*/





    //function to convert the dcel into a triangulation
    Triangulation<local_dim, embed_dim> DCEL_to_Triangulation() {  
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nodes(dcel_.n_nodes(), embed_dim);
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> cells(dcel_.n_cells(), n_nodes_cell);
        Eigen::Matrix<int, Eigen::Dynamic, 1> boundary_markers(dcel_.n_nodes());

        // Fill nodes matrix
        int node_idx = 0;
        for (auto it = dcel_.nodes_begin(); it != dcel_.nodes_end(); ++it) {
            nodes.row(node_idx) = it->coords().transpose();
            boundary_markers(node_idx) = it->on_boundary() ? 1 : 0;
            node_idx++;
        }

        // Fill cells matrix
        int cell_idx = 0;
        for (auto it = dcel_.cells_begin(); it != dcel_.cells_end(); ++it) {
            halfedge_t* h = it->halfedge();
            for (int i = 0; i < n_nodes_cell; ++i) {
                cells(cell_idx, i) = h->node()->id();
                h = h->next();
            }
            cell_idx++;
        }

        // Create Triangulation object
        Triangulation<local_dim, embed_dim> triangulation(nodes, cells, boundary_markers);
        return triangulation;
    }

    void export_triangulation_to_txt(const Triangulation<LocalDim, EmbedDim>& triangulation, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return;
        }

        file << "Nodes:\n";
        for (int i = 0; i < triangulation.n_nodes(); ++i) {
            auto coords = triangulation.node(i);
            int marker = triangulation.is_node_on_boundary(i) ? 1 : 0;
            file << i << " " << coords(0) << " " << coords(1) << " " << marker << "\n";
        }

        file << "\nCells:\n";
        for (int i = 0; i < triangulation.n_cells(); ++i) {
            auto cell = triangulation.cells().row(i);;
            file << i << " " << cell(0) << " " << cell(1) << " " << cell(2) << "\n";
        }

        file.close();
    }



    Triangulation() = default;

    //new conctructor needed as a semplification of trianagulation for dcel
    Triangulation(
        const Eigen::Matrix<double, Dynamic, Dynamic>& nodes, 
        const Eigen::Matrix<int, Dynamic, Dynamic>& cells,
        const Eigen::Matrix<int, Dynamic, Dynamic>& boundary, 
        int flags = 0) :
          Base(nodes, cells, boundary, flags) {}
    };





    Delaunay(const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary,
        const std::vector<Eigen::Matrix<double, Eigen::Dynamic, embed_dim>>& holes = {})
    : dcel(holes.empty() ? DCEL<local_dim, embed_dim>::make_polygon(boundary)
                    : DCEL<local_dim, embed_dim>::make_polygon(boundary, holes)) { }




                    static void triangulate(dcel_t& dcel, int N, const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary) {
                        double min_x = boundary.col(0).minCoeff();
                        double max_x = boundary.col(0).maxCoeff();
                        double min_y = boundary.col(1).minCoeff();
                        double max_y = boundary.col(1).maxCoeff();
    
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_real_distribution<double> dist_x(min_x, max_x);
                        std::uniform_real_distribution<double> dist_y(min_y, max_y);
                    
                        int first_internal_id = -1;
                        int generated_points = 0;
            
                        while (generated_points < N) {
                            coords_t u;
                            u << dist_x(gen), dist_y(gen);
                
                            if (!fdapde::internals::point_in_polygon(boundary, u))
                                continue;
                
                            //std::cout << "internal point: " << u.transpose() << std::endl;
                
                            node_t* n = dcel.insert_node(node_t(dcel.n_nodes(), false, u));
                
                            if (generated_points == 0) {
                                //add_first_triangle(dcel, n, boundary);
                                initialize_triangulation(dcel, boundary);
                                flip(dcel);
                                first_internal_id = dcel.n_nodes() - 1;
                            } else {
                                detect_conflicts(dcel, n);
                            }
                
                            ++generated_points;
                        }
                        
                    
                        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
                            node_t* u = &(*it);
                            if (!u->on_boundary() && u->id() != first_internal_id)
                                insert_vertex_at_conflict(dcel, u);
                        }
                    
                        int cont = 0;
                        for (auto it = dcel.cells_begin(); it != dcel.cells_end(); ++it)
                            it->set_id(cont++);
                        dcel.set_n_cells_(cont);
                    
                        cont = 0;
                        for (auto it = dcel.halfedges_begin(); it != dcel.halfedges_end(); ++it)
                            it->set_id(cont++);
                    }









                    /*
    static void triangulate(dcel_t& dcel, int N, const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary) {
        double min_x = boundary.col(0).minCoeff();
        double max_x = boundary.col(0).maxCoeff();
        double min_y = boundary.col(1).minCoeff();
        double max_y = boundary.col(1).maxCoeff();
    
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist_x(min_x, max_x);
        std::uniform_real_distribution<double> dist_y(min_y, max_y);
    
        int first_internal_id = -1;
        int generated_points = 0;

        while (generated_points < N) {
            coords_t u;
            u << dist_x(gen), dist_y(gen);

            if (!fdapde::internals::point_in_polygon(boundary, u))
                continue;

            //std::cout << "internal point: " << u.transpose() << std::endl;

            node_t* n = dcel.insert_node(node_t(dcel.n_nodes(), false, u));

            if (generated_points == 0) {
                //add_first_triangle(dcel, n, boundary);
                initialize_triangulation(dcel, boundary);
                //FLIP SECONDO ME NON SERVE PIU VISTO CHE POLYGON LAVORA SOLO CON I PUNTI DI BORDO
                flip(dcel);
                first_internal_id = dcel.n_nodes() - 1;
            } else {
                detect_conflicts(dcel, n);
            }

            ++generated_points;
            
        }
    
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
            node_t* u = &(*it);
            if (!u->on_boundary() && u->id() != first_internal_id)
                insert_vertex_at_conflict(dcel, u);
        }*/
    /*
        int cont = 0;
        for (auto it = dcel.cells_begin(); it != dcel.cells_end(); ++it)
            it->set_id(cont++);
        dcel.set_n_cells_(cont);
    
        cont = 0;
        for (auto it = dcel.halfedges_begin(); it != dcel.halfedges_end(); ++it)
            it->set_id(cont++);*/
 //   }




 /*
    static void add_first_triangle(dcel_t& dcel, node_t* n, const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary){   
        bool concave=false;
        auto iter = dcel.halfedges_begin();
        halfedge_t* first_h= dcel.emplace_halfedge_(n);
        auto& cell_begin = *dcel.cells_begin();
        first_h->set_cell(&cell_begin);

        // iterate over all boundary edges to connect to node, if possible
        for (int il = 0; il < boundary.rows(); ++il, ++iter) { 
            halfedge_t* v = &(*iter);                                                                  
            cell_t* c= v->cell();                                                                                    
            std::vector<halfedge_t*> halfedges_to_call(n_nodes_cell); 

            if(dcel.find_halfedge(n,c)){
                halfedge_t* h = dcel.find_halfedge(n,c);
                halfedges_to_call[2] = h;
            }
            else{  
                halfedges_to_call[2]= first_h;
            }
            halfedges_to_call[0] = v;
            if(v->next()->cell()==halfedges_to_call[2]->cell())
                halfedges_to_call[1] = v->next();
            else{
                halfedges_to_call[1] = dcel.find_halfedge(v->next()->node(),halfedges_to_call[2]->cell());
            }
            // add edges
            for (int i = 0; i < n_nodes_cell ; ++i) {
                halfedge_t* h1 = halfedges_to_call[i];
                halfedge_t* h2 = halfedges_to_call[(i + 1) % (n_nodes_cell)];
                
                // check for intersection with boundary edges
                coords_t A = h1->node()->coords();
                coords_t B = h2->node()->coords();
                coords_t C;
                coords_t D;
                node_t* node_C;
                node_t* node_D;
                bool intersect=false;
                if(!(h1->on_boundary() && h2->on_boundary())) {  //if the edges are not both on the boundary
                    for (size_t i = 0; i < boundary.rows(); ++i) {  
                        C = boundary.row(i).transpose();
                        D = boundary.row((i + 1) % boundary.rows()).transpose();  
                        if (A != C && A != D && B != C && B != D && fdapde::internals::intersect(A, B, C, D)) {  
                            intersect = true;
                            concave=true;
                            break; 
                        }
                    }
                }
                if(!intersect){
                    halfedge_t* h = dcel.insert_edge(h1, h2); 
                    if(h && h!=h1)
                        halfedges_to_call[(i + 1) % (n_nodes_cell)] = h->next();  
                } 
            }
        }
        
        if(concave){
            // cycle over boundary edges only for safe handling of edges to insert
            for(auto it=dcel.halfedges_begin(); it!=dcel.halfedges_end() && (&(*it))->cell(); ++it){
                halfedge_t* h = &(*it);
                int i=0;
                do{
                    h=h->next();
                    i++;
                }while(h!=&(*it));
                if(i>n_nodes_cell){ //not a triangle yet
                    for(int l=0; l<i-n_nodes_cell; ++l){
                      halfedge_t* n=h->next()->next();
                      if(h->node()->on_boundary() && h->next()->next()->node()->on_boundary() 
                         && !fdapde::internals::collinear(h->node()->coords(),h->next()->node()->coords(),h->next()->next()->node()->coords())
                         && fdapde::internals::are_2d_counterclockwise_sorted(h->node()->coords(),h->next()->node()->coords(),h->next()->next()->node()->coords()) ){
                        dcel.insert_edge(h, n);
                      }
                      h=n;   
                    }
                }
            }
        }
        flip(dcel);
    }
*/



 /*   static void triangulate(dcel_t& dcel, int N, const Eigen::Matrix<double, Eigen::Dynamic, embed_dim>& boundary,
                            double jitter_ratio = 0.01) {
        double min_x = boundary.col(0).minCoeff();
        double max_x = boundary.col(0).maxCoeff();
        double min_y = boundary.col(1).minCoeff();
        double max_y = boundary.col(1).maxCoeff();
        int generated_points = 0;

        int points_per_row = static_cast<int>(std::ceil(std::sqrt(N)));
        double dx = (max_x - min_x) / (points_per_row + 1);
        double dy = (max_y - min_y) / (points_per_row + 1);

        // inizializza triangolazione iniziale del dominio
        initialize_triangulation(dcel, boundary);
        flip(dcel);

        // generatore random deterministico
        std::mt19937 gen(42); // seme fisso → riproducibile
        std::uniform_real_distribution<double> jitter_x(-jitter_ratio * dx, jitter_ratio * dx);
        std::uniform_real_distribution<double> jitter_y(-jitter_ratio * dy, jitter_ratio * dy);

        for (int i = 1; i <= points_per_row; ++i) {
            for (int j = 1; j <= points_per_row; ++j) {
                coords_t u;
                u << min_x + i * dx + jitter_x(gen),
                min_y + j * dy + jitter_y(gen);

                if (!fdapde::internals::point_in_polygon(boundary, u))
                    continue;

                node_t* n = dcel.insert_node(node_t(dcel.n_nodes(), false, u));
                detect_conflicts(dcel, n);

                ++generated_points;
                if (generated_points >= N)
                    break;
            }
            if (generated_points >= N)
                break;
        }

        // Inserimento dei punti interni nella triangolazione
        for (auto it = dcel.nodes_begin(); it != dcel.nodes_end(); ++it) {
            node_t* u = &(*it);
            if (!u->on_boundary())
                insert_vertex_at_conflict(dcel, u);
        }

        // Riordino ID
        int cont = 0;
        for (auto it = dcel.cells_begin(); it != dcel.cells_end(); ++it)
            it->set_id(cont++);
        dcel.set_n_cells_(cont);

        cont = 0;
        for (auto it = dcel.halfedges_begin(); it != dcel.halfedges_end(); ++it)
            it->set_id(cont++);
    }*/


    void set_from_triangulation(const triangulation_t& triang) {
        this->nodes_ = triang.nodes();
        this->cells_ = triang.cells();
        this->boundary_markers_ = triang.boundary_nodes();
        this->neighbors_ = triang.neighbors();
        this->n_nodes_ = triang.n_nodes();
        this->n_cells_ = triang.n_cells();
    }










 std::cout << "ENCROACHED EDGES:\n";
 std::queue<halfedge_t*> debug_edges = encroached_edges;  // copia della coda originale
 
 while (!debug_edges.empty()) {
     halfedge_t* e = debug_edges.front();
     debug_edges.pop();
 
     if (!e) continue;
 
     int id = e->id();
     int from = e->node() ? e->node()->id() : -1;
     int to = e->twin() && e->twin()->node() ? e->twin()->node()->id() : -1;
 
     std::cout << "Edge ID: " << id << ", from node " << from << " to node " << to << "\n";
 }
 
 
 std::cout << "BAD TRIANGLES:\n";
 std::queue<cell_t*> debug_triangles = bad_triangles;  // copia della coda originale
 
 while (!debug_triangles.empty()) {
     cell_t* t = debug_triangles.front();
     debug_triangles.pop();
 
     if (!t || !t->halfedge()) continue;
 
     halfedge_t* h = t->halfedge();
     int id0 = h->node()->id();
     int id1 = h->next()->node()->id();
     int id2 = h->prev()->node()->id();
 
     std::cout << "Triangle: [" << id0 << ", " << id1 << ", " << id2 << "]\n";
 }






 
    // Attempts to split a bad triangle by inserting its circumcenter.
    // If the circumcenter encroaches a segment of the PLC, it splits that segment instead.
    // Returns true if a refinement was performed.
    /*bool split_triangle(cell_t* t, std::unordered_set<halfedge_t*>& encroached_edges,
        std::unordered_set<cell_t*>& bad_triangles, double rho_bar) {
        // Get triangle vertices A, B, C
        coords_t A = t->halfedge()->prev()->node()->coords();
        coords_t B = t->halfedge()->node()->coords();
        coords_t C = t->halfedge()->next()->node()->coords();

        // Compute the circumcenter of triangle ABC
        coords_t c = fdapde::internals::circumcenter(A, B, C);  

        // Check if a node already exists at the circumcenter (avoid duplicates)
        for (auto it = dcel_.nodes_begin(); it != dcel_.nodes_end(); ++it) {
            if ((it->coords() - c).norm() < 1e-12) {
                return false;  // Do not insert if a node is already at c
            }
        }

        // Check whether the circumcenter c encroaches any boundary segment
        for (auto it = dcel_.halfedges_begin(); it != dcel_.halfedges_end(); ++it) {
            halfedge_t* e = &(*it);

            // We only care about edges that are part of the PLC (on the boundary)
            if (e->on_boundary() && e->cell()) {
                coords_t a = e->node()->coords();
                coords_t b = e->twin()->node()->coords();

                // If the circumcenter c encroaches the boundary segment ab
                if (fdapde::internals::is_encroached(c, a, b)) {
                    // Only split if the edge and its neighborhood is not marked as "seditious"
                    if (!is_edge_seditious(e) && !is_edge_seditious(e->twin()) &&
                        !is_edge_seditious(e->next()) && !is_edge_seditious(e->prev()) &&
                        !is_edge_seditious(e->next()->twin()) && !is_edge_seditious(e->prev()->twin())) {
                        
                        // Split the encroached subsegment instead of inserting the circumcenter
                        split_subsegment(e, encroached_edges, bad_triangles, rho_bar);
                        return true;
                    }

                    // If the segment is seditious, do nothing now (will be retried later)
                    return false;
                }
            }
        }

        // If no encroachment is detected, insert the circumcenter into the mesh
        node_t* circ = dcel_.insert_node(node_t(dcel_.n_nodes(), false, c));
        // Locate the triangle containing the new point
        cell_t* cf = find_triangle_local(c, t); 
        // Insert the new node into the triangulation (splitting the containing triangle)
        insert_vertex(circ, cf, encroached_edges, bad_triangles, rho_bar);

        return true;
    }*/