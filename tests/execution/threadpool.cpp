// parallel-for

// int main(int argc,char** argv)
// {   
//     int granularity = std::stoi(argv[1]);
//     fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing> tp(2048,8);
    
// {// parallel_for  gran1 int
//     int end = 1001;
//     int start = 10;
//     std::vector<int> seq;
//     for(size_t i = start; i<end; i++){
//         seq.push_back(i);
//     } 
//     std::vector<int> par(end-start);
//     tp.parallel_for(start,end,[&](int i){
//         par[i-start] = i;
//     });

//     int uguali = seq.size() == par.size();
//     for(int i = 0; i<seq.size(); i++){
//         if(seq[i] != par[i]){
//             uguali *= 0;
//         } 
//     }
//     if(!uguali){std::cout<<"gran1-int NON funziona";}
//     else{std::cout<<"gran1-int funziona";}
// }
// std::cout<<std::endl; 
// {// parallel_for  gran1 it
//     int end = 10000;
//     int start = 0;
//     std::vector<int> seq;
//     for(size_t i = start; i<end; i++){
//         seq.push_back(i);
//     } 
//     std::vector<int> par(end-start);
//     tp.parallel_for(par.begin(),par.end(),[&](std::vector<int>::iterator i){
//         *i = (i-par.cbegin());
//     });

//     int uguali = seq.size() == par.size();
//     for(int i = 0; i<seq.size(); i++){
//         if(seq[i] != par[i]){
//             uguali *= 0;
//         } 
//     }
//     if(!uguali){std::cout<<"gran1-iterator NON funziona";}
//     else{std::cout<<"gran1-iterator funziona";}
// } 
// std::cout<<std::endl; 
// {// parallel_for  gran_input int
//     int end = 10000;
//     int start = 0;
//     std::vector<int> seq;
//     for(size_t i = start; i<end; i++){
//         seq.push_back(i);
//     } 
//     std::vector<int> par(end-start);
//     tp.parallel_for(start,end,[&](int i, int worker_index){
//         par[i-start] = i;
//     },granularity);

//     int uguali = seq.size() == par.size();
//     for(int i = 0; i<seq.size(); i++){
//         if(seq[i] != par[i]){
//             uguali *= 0;
//         } 
//     }
//     if(!uguali){std::cout<<"gran_input-int NON funziona";}
//     else{std::cout<<"gran_input-int funziona";}
// }
// std::cout<<std::endl; 
// {// parallel_for  gran_input it
//     int end = 10000;
//     int start = 0;
//     std::vector<int> seq;
//     for(size_t i = start; i<end; i++){
//         seq.push_back(i);
//     } 
//     std::vector<int> par(end-start);
//     tp.parallel_for(par.begin(),par.end(),[&](std::vector<int>::iterator i, int worker_index){
//         *i = (i-par.cbegin());
//     },granularity);

//     int uguali = seq.size() == par.size();
//     for(int i = 0; i<seq.size(); i++){
//         if(seq[i] != par[i]){
//             uguali *= 0;
//         } 
//     }
//     if(!uguali){std::cout<<"gran_input-iterator NON funziona";}
//     else{std::cout<<"gran_input-iterator funziona";}
// }

// std::cout<<std::endl;
// {// parallel_for vector of granularities
//     int end = 111;
//     int start = 11;
//     std::vector<int> grans = {7,9,3,1,20,23,37};
//     std::vector<int> seq;
//     for(size_t i = start; i<end; i++){
//         seq.push_back(i);
//     } 
//     std::vector<int> par(end-start);
//     tp.parallel_for(start,end,[&](int i, int worker_index){
//         par[i-start]=i;
//     },grans);

//     int uguali = seq.size() == par.size();
//     for(int i = 0; i<seq.size(); i++){
//         if(seq[i] != par[i]){
//             uguali *= 0;
//         } 
//     }
//     if(!uguali){std::cout<<"vector_gran NON funziona";}
//     else{std::cout<<"vector_gran funziona";}
// }
// std::cout<<std::endl;
// {// parallel_for  gran1-incremento personalizzato
//     int end = 10;
//     int start = 0;
//     int seq = 0;
//     for(size_t i = start; i<end; i= i+2){
//         seq+=i;
//     } 
//     std::atomic<int> par = 0;
//     tp.parallel_for(start,end,[&](int i){
//         par.fetch_add(i);
//     },[](int i){return i+2;});
    
//     if(seq != par.load()){std::cout<<"incr_personalizzato NON funziona";}
//     else{std::cout<<"incr_personalizzato funziona";}
// }
// std::cout<<std::endl;
// return 0;
// }

// reduce

// int main(int argc,char** argv){
//     int size_v = std::stoi(argv[1]);
//     fdapde::threadpool tp(1024,8);

// {
// std::cout<<"================ reduce sum  ================"<<std::endl; 
//     double a = 2;
//     std::vector<double> v (size_v,a);
//     double sum = 0;
//     sum = tp.reduce(v.begin(),v.end(),0.0,[](double a, double b){
//         return a+b;
//     }); 
//     std::cout<<"atteso: "<<a*size_v<<" , ottenuto: "<< sum<<std::endl;
// }
// {
// std::cout<<"================ reduce dot  ================"<<std::endl; 
//     std::vector<double> v (size_v,1);
//     double a = 13;
//     v[0]=a;
//     double dot = 0;
//     dot = tp.reduce(v.begin(),v.end(),1.0,[](double a, double b){
//         return a*b;
//     }); 
//     std::cout<<"atteso: "<<a<<" , ottenuto: "<< dot<<std::endl;
// }
// {
// std::cout<<"================ reduce min  ================"<<std::endl; 
//     double a = 1;
//     std::vector<double> v (size_v,a+1);
//     v[size_v/2]=a;
//     double min = 0;
//     min = tp.reduce(v.begin(),v.end(),a+2,[](double a, double b){
//         return (a<=b)? a : b;
//     }); 
//     std::cout<<"atteso: "<<a<<" , ottenuto: "<< min<<std::endl;
// }
// {
// std::cout<<"================ reduce max  ================"<<std::endl; 
//     double a = 3;
//     std::vector<double> v (size_v,a-1);
//     v[size_v/2]=a;
//     double max = 0;
//     max = tp.reduce(v.begin(),v.end(),a-2,[](double a, double b){
//         return (a>=b)? a : b;
//     }); 
//     std::cout<<"atteso: "<<a<<" , ottenuto: "<< max<<std::endl;
// }


//     return 0; //
// }

// send_steal

// #include <atomic>
// #include <future>
// #include <iostream>

// int main(int argc,char** argv){
//     int n_worker = std::stoi(argv[1]);
//     int n = 10000;

//     // round_robin + max_load_stealing
//     {
//         fdapde::threadpool<fdapde::round_robin_scheduling,fdapde::max_load_stealing> tp(1024,n_worker);
//         std::atomic<int> a = 0;
//         std::vector<std::future<void>> futs;
//         for (int i = 0; i< n; i++){
//             futs.emplace_back(tp.send([&](){a++;}));
//         }
//         for (auto& f:futs){f.get();}
//         if(a.load()==n){std::cout<<"round-max funziona\n";}
//         else{std::cout<<"round-max NON funziona\n";}
//     }

//     // round_robin + random_stealing
//     {
//         fdapde::threadpool<fdapde::round_robin_scheduling,fdapde::random_stealing> tp(1024,n_worker);
//         std::atomic<int> a = 0;
//         std::vector<std::future<void>> futs;
//         for (int i = 0; i< n; i++){
//             futs.emplace_back(tp.send([&](){a++;}));
//         }
//         for (auto& f:futs){f.get();}
//         if(a.load()==n){std::cout<<"round-random funziona\n";}
//         else{std::cout<<"round-random NON funziona\n";}
//     }

//     // round_robin + top_half_random_stealing
//     {
//         fdapde::threadpool<fdapde::round_robin_scheduling,fdapde::top_half_random_stealing> tp(1024,n_worker);
//         std::atomic<int> a = 0;
//         std::vector<std::future<void>> futs;
//         for (int i = 0; i< n; i++){
//             futs.emplace_back(tp.send([&](){a++;}));
//         }
//         for (auto& f:futs){f.get();}
//         if(a.load()==n){std::cout<<"round-top funziona\n";}
//         else{std::cout<<"round-top NON funziona\n";}
//     }

//     // least_loaded + max_load_stealing
//     {
//         fdapde::threadpool<fdapde::least_loaded_scheduling,fdapde::max_load_stealing> tp(1024,n_worker);
//         std::atomic<int> a = 0;
//         std::vector<std::future<void>> futs;
//         for (int i = 0; i< n; i++){
//             futs.emplace_back(tp.send([&](){a++;}));
//         }
//         for (auto& f:futs){f.get();}
//         if(a.load()==n){std::cout<<"least-max funziona\n";}
//         else{std::cout<<"least-max NON funziona\n";}
//     }

//     // least_loaded + random_stealing
//     {
//         fdapde::threadpool<fdapde::least_loaded_scheduling,fdapde::random_stealing> tp(1024,n_worker);
//         std::atomic<int> a = 0;
//         std::vector<std::future<void>> futs;
//         for (int i = 0; i< n; i++){
//             futs.emplace_back(tp.send([&](){a++;}));
//         }
//         for (auto& f:futs){f.get();}
//         if(a.load()==n){std::cout<<"least-random funziona\n";}
//         else{std::cout<<"least-random NON funziona\n";}
//     }

//     // least_loaded + top_half_random_stealing
//     {
//         fdapde::threadpool<fdapde::least_loaded_scheduling,fdapde::top_half_random_stealing> tp(1024,n_worker);
//         std::atomic<int> a = 0;
//         std::vector<std::future<void>> futs;
//         for (int i = 0; i< n; i++){
//             futs.emplace_back(tp.send([&](){a++;}));
//         }
//         for (auto& f:futs){f.get();}
//         if(a.load()==n){std::cout<<"least-top funziona\n";}
//         else{std::cout<<"least-top NON funziona\n";}
//     }

//     return 0;
// }



// assemble

// int main(int argc, char** argv){
//     int nodi = std::stoi(argv[1]);
//     int workers = std::stoi(argv[2]);
//     int granularity = std::stoi(argv[3]);

//     fdapde::threadpool Tp(1000,workers);
//     Triangulation<2, 2> unit_square = Triangulation<2, 2>::UnitSquare(nodi);
//     FeSpace Vh(unit_square, P1<1>);
//     TrialFunction u(Vh);
//     TestFunction  v(Vh);
//     auto a = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
//     auto b = integral(unit_square)(dot(grad(u), grad(v))); // laplacian weak form
    
// //cronometro assemblaggio non parallello
//     auto start = std::chrono::high_resolution_clock::now();
//     Eigen::SparseMatrix<double> A = a.assemble();
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
//     std::cout<<"tempo (microsec) assemblaggio non parallelo: "<<duration.count()<<std::endl; 

// //cronometro assemblaggio parallelo
//     auto start2 = std::chrono::high_resolution_clock::now();
//     Eigen::SparseMatrix<double> A2 = b.assemble(execution::par,Tp, granularity); // use parallel version
//     auto end2 = std::chrono::high_resolution_clock::now();
//     auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);  
//     std::cout<<"tempo (microsec) assemblaggio parallelo (thread: "<<workers<<" ) : "<<duration2.count()<<std::endl;  
    

//     // std::cout<<A.size()<<std::endl;
//     // std::cout<<A2.size()<<std::endl;
//     //std::cout<<A<<std::endl;
//     //std::cout<<A2<<std::endl;
//     if (A.isApprox(A2, 0.000000000000001)) {
//         std::cout << "Le matrici sono identiche." << std::endl; 
//     }else{
//         std::cout << "Le matrici sono diverse." << std::endl;
//     }

//     return 0;
// }


// grid search

// int main(int argc, char** argv){
//     double lower =-5;
//     double upper = 5;
//     std::uniform_real_distribution<double> unif(lower,upper);
//     std::default_random_engine re;
    
//     int grid_size = 0;
//     int n_threads = 1;
//     int granularity = -1; 
//     std::cout<<"numero elementi in griglia: ";
//     std::cin>>grid_size;
//     std::cout<<"numero worker in threadpool: ";
//     std::cin>>n_threads;
//     std::cout<<"granularity: ";
//     std::cin>>granularity;
//     std::cout<<std::endl;

//     //rastrigin function min in (0,0)
//     fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return 20 + p[0]*p[0] + p[1]*p[1] - 10*std::cos(2*M_PI*p[0]) - 10*std::cos(2*M_PI*p[1]); })> rastrigin;
//     //rosenbrock. min in (1,1)
//     fdapde::ScalarField<2, decltype([](const Eigen::Matrix<double, 2, 1>& p) { return std::pow(1 - p[0], 2) + 100 * std::pow(p[1] - p[0]*p[0], 2); })> rosenbrock;

//     // definizione di griglia di possibili valori
    
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> grid;
//     grid.resize(grid_size,2);

//     // grid da popolare con la griglia dei valori da esplorare
//     for(int i =0; i<grid.rows();++i){
//         grid(i,0) = unif(re);
//         grid(i,1) = unif(re);
//     }
    
//  { //rastrigin
//     std::cout<<"=========minimization of Rastrigin function ==================================="<<std::endl;
//     //creazione threadpool per versioni con Tp in input
//     fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::max_load_stealing> Tp(1024, n_threads);


//     //---------------------- no parallel---------------------- ---------------------- ---------------------- ---------------------- 
//     std::cout<<"========================sequenziale========================"<<std::endl;
//     fdapde::GridSearch<2> opt2;

//     auto start3 = std::chrono::high_resolution_clock::now();

//     opt2.optimize(rastrigin, grid); // <- da modificare questo step

//     auto end3 = std::chrono::high_resolution_clock::now();
//     auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
//     std::cout<<"tempo impiegato: "<<duration3.count()<<","<<std::endl;
//     std::cout<<"value: "<<opt2.value()<<std::endl; 
//     std::cout<<"optimum: "<<opt2.optimum()<<std::endl; 
//     std::cout<<std::endl;
    

// //---------------------- parallel: optimize (parallel_for)---------------------- ---------------------- ---------------------- ---------------------- 
//     std::cout<<"========================parallel_for optimize, granularity:"<<granularity<<", threads: "<<n_threads<<"========================"<<std::endl;
//     // definizione dell'ottimizzatore 
//     fdapde::GridSearch<2> opt3;

//     auto start4 = std::chrono::high_resolution_clock::now();

//     opt3.optimize(rastrigin, grid, execution::par,Tp,granularity);  

//     auto end4 = std::chrono::high_resolution_clock::now();
//     auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(end4 - start4);  
//     std::cout<<"tempo impiegato: "<<duration4.count()<<","<<std::endl;
//     std::cout<<"value: "<<opt3.value()<<std::endl; 
//     std::cout<<"optimum: "<<opt3.optimum()<<std::endl;
//     std::cout<<std::endl;

//     // //prima scommentare in grid_search.h il metodo optimize_variadic
//     // //---------------------- parallel_variadic : optimize (parallel_for_granularity_variadic)---------------------- ---------------------- ---------------------- ---------------------- 
//     // std::cout<<"========================parallel_for_granularity_variadic, gran:"<<granularity<<", threads: "<<n_threads<<"========================"<<std::endl;
//     // // definizione dell'ottimizzatore 
//     // fdapde::GridSearch<2> opt4;

//     // auto start5 = std::chrono::high_resolution_clock::now();

//     // opt4.optimize_variadic(rastrigin, grid, execution::par,Tp,granularity);  

//     // auto end5 = std::chrono::high_resolution_clock::now();
//     // auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(end5 - start5);  
//     // std::cout<<"tempo impiegato: "<<duration5.count()<<","<<std::endl;
//     // std::cout<<"value: "<<opt4.value()<<std::endl; 
//     // std::cout<<"optimum: "<<opt4.optimum()<<std::endl;
//     // std::cout<<std::endl;

//  }

//     return 0;
// }

