// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <fdaPDE/execution.h>
#include <gtest/gtest.h>   // testing framework
using namespace fdapde;

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::microseconds;

// stress test using a task object to detect data races 
struct chase_lev_task {
    std::atomic<bool> taken {false};
    int id;
    std::function<void(int&)> data;
    // large payload to force std::function to use the heap and increases the time spent in the copy/move constructor.
    std::vector<char> payload;

    chase_lev_task() : id(-1) {}
    chase_lev_task(int i) : id(i), payload(1024, 'a') {
        data = [p = payload](int& counter) { 
            // accessing the captured vector ensures the heap memory is touched
            if(p[0] == 'a') counter++;
        };
    }
    template <typename Task> chase_lev_task(int i, Task&& task) : id(i), data(task) { }
};
// TEST(execution, chase_lev_queue) {
//     const int num_items = 1000000;
//     const int num_stealers = 7;
//     // Small buffer size to force wrap-around and frequent slot reuse
//     internals::chase_lev_queue<chase_lev_task*> queue(1024); 

//     std::atomic<bool> producer_done {false};
//     std::vector<int> count(num_stealers + 1, 0);
//     auto results = std::make_unique<std::atomic<int>[]>(num_items);
//     for (int i = 0; i < num_items; ++i) { results[i].store(0); }

//     std::atomic<int> work{1};
    
//     // Stealer threads
//     std::vector<std::thread> stealers;
//     for (int i = 0; i < num_stealers; ++i) {
//         stealers.emplace_back([&, i]() {
//             while (!producer_done || (work.load(std::memory_order_acquire) > 0)) {
//                 auto val = queue.pop_back();
//                 if (val) {
//                     if ((*val)->data) {
//                         ((*val)->data)(count[i]);
//                         std::this_thread::sleep_for(std::chrono::nanoseconds(100));   // make stealers slow
//                         results[(*val)->id].fetch_add(1);
//                         delete (*val);
// 			work.fetch_sub(1, std::memory_order_release);
//                     }
//                 }
//             }
//         });
//     }

//     std::vector<char> vec(1024, 'a');
//     // Owner thread
//     for (int i = 0; i < num_items; ++i) {
//         auto data = [p = vec, i_ = i](int& counter) {
//             if (p[i_ % p.size()] == 'a') counter++;
//         };
//         auto* task_ptr = new chase_lev_task(i, data);
//         while (!queue.push_front(task_ptr)) { std::this_thread::yield(); }
//         work.fetch_add(1, std::memory_order_release);
//         if (i % 2 == 0) {
//             auto val = queue.pop_front();
//             if (val) {
//                 if ((*val)->data) {
//                     if ((*val)->taken.exchange(true)) { std::abort(); }
//                     ((*val)->data)(count[num_stealers]);
//                     results[(*val)->id].fetch_add(1);
// 		    delete (*val);
// 		    work.fetch_sub(1, std::memory_order_release);
//                 }
//             }
//         }
//     }

//     producer_done = true;
//     work.fetch_sub(1, std::memory_order_release);
//     for (auto& t : stealers) { t.join(); }

//     int missing = 0;
//     int duplicates = 0;
//     for (int i = 0; i < num_items; ++i) {
//             if (results[i] == 0) { missing++; }
//             if (results[i] > 1) { duplicates++; }
//     }

//     EXPECT_TRUE(missing == 0);
//     EXPECT_TRUE(duplicates == 0);
//     int sum_count = std::accumulate(count.begin(), count.end(), 0);
//     EXPECT_TRUE(sum_count == num_items);
// }

// TEST(execution, mpsc_queue) {
//     internals::mpsc_queue<int> queue;

//     queue.push(10);
//     queue.push(20);
    
//     auto v1 = queue.pop();
//     auto v2 = queue.pop();
//     auto v3 = queue.pop();
    
//     EXPECT_TRUE(v1.has_value());
//     EXPECT_EQ(v1.value(), 10);
//     EXPECT_TRUE(v2.has_value());
//     EXPECT_EQ(v2.value(), 20);
//     EXPECT_FALSE(v3.has_value());

//     {
//         internals::mpsc_queue<int> queue;
//         const int num_producers = 7;
//         const int items_per_producer = 100000;

//         // 1. Launch Producers
//         std::vector<std::thread> producers;
//         for (int i = 0; i < num_producers; ++i) {
//             producers.emplace_back([&, i]() {
//                 for (int j = 0; j < items_per_producer; ++j) { queue.push(i * items_per_producer + j); }
//             });
//         }

//         // 2. Launch Consumer
//         std::vector<int> results;
//         std::thread consumer([&]() {
//             std::size_t total_expected = num_producers * items_per_producer;
//             while (results.size() < total_expected) {
//                 auto val = queue.pop();
//                 if (val) {
//                     results.push_back(*val);
//                 } else {
//                     std::this_thread::yield();   // Wait for producers
//                 }
//             }
//         });

//         for (auto& t : producers) t.join();
//         consumer.join();

//         // 4. Verification
//         EXPECT_EQ(results.size(), num_producers * items_per_producer);

//         // Ensure all unique values are present
//         std::set<int> unique_vals(results.begin(), results.end());
//         EXPECT_EQ(unique_vals.size(), num_producers * items_per_producer);
//     }
// }


TEST(execution, threadpool) {

  {
    ThreadPool tp;

    // tp.parallel_for(0, 10, [] (int i){ std::cout << i << " ";});
    // std::cout << std::endl;

    // tp.parallel_for(0, 100, 7, [] (int) { std::cout << this_worker_id() << std::endl; });

    // tp.send([]{ std::cout << "ciao da " << this_worker_id() << std::endl;});
    // tp.send([]{ std::cout << "ciao da " << this_worker_id() << std::endl;});
    // tp.send([]{ std::cout << "ciao da " << this_worker_id() << std::endl;});
    // tp.send([]{ std::cout << "ciao da " << this_worker_id() << std::endl;});

    // tp.join();

    std::vector<double> v1;
    v1.resize(10000);
    for (int i = 0; i < int(v1.size()); ++i) { v1[i] = 1; }
    std::vector<double> v2;
    v2.resize(10000);
    for (int i = 0; i < int(v2.size()); ++i) { v2[i] = 1; }

    std::vector<double> x;
    x.resize(tp.n_workers());
    std::fill(x.begin(), x.end(), 0);

    std::cout << "deadlock al primo" << std::endl;
    std::cout << "n workers: " << tp.n_workers() << std::endl;
    
    // tp.parallel_for(0, int(v1.size()), 20, [&](int i) { x[this_thread_id()] += v1[i] * v2[i]; });

    std::cout << "mi blocco al secondo" << std::endl;
    auto t1 = high_resolution_clock::now();
    tp.parallel_for(0, int(v1.size()), 100, [&](int i) mutable { x[this_thread_id()] += v1[i] * v2[i]; });
    auto t2 = high_resolution_clock::now();

    std::cout << "proseguo threadpool distrutta" << std::endl;

    auto us_int_1 = duration_cast<microseconds>(t2 - t1);

    std::cout << "sum: " << std::accumulate(x.begin(), x.end(), 0) << std::endl;
    std::cout << us_int_1.count() << "us\n";

  }
    
    // double y = 0;

    // auto t3 = high_resolution_clock::now();
    // for (int i = 0; i < int(v1.size()); ++i) { y += v1[i] * v2[i]; }
    // auto t4 = high_resolution_clock::now();

    // auto us_int_2 = duration_cast<microseconds>(t4 - t3);

    // std::cout << "sum: " << y << std::endl;
    // std::cout << us_int_2.count() << "us\n";

    // // fdapde::parallel_for_each(v1, [](auto& v) { v += 2; });

    // for (int i = 0; i < int(v1.size()); ++i) { y += v1[i]; }
    // std::cout << y << std::endl;

    // tp.join();
    
}

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

