// ---------------------------- test constructor

// int main(){
// {//relaxed o wait
//     std::cout<<"test costruttori con relaxed"<<std::endl;
//     using value = int;
//     value el = 1;
//     std::list<value> v ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::relaxed> q(v.begin(),v.end());
//     std::cout<<"synchro_queue di soli 1 costruita da lista di soli 1: ";
//     q.print();
//     std::vector<value> v1 ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::relaxed> q1(v1.begin(),v1.end());
//     std::cout<<"synchro_queue di soli 1 costruita da vector di soli 1: ";
//     q1.print();
//     std::array<value,3> v2 ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::relaxed>q2(v2.begin(),v2.end());
//     std::cout<<"synchro_queue di soli 1 costruita da array di soli 1: ";
//     q2.print();
// }
// std::cout<<std::endl;
// {// hold nowait
//     std::cout<<"test costruttori con deferred"<<std::endl;
//     using value = int;
//     value el = 1;
//     std::list<value> v ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::deferred> q(v.begin(),v.end());
//     std::cout<<"synchro_queue di soli 1 costruita da lista di soli 1: ";
//     q.print();
//     std::vector<value> v1 ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::deferred> q1(v1.begin(),v1.end());
//     std::cout<<"synchro_queue di soli 1 costruita da vector di soli 1: ";
//     q1.print();
//     std::array<value,3> v2 ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::deferred>q2(v2.begin(),v2.end());
//     std::cout<<"synchro_queue di soli 1 costruita da array di soli 1: ";
//     q2.print();
// }
// std::cout<<std::endl;
// {//hold wait
//     std::cout<<"test costruttori con blocking"<<std::endl;
//     using value = int;
//     value el = 1;
//     std::list<value> v ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::blocking> q(v.begin(),v.end());
//     std::cout<<"synchro_queue di soli 1 costruita da lista di soli 1: ";
//     q.print();
//     std::vector<value> v1 ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::blocking> q1(v1.begin(),v1.end());
//     std::cout<<"synchro_queue di soli 1 costruita da vector di soli 1: ";
//     q1.print();
//     std::array<value,3> v2 ={el,el,el};

//     fdapde::synchro_queue<int,fdapde::blocking>q2(v2.begin(),v2.end());
//     std::cout<<"synchro_queue di soli 1 costruita da array  di soli 1: ";
//     q2.print();
// }
// std::cout<<std::endl;
//     return 1;

// }

// ---------------------- test pop/push

//  int main(){
//     std::cout<<"size di code (10): ";
//     int size;
//     std::cin>>size;
// {//relaxed_owait
//     std::cout<<"---------------------------------test pop push con relaxed---------------------------------"<<std::endl;
//     fdapde::synchro_queue<int,fdapde::relaxed> q(size);

//     //push_front()
//     for (int i =0; i<size; i++){
//         q.push_front(i);
//     }
//     std::cout<<"push_front():  "<<std::endl;
//     q.print();
//     // std::cout<<"head: "<<q.get_head()<<std::endl;
//     // std::cout<<"tail: "<<q.get_tail()<<std::endl;
//     //pop_front
//     for(int j=0; j<size; j++)
//         q.pop_front();
//     std::cout<<"pop_front():  "<<std::endl;
//     q.print();

//     //push_back()
//     for (int i =0; i<size; i++){
//         q.push_back(i);
//     }
//     std::cout<<"push_back():  "<<std::endl;
//     q.print();
 
//     //pop_back()
//     for(int j=0; j<size; j++)
//         q.pop_back();
//     std::cout<<"pop_back():  "<<std::endl;
//     q.print();
// }



// {//deferred
//     std::cout<<"---------------------------------test pop push con deferred---------------------------------"<<std::endl;

//     fdapde::synchro_queue<int,fdapde::deferred> q(size);

//     //push_front()
//     for (int i =0; i<size; i++){
//         q.push_front(i);
//     }
//     std::cout<<"push_front():  "<<std::endl;
//     q.print();
 
//     //pop_front
//     for(int j=0; j<size; j++)
//         q.pop_front();
//     std::cout<<"pop_front():  "<<std::endl;
//     q.print();

//     //push_back()
//     for (int i =0; i<size; i++){
//         q.push_back(i);
//     }
//     std::cout<<"push_back():  "<<std::endl;
//     q.print();
 
//     //pop_back()
//     for(int j=0; j<size; j++)
//         q.pop_back();
//     std::cout<<"pop_back():  "<<std::endl;
//     q.print();
// }

// {
//     std::cout<<"---------------------------------test pop push con blocking---------------------------------"<<std::endl;
//     fdapde::synchro_queue<int,fdapde::blocking> q(size);

//     //push_front()
//     for (int i =0; i<size; i++){
//         q.push_front(i);
//     }
//     std::cout<<"push_front():  "<<std::endl;
//     q.print();
 
//     //pop_front
//     for(int j=0; j<size; j++)
//         q.pop_front();
//     std::cout<<"pop_front():  "<<std::endl;
//     q.print();

//     //push_back()
//     for (int i =0; i<size; i++){
//         q.push_back(i);
//     }
//     std::cout<<"push_back():  "<<std::endl;
//     q.print();
 
//     //pop_back()
//     for(int j=0; j<size; j++)
//         q.pop_back();
//     std::cout<<"pop_back():  "<<std::endl;
//     q.print();
// }
//      return 0;
//     }


// test pop-push multithread

// template<typename sq>
// void pushback(sq & q, int size_local){
//     for (int i = 0; i<size_local; i++){
//         q.push_back(i);
//         std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
//     }
// }
// template<typename sq>
// void pushfront(sq & q, int size_local){
//     for (int i = 0; i<size_local; i++){
//         q.push_front(i);
//         std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
//     }
// }
// template<typename sq>
// void popback(sq & q, int size_local){
//     for (int i = 0; i<size_local; i++){
//         q.pop_back();
//         std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
//     }
// }
// template<typename sq>
// void popfront(sq & q, int size_local){
//     for (int i = 0; i<size_local; i++){
//         q.pop_front();
//         std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
//     }
// }

// int main(int argc, char** argv){
//     int size_coda=30;
//     int n_thread = 3;
//     int size_local = 10;
//     // using queue = fdapde::synchro_queue<int,fdapde::relaxed>;
//     // using queue = fdapde::synchro_queue<int,fdapde::deferred>;
//     using queue = fdapde::synchro_queue<int,fdapde::blocking>;
//     {
//     using queue = fdapde::synchro_queue<int,fdapde::relaxed>;
//     std::cout<<"relaxed pop/push front/back multithread"<<std::endl;
//         std::cout<<"push_front concorrente:"<<std::endl;
//         queue q1(size_coda);
//         std::vector<std::thread> threadpool;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(pushfront<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"pop_front concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(popfront<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"push_back concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(pushback<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"pop_back concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(popback<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();
//     }
//     std::cout<<std::endl;
//     {
//     using queue = fdapde::synchro_queue<int,fdapde::deferred>;
//     std::cout<<"deferred pop/push front/back multithread"<<std::endl;
//         std::cout<<"push_front concorrente:"<<std::endl;
//         queue q1(size_coda);
//         std::vector<std::thread> threadpool;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(pushfront<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"pop_front concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(popfront<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"push_back concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(pushback<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"pop_back concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(popback<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();
//     }
//     std::cout<<std::endl;
//     {
//     using queue = fdapde::synchro_queue<int,fdapde::blocking>;
//     std::cout<<"blocking pop/push front/back multithread"<<std::endl;
//         std::cout<<"push_front concorrente:"<<std::endl;
//         queue q1(size_coda);
//         std::vector<std::thread> threadpool;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(pushfront<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"pop_front concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(popfront<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"push_back concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(pushback<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();

//         threadpool.clear();
//         std::cout<<"pop_back concorrente:"<<std::endl;
//         for(int i = 0; i<n_thread; i++){
//             threadpool.emplace_back(popback<queue>,std::ref(q1),size_local);
//         }
//         for(int i = 0; i<n_thread; i++){
//             threadpool[i].join();
//         }
//         q1.print();
//     }
// }

// test wait pop push

// int main(){

// // blocking
// {
//     std::cout<<"---------------------------------test push back/front wait_for con blocking---------------------------------"<<std::endl;
//     fdapde::synchro_queue<int,fdapde::blocking> q1(5);
//     //popolo
//     for(int j=0; j<5; j++){
//         q1.push_front(1);
//     }
//     std::cout<<"queue: ";
//     q1.print();
//     //push
//     std::thread d(&fdapde::synchro_queue<int,fdapde::blocking>::push_back_or_wait_for,std::ref(q1),9,5); //prova a fare push ma coda piena quindi aspetta finche non viene fatto pop
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     q1.pop_back();
//     d.join();
//     q1.print(); 
    
//     std::thread d2(&fdapde::synchro_queue<int,fdapde::blocking>::push_front_or_wait_for,std::ref(q1),9,5); //prova a fare push ma coda piena quindi aspetta finche non viene fatto push
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     q1.pop_front();
//     d2.join();
//     q1.print();

// }
// {
//     std::cout<<"---------------------------------test pop bback/front wait_for con blocking---------------------------------"<<std::endl;
//     fdapde::synchro_queue<int,fdapde::blocking> q1(5);
//     std::cout<<"queue: ";
//     q1.print();
//     //pop
//     std::thread d(&fdapde::synchro_queue<int,fdapde::blocking>::pop_back_or_wait_for,std::ref(q1),5); //prova a fare pop ma coda vuota quindi aspetta finche non viene fatto pop
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     q1.push_back(9);
//     d.join();
//     q1.print();

//     std::thread d2(&fdapde::synchro_queue<int,fdapde::blocking>::pop_front_or_wait_for,std::ref(q1),5); //prova a fare pop ma coda vuota quindi aspetta finche non viene fatto push
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     q1.push_front(9);
//     d2.join();
//     q1.print();

// }
// {
//     std::cout<<"---------------------------------test push back/front wait con blocking---------------------------------"<<std::endl;
//     fdapde::synchro_queue<int,fdapde::blocking> q1(5);
//     //popolo
//     for(int j=0; j<5; j++){
//         q1.push_front(1);
//     }
//     std::cout<<"queue: ";
//     q1.print();
//     //push
//     std::thread d(&fdapde::synchro_queue<int,fdapde::blocking>::push_back_or_wait,std::ref(q1),9); //prova a fare push ma coda piena quindi aspetta finche non viene fatto pop
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     q1.pop_back();
//     d.join();
//     q1.print(); 
    
//     std::thread d2(&fdapde::synchro_queue<int,fdapde::blocking>::push_front_or_wait,std::ref(q1),9); //prova a fare push ma coda piena quindi aspetta finche non viene fatto pop
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     q1.pop_front();
//     d2.join();
//     q1.print();


// }
// {
//     std::cout<<"---------------------------------test pop bback/front wait con blocking---------------------------------"<<std::endl;
//     fdapde::synchro_queue<int,fdapde::blocking> q1(5);
//     std::cout<<"queue: ";
//     q1.print();
//     //pop
//     std::thread d(&fdapde::synchro_queue<int,fdapde::blocking>::pop_back_or_wait,std::ref(q1)); //prova a fare pop ma coda vuota quindi aspetta finche non viene fatto push
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     q1.push_back(9);
//     d.join();
//     q1.print();

//     std::thread d2(&fdapde::synchro_queue<int,fdapde::blocking>::pop_front_or_wait,std::ref(q1)); //prova a fare pop ma coda vuota quindi aspetta finche non viene fatto push
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     q1.push_front(9);
//     d2.join();
//     q1.print();

// }
// return 0;
// }

