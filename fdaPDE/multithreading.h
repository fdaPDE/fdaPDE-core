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

#ifndef __FDAPDE_MULTITHREADING_MODULE_H__
#define __FDAPDE_MULTITHREADING_MODULE_H__

// clang-format off
#include<vector>
#include<mutex>
#include<thread>
#include<memory>
#include<optional>
#include<atomic>
#include<condition_variable>
#include<functional>
#include<concepts>
#include<list>
#include<future>
#include<random>
#include <shared_mutex>
#include<numeric>

// per debug e test momentaneo
#include<iostream>
#include<chrono>

#include "src/multithreading/synchro_queue.h"
#include "src/multithreading/threadpool.h"
#include "src/multithreading/execution_type.h"
// clang-format on

#endif
