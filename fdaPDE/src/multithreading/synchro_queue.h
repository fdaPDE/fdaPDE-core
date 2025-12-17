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
//
//

#ifndef __FDAPDE_SYNCHRO_QUEUE_H__
#define __FDAPDE_SYNCHRO_QUEUE_H__

#include "header_check.h"

namespace fdapde {
namespace internals {
// define concept (iterator of vector,array,list)
template <typename Iterator, typename T>
concept vector_array_list =
  std::contiguous_iterator<Iterator> || std::same_as<Iterator, typename std::list<T>::iterator>;
}   // namespace internals

// access_model:
struct relaxed { };
struct deferred { };
struct blocking { };

// forward declaration
template <typename value_type, typename access_model> class synchro_queue;

template <typename value_type> class synchro_queue<value_type, relaxed> {
   public:
    // enumerator state of elem.
    static constexpr int Empty = 1;
    static constexpr int Busy = 2;
    static constexpr int Full = 0;

    // elem relaxed
    struct elem {
        std::atomic<int> state_ = Empty;
        std::optional<value_type> v_;
    };
   private:
    typedef std::vector<elem> container;
    container queue_;
    int head_ = 0;   // indx of first element
    int tail_ = 0;   // indx of 1 over last element
    int size_ = 0;
    mutable std::mutex m_;
   public:
    // default constructor
    synchro_queue() = default;
    // construct whit size of queue_=n;
    synchro_queue(int n) : queue_(n) { size_ = n; }
    // construct from vector, array, list
    template <typename Iterator>
        requires internals::vector_array_list<Iterator, value_type>
    synchro_queue(Iterator begin, Iterator end) {
        int n = std::distance(begin, end);   // itertor of list doesn't support "end-begin"
        std::vector<elem> temp_queue(n);
        for (int i = 0; i < n; i++) {
            temp_queue[i].state_.store(Full);
            temp_queue[i].v_ = *(begin);
            std::advance(begin, 1);
        }
        std::swap(queue_, temp_queue);
        size_ = queue_.size();
    }

    synchro_queue(const synchro_queue&) = delete;
    void operator=(const synchro_queue&) = delete;

    void resize(int n) {
        std::lock_guard<std::mutex> loc(m_);
        std::vector<elem> temp_queue(n);
        std::swap(queue_, temp_queue);
        size_ = n;
        head_ = 0;
        tail_ = 0;
    }

    int get_size() const { return size_; }
    int get_tail() const { return tail_; }
    int get_head() const { return head_; }
    void clear() {
        std::lock_guard loc(m_);
        queue_.clear();
        head_ = 0;
        tail_ = 0;
    }
    // per debug momentanei
    void print() {
        for (int i = 0; i < size_; i++) {
            if (queue_[i].state_.load(std::memory_order_acquire) == Empty) {
                std::cout << 0 << " ";
            } else {
                std::cout << queue_[i].v_.value() << "  ";
            }
        }
        std::cout << std::endl;
    }

    // Try to insert val in the position before the one pointed by head and decrement head
    bool push_front(value_type val) {
        std::unique_lock<std::mutex> loc(m_);                      // lock queue's mutex
        int new_head = (head_ == 0) ? (size_ - 1) : (head_ - 1);   // index where we want to insert val
        if (queue_[new_head].state_.load(std::memory_order_acquire) != Empty)
            return false;                                                 // if element Full or Busy, abort
        queue_[new_head].state_.store(Busy, std::memory_order_relaxed);   // from Empty to Busy
        head_ = new_head;                                                 // update head_
        loc.unlock();                                                     // unlock queue's mutex
        // actual push
        queue_[new_head].v_ = std::move(val);
        queue_[new_head].state_.store(Full, std::memory_order_release);   // from Busy to Full
        return true;
    }

    // Try to remove value of element in the position pointed by head and increment head
    std::optional<value_type> pop_front() {
        std::unique_lock<std::mutex> loc(m_);   // lock queue's mutex
        int h = head_;                          // index where we want to pop
        if (queue_[h].state_.load(std::memory_order_acquire) != Full)
            return std::nullopt;                                   // if element Empty or Busy, abort
        queue_[h].state_.store(Busy, std::memory_order_relaxed);   // from Full to Busy
        head_ = (head_ == size_ - 1) ? (0) : (head_ + 1);          // update head_
        loc.unlock();                                              // unlock queue's mutex
        // actual pop
        value_type ret = std::move(queue_[h].v_.value());
        queue_[h].v_ = std::nullopt;
        queue_[h].state_.store(Empty, std::memory_order_release);   // from Busy to Empty
        return ret;
    }

    // Try to insert val in the position pointed by tail_ and increment tail_
    bool push_back(value_type val) {
        std::unique_lock<std::mutex> loc(m_);                                          // lock queue's mutex
        int t = tail_;                                                                 // index where we want to push
        if (queue_[t].state_.load(std::memory_order_acquire) != Empty) return false;   // if element Full or Busy, abort
        queue_[t].state_.store(Busy, std::memory_order_relaxed);                       // from Empty to Busy
        tail_ = (tail_ == size_ - 1) ? (0) : (tail_ + 1);                              // update tail_
        loc.unlock();                                                                  // unlock queue's mutex
        // actual push
        queue_[t].v_ = std::move(val);
        queue_[t].state_.store(Full, std::memory_order_release);   // from Busy to Full
        return true;
    }

    // Try to remove value of element in the position before the one pointed by tail_
    std::optional<value_type> pop_back() {
        std::unique_lock<std::mutex> loc(m_);                      // lock queue's mutex
        int new_tail = (tail_ == 0) ? (size_ - 1) : (tail_ - 1);   // index to pop
        if (queue_[new_tail].state_.load(std::memory_order_acquire) != Full)
            return std::nullopt;                                          // if element Empty or Busy, abort
        queue_[new_tail].state_.store(Busy, std::memory_order_relaxed);   // from Full to Busy
        tail_ = new_tail;                                                 // update tail_
        loc.unlock();                                                     // unlock queue's mutex
        // actual pop
        value_type ret = std::move(queue_[new_tail].v_.value());
        queue_[new_tail].v_ = std::nullopt;
        queue_[new_tail].state_.store(Empty, std::memory_order_release);   // from Busy to Empty
        return ret;
    }

    // return true if queue is empty, false otherwise
    bool empty() const {
        std::lock_guard<std::mutex> loc(m_);
        // lock mutex so head and tail do not change in the meantime
        if (head_ == tail_) {
            // head_ == tail_ either because full or empty queue
            while (true) {   // while waiting for possible elem's state Busy
                if (queue_[tail_].state_.load(std::memory_order_acquire) == Empty) { return true; }
                if (queue_[tail_].state_.load(std::memory_order_acquire) == Full) { return false; }
            }
        } else {
            return false;
        }
    }
};

template <typename value_type> class synchro_queue<value_type, deferred> {
   public:
    // enumerator state of elem.
    static constexpr int Empty = 1;
    static constexpr int Full = 0;
    // elem hold
    struct elem {
        int state_ = Empty;
        std::optional<value_type> v_;
        mutable std::mutex m_el_;
        std::condition_variable cv_ready_to_push_;   // CV used to ensure that push methods can proceed
        std::condition_variable cv_ready_to_pop_;    // CV used to ensure that pop methods can proceed
        int count_pop_ = 0;   // counter of pops currently in progress on this element, used in the empty() method
    };
   private:
    typedef std::vector<elem> container;
    container queue_;
    int head_ = 0;   // indx of first element
    int tail_ = 0;   // indx of 1 over last element
    int size_ = 0;
    bool empty_queue_ =
      true;   // (head_==tail_ & empty_queue_)->queue is empty, (head_==tail_ & !empty_queue_)->queue is full
    mutable std::mutex m_;
   public:
    // default constructor
    synchro_queue() = default;
    // construct whit size of queue_=n;
    synchro_queue(int n) : queue_(n) { size_ = n; }
    // constructor from list array vector
    template <typename Iterator>
        requires internals::vector_array_list<Iterator, value_type>
    synchro_queue(Iterator begin, Iterator end) {
        int n = std::distance(begin, end);
        std::vector<elem> temp_queue(n);
        for (int i = 0; i < n; i++) {
            temp_queue[i].state_ = Full;
            temp_queue[i].v_ = *(begin);
            std::advance(begin, 1);
        }
        std::swap(queue_, temp_queue);
        size_ = queue_.size();
        empty_queue_ = false;
    }

    synchro_queue(const synchro_queue&) = delete;
    void operator=(const synchro_queue&) = delete;

    void resize(int n) {
        std::lock_guard<std::mutex> loc(m_);
        std::vector<elem> temp_queue(n);
        std::swap(queue_, temp_queue);
        size_ = n;
        head_ = 0;
        tail_ = 0;
        empty_queue_ = true;
    }

    int get_size() const { return size_; }
    int get_tail() const { return tail_; }
    int get_head() const { return head_; }
    void clear() {
        std::lock_guard loc(m_);
        queue_.clear();
        head_ = 0;
        tail_ = 0;
        empty_queue_ = true;
    }

    // per debug momentanei
    void print() {
        for (int i = 0; i < size_; i++) {
            if (queue_[i].state_ == Empty) {
                std::cout << 0 << " ";
            } else {
                std::cout << queue_[i].v_.value() << "  ";
            }
        }
        std::cout << std::endl;
    }

    // Try to insert val in the position preceding the one pointed by head and decrement head
    bool push_front(value_type val) {
        std::unique_lock<std::mutex> loc(m_);                    // lock queue's mutex
        if (head_ == tail_ && !empty_queue_) { return false; }   // if queue is full, abort
        empty_queue_ = false;   // from here on the method cannot abort, so set empty_queue_ = false (already false for
                                // all push_front except the first)
        int new_head = (head_ == 0) ? (size_ - 1) : (head_ - 1);   // index where val will be inserted
        head_ = new_head;                                          // upload head_
        loc.unlock();                                              // unlock queue's mutex

        std::unique_lock<std::mutex> loc_el(queue_[new_head].m_el_);   // lock element's mutex
        queue_[new_head].cv_ready_to_push_.wait(
          loc_el, [this, new_head]() { return queue_[new_head].state_ == Empty; });   // wait until state_ == Empty
        // actual push
        queue_[new_head].v_ = std::move(val);
        queue_[new_head].state_ = Full;                   // from Empty to Full
        loc_el.unlock();                                  // unlock element's mutex
        queue_[new_head].cv_ready_to_pop_.notify_one();   // notify any waiting pop on this element
        return true;
    }

    // Try to remove the value of the element pointed by head and increment head
    std::optional<value_type> pop_front() {
        std::unique_lock<std::mutex> loc(m_);               // lock queue's mutex
        if (empty_queue_) { return std::nullopt; }          // if queue is empty, abort
        int h = head_;                                      // index from which pop will be performed
        head_ = (head_ == size_ - 1) ? (0) : (head_ + 1);   // upload head_
        if (head_ == tail_) { empty_queue_ = true; }        // update empty_queue_ if queue becomes empty
        queue_[h].count_pop_++;                             // mark that a pop is pending on this element
        loc.unlock();                                       // unlock queue's mutex

        std::unique_lock<std::mutex> loc_el(queue_[h].m_el_);   // lock element's mutex
        queue_[h].cv_ready_to_pop_.wait(
          loc_el, [this, h]() { return queue_[h].state_ == Full; });   // wait until state_ == Full
        // actual pop
        value_type ret = std::move(queue_[h].v_.value());
        queue_[h].v_ = std::nullopt;
        queue_[h].state_ = Empty;                   // from Full to Empty
        queue_[h].count_pop_--;                     // pop completed
        loc_el.unlock();                            // unlock element's mutex
        queue_[h].cv_ready_to_push_.notify_one();   // notify any waiting push on this element
        return ret;
    }

    // Try to insert val in the position pointed by tail_ and increment tail_
    bool push_back(value_type val) {
        std::unique_lock<std::mutex> loc(m_);                    // lock queue's mutex
        if (head_ == tail_ && !empty_queue_) { return false; }   // if queue is full, abort
        empty_queue_ = false;   // already false for all push_back except the first, but avoids branching
        int t = tail_;          // index where push will be performed
        tail_ = (tail_ == size_ - 1) ? (0) : (tail_ + 1);   // upload tail_
        loc.unlock();                                       // unlock queue's mutex

        std::unique_lock<std::mutex> loc_el(queue_[t].m_el_);   // lock element's mutex
        queue_[t].cv_ready_to_push_.wait(
          loc_el, [this, t]() { return queue_[t].state_ == Empty; });   // wait until state_ == Empty
        // actual push
        queue_[t].v_ = std::move(val);
        queue_[t].state_ = Full;                   // from Empty to Full
        loc_el.unlock();                           // unlock element's mutex
        queue_[t].cv_ready_to_pop_.notify_one();   // notify any waiting pop on this element

        return true;
    }

    // Try to remove the value of the element before tail_ and decrement tail_
    std::optional<value_type> pop_back() {
        std::unique_lock<std::mutex> loc(m_);                      // lock queue's mutex
        if (empty_queue_) { return std::nullopt; }                 // if queue is empty, abort
        int new_tail = (tail_ == 0) ? (size_ - 1) : (tail_ - 1);   // index where pop will be performed
        tail_ = new_tail;                                          // upload tail_
        if (head_ == tail_) { empty_queue_ = true; }               // update empty_queue_ if queue becomes empty
        queue_[new_tail].count_pop_++;                             // mark that a pop is pending on this element
        loc.unlock();                                              // unlock queue's mutex

        std::unique_lock<std::mutex> loc_el(queue_[new_tail].m_el_);   // lock element's mutex
        queue_[new_tail].cv_ready_to_pop_.wait(
          loc_el, [this, new_tail]() { return queue_[new_tail].state_ == Full; });   // wait until state_ == Full
        // actual pop
        value_type ret = std::move(queue_[new_tail].v_.value());
        queue_[new_tail].v_ = std::nullopt;
        queue_[new_tail].state_ = Empty;                   // from Full to Empty
        queue_[new_tail].count_pop_--;                     // pop completed
        loc_el.unlock();                                   // unlock element's mutex
        queue_[new_tail].cv_ready_to_push_.notify_one();   // notify any waiting push on this element
        return ret;
    }

    bool empty() const {
        std::lock_guard<std::mutex> loc(m_);
        if (empty_queue_) {
            // reading count pop inside element's mutex to ensure that if count_pop==0 then the last pop (moving the
            // element out of the queue) has indeed been executed.
            bool not_empty = true;
            while (not_empty) {   // wait untile count_pop == 0
                std::unique_lock<std::mutex> loc_el(queue_[tail_].m_el_);
                not_empty = (queue_[tail_].count_pop_ != 0);
                loc_el.unlock();
                std::this_thread::yield();
            };
            return true;
        } else
            return false;
    }
};

template <typename value_type> class synchro_queue<value_type, blocking> {
   public:
    // elem hold
    struct elem {
        int state_ = Empty;
        std::optional<value_type> v_;
        mutable std::mutex m_el_;
        std::condition_variable
          cv_ready_to_push_;   // CV used to ensure that push methods can proceed after lock elem's mutex
        std::condition_variable cv_ready_to_pop_;   // CV used to ensure that pop methods can proceed
        int count_pop_ = 0;   // counter of pops currently in progress on this element, used in the empty() method
    };
   private:
    typedef std::vector<elem> container;
    container queue_;
    int head_ = 0;   // indx of first element
    int tail_ = 0;   // indx of 1 over last element
    int size_ = 0;
    bool empty_queue_ = true;
    mutable std::mutex m_;
    std::condition_variable cv_can_push_;   // CV used to ensure that push methods can proceed after lock queue's mutex
                                            // (used in push*_wait*() method)
    std::condition_variable cv_can_pop_;    // CV used to ensure that pop methods can proceed after lock queue's mutex
                                            // (used in pop*_wait*() method)

    // private auxiliary methods to avoid rewriting the same code
    // note: notifica a cv_can_*_ unica differenza con deferred, quindi commenti codice non duplicati
    bool push_front_(
      value_type& val, std::unique_lock<std::mutex>&
                         loc) {   // unique_lock of queue's mutex in input already lock. reference a val because in only
                                  // an intermediet staep, in queue store a copy of original val
        empty_queue_ = false;
        int new_head = (head_ == 0) ? (size_ - 1) : (head_ - 1);
        head_ = new_head;
        loc.unlock();
        cv_can_pop_.notify_one();   // notify any waiting pop_*_or_wait*
        std::unique_lock<std::mutex> loc_el(queue_[new_head].m_el_);
        queue_[new_head].cv_ready_to_push_.wait(
          loc_el, [this, new_head]() { return queue_[new_head].state_ == Empty; });
        queue_[new_head].v_ = std::move(val);
        queue_[new_head].state_ = Full;
        loc_el.unlock();
        queue_[new_head].cv_ready_to_pop_.notify_one();
        return true;
    }

    value_type pop_front_(std::unique_lock<std::mutex>& loc) {
        int h = head_;
        head_ = (head_ == size_ - 1) ? (0) : (head_ + 1);
        if (head_ == tail_) { empty_queue_ = true; }
        queue_[h].count_pop_++;
        loc.unlock();
        cv_can_push_.notify_one();   // notify any waiting push_*_or_wait*
        std::unique_lock<std::mutex> loc_el(queue_[h].m_el_);
        queue_[h].cv_ready_to_pop_.wait(loc_el, [this, h]() { return queue_[h].state_ == Full; });
        value_type ret = std::move(queue_[h].v_.value());
        queue_[h].v_ = std::nullopt;
        queue_[h].state_ = Empty;
        queue_[h].count_pop_--;
        loc_el.unlock();
        queue_[h].cv_ready_to_push_.notify_one();
        return ret;
    }

    bool push_back_(value_type& val, std::unique_lock<std::mutex>& loc) {
        empty_queue_ = false;
        int t = tail_;
        tail_ = (tail_ == size_ - 1) ? (0) : (tail_ + 1);
        loc.unlock();
        cv_can_pop_.notify_one();   // notify any waiting pop_*_or_wait*
        std::unique_lock<std::mutex> loc_el(queue_[t].m_el_);
        queue_[t].cv_ready_to_push_.wait(loc_el, [this, t]() { return queue_[t].state_ == Empty; });
        queue_[t].v_ = std::move(val);
        queue_[t].state_ = Full;
        loc_el.unlock();
        queue_[t].cv_ready_to_pop_.notify_one();
        return true;
    }

    value_type pop_back_(std::unique_lock<std::mutex>& loc) {
        int new_tail = (tail_ == 0) ? (size_ - 1) : (tail_ - 1);
        tail_ = new_tail;
        if (head_ == tail_) { empty_queue_ = true; }
        queue_[new_tail].count_pop_++;
        loc.unlock();
        cv_can_push_.notify_one();   // notify any pending push_*_or_wait*
        std::unique_lock<std::mutex> loc_el(queue_[new_tail].m_el_);
        queue_[new_tail].cv_ready_to_pop_.wait(loc_el, [this, new_tail]() { return queue_[new_tail].state_ == Full; });
        value_type ret = std::move(queue_[new_tail].v_.value());
        queue_[new_tail].v_ = std::nullopt;
        queue_[new_tail].state_ = Empty;
        queue_[new_tail].count_pop_--;
        loc_el.unlock();
        queue_[new_tail].cv_ready_to_push_.notify_one();
        return ret;
    }
   public:
    // enumerator state of elem.
    static constexpr int Empty = 1;
    static constexpr int Full = 0;
    // default constructor
    synchro_queue() = default;
    // construct whit size of queue_=n;
    synchro_queue(int n) : queue_(n) { size_ = n; }
    // constructor from list array vector
    template <typename Iterator>
        requires internals::vector_array_list<Iterator, value_type>
    synchro_queue(Iterator begin, Iterator end) {
        int n = std::distance(begin, end);
        std::vector<elem> temp_queue(n);
        for (int i = 0; i < n; i++) {
            temp_queue[i].state_ = Full;
            temp_queue[i].v_ = *(begin);
            std::advance(begin, 1);
        }
        std::swap(queue_, temp_queue);
        size_ = queue_.size();
        empty_queue_ = false;
    }

    synchro_queue(const synchro_queue&) = delete;
    void operator=(const synchro_queue&) = delete;

    void resize(int n) {
        std::lock_guard<std::mutex> loc(m_);
        std::vector<elem> temp_queue(n);
        std::swap(queue_, temp_queue);
        size_ = n;
        head_ = 0;
        tail_ = 0;
        empty_queue_ = true;
    }

    int get_size() const { return size_; }
    int get_tail() const { return tail_; }
    int get_head() const { return head_; }
    void clear() {
        std::lock_guard loc(m_);
        queue_.clear();
        head_ = 0;
        tail_ = 0;
        empty_queue_ = true;
    }
    // per debug momentanei
    void print() {
        for (int i = 0; i < size_; i++) {
            if (queue_[i].state_ == Empty) {
                std::cout << 0 << " ";
            } else {
                std::cout << queue_[i].v_.value() << "  ";
            }
        }
        std::cout << std::endl;
    }

    bool push_front(value_type val) {
        std::unique_lock<std::mutex> loc(m_);
        if (head_ == tail_ && !empty_queue_) { return false; }   // if the queue is full, abort

        return push_front_(val, loc);
    }

    bool push_front_or_wait_for(value_type val, int s) {
        std::unique_lock<std::mutex> loc(m_);
        bool flag = cv_can_push_.wait_for(loc, std::chrono::seconds(s), [this]() {
            return this->head_ != this->tail_ || this->empty_queue_;
        });                            // wait up to s seconds for the queue to become non-full
        if (!flag) { return false; }   // abort if the timer expires

        return push_front_(val, loc);
    }

    bool push_front_or_wait(value_type val) {
        std::unique_lock<std::mutex> loc(m_);
        cv_can_push_.wait(loc, [this]() {
            return this->head_ != this->tail_ || this->empty_queue_;
        });   // wait until the queue becomes non-full

        return push_front_(val, loc);
    }

    std::optional<value_type> pop_front() {
        std::unique_lock<std::mutex> loc(m_);
        if (empty_queue_) { return std::nullopt; }   // if the queue is empty, abort

        return pop_front_(loc);
    }

    std::optional<value_type> pop_front_or_wait_for(int s) {
        std::unique_lock<std::mutex> loc(m_);
        bool flag = cv_can_pop_.wait_for(loc, std::chrono::seconds(s), [this]() {
            return !this->empty_queue_;
        });                                   // wait up to s seconds for the queue to become non-empty
        if (!flag) { return std::nullopt; }   // abort if the timer expires

        return pop_front_(loc);
    }

    value_type pop_front_or_wait() {
        std::unique_lock<std::mutex> loc(m_);
        cv_can_pop_.wait(loc, [this]() { return !this->empty_queue_; });   // wait until the queue becomes non-empty

        return pop_front_(loc);
    }

    bool push_back(value_type val) {
        std::unique_lock<std::mutex> loc(m_);
        if (head_ == tail_ && !empty_queue_) { return false; }   // if the queue is full, abort

        return push_back_(val, loc);
    }

    bool push_back_or_wait_for(value_type val, int s) {
        std::unique_lock<std::mutex> loc(m_);
        bool flag = cv_can_push_.wait_for(loc, std::chrono::seconds(s), [this]() {
            return this->head_ != this->tail_ || this->empty_queue_;
        });                            // wait up to s seconds for the queue to become non-full
        if (!flag) { return false; }   // abort if the timer expires

        return push_back_(val, loc);
    }

    bool push_back_or_wait(value_type val) {
        std::unique_lock<std::mutex> loc(m_);
        cv_can_push_.wait(loc, [this]() {
            return this->head_ != this->tail_ || this->empty_queue_;
        });   // wait until the queue becomes non-full

        return push_back_(val, loc);
    }

    std::optional<value_type> pop_back() {
        std::unique_lock<std::mutex> loc(m_);
        if (empty_queue_) { return std::nullopt; }   // if the queue is empty, abort

        return pop_back_(loc);
    }

    std::optional<value_type> pop_back_or_wait_for(int s) {
        std::unique_lock<std::mutex> loc(m_);
        bool flag = cv_can_pop_.wait_for(loc, std::chrono::seconds(s), [this]() {
            return !this->empty_queue_;
        });                                   // wait up to s seconds for the queue to become non-empty
        if (!flag) { return std::nullopt; }   // abort if the timer expires

        return pop_back_(loc);
    }

    value_type pop_back_or_wait() {
        std::unique_lock<std::mutex> loc(m_);
        cv_can_pop_.wait(loc, [this]() { return !this->empty_queue_; });   // wait until the queue becomes non-empty

        return pop_back_(loc);
    }

    // return true if the queue is empty, false otherwise. (see deferred for comments)
    bool empty() const {
        std::lock_guard<std::mutex> loc(m_);
        if (empty_queue_) {
            bool not_empty = true;
            while (not_empty) {
                std::unique_lock<std::mutex> loc_el(queue_[tail_].m_el_);
                not_empty = (queue_[tail_].count_pop_ != 0);
                loc_el.unlock();
                std::this_thread::yield();
            };
            return true;
        } else
            return false;
    }
};
}   // namespace fdapde
#endif
