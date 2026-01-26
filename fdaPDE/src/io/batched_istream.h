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

#ifndef __FDAPDE_BATCHED_ISTREAM_H__
#define __FDAPDE_BATCHED_ISTREAM_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// a buffered input stream implementation for block-reading of files
struct batched_istream {
    using char_t = char;
    using size_t = std::size_t;
    using buff_t = char_t*;
    static constexpr size_t blk_sz__ = 1024;   // block size is expressed as multiples of 1KB

    batched_istream() noexcept = default;
    batched_istream(const std::string& filename, size_t blk_factor = 4) :
        stream_(), size_(0), blk_sz_(blk_sz__ * blk_factor), pos_(0) {
        open(filename, blk_sz_);
    }
    batched_istream(const char* filename, size_t blk_factor = 4) :
        stream_(), size_(0), blk_sz_(blk_sz__ * blk_factor), pos_(0) {
        open(filename, blk_sz_);
    }
    batched_istream(const std::filesystem::path& filename, size_t blk_factor = 4) :
        stream_(), size_(0), blk_sz_(blk_sz__ * blk_factor), pos_(0) {
        open(filename, blk_sz_);
    }
    // delete copy/move semantic
    batched_istream(const batched_istream&) = delete;
    batched_istream& operator=(const batched_istream&) = delete;
    batched_istream(batched_istream&&) = delete;
    batched_istream& operator=(batched_istream&&) = delete;

    // read next block of data, always guarantees each block to have complete lines
    void read() {
        // read the raw block
        stream_.read(buff_, blk_sz_);
        size_t bytes_read = stream_.gcount();
        if (bytes_read == 0) {
            buff_sz_ = 0;
            return;
        }
        // find last newline
        if (!stream_.eof()) {
            size_t last_nl = bytes_read;
            while (last_nl > 0 && buff_[last_nl - 1] != '\n') { last_nl--; }

            if (last_nl > 0) {
                // rewind the file pointer to the character after the newline
                size_t overshoot = bytes_read - last_nl;
                stream_.seekg(-overshoot, std::ios::cur);
                buff_sz_ = last_nl;
            } else {
                // no newline found in the entire block (very long line)
                buff_sz_ = bytes_read;
            }
        } else {
            buff_sz_ = bytes_read;
        }
        pos_++;
        return;
    }
    // return number of valid bytes last extracted in buffer
    size_t size() const { return buff_sz_; }
    // pointer to read data
    const char* data() const { return buff_; }
    size_t tellg() const { return pos_; }
    batched_istream& seekg(size_t pos) {
        fdapde_assert(pos < n_blk_);
        pos_ = pos;
        return *this;
    }
    bool end() { return stream_.peek() == EOF; }
    operator bool() { return !end(); }
    // counts how many newline characters '\n' are found in the file (requires to read the entire file)
    size_t n_lines() {
        size_t n = 0;
        while (!end()) {
            read();
            for (size_t i = 0; i < size(); ++i) {
                if (buff_[i] == '\n') { n++; }
            }
        }
        // reset status
        stream_.clear();
        stream_.seekg(0, std::ios::beg);   // rewind to the beginning
        pos_ = 0;
        return n;
    }
    // file operations
    void close() {
        delete[] buff_;   // deallocate memory
	buff_ = nullptr;
        // reset status
        stream_.close();
        size_ = 0, n_blk_ = 0, buff_sz_ = 0, pos_ = 0;
    }
    bool is_open() const { return stream_.is_open(); }
    void open(const char* filename, size_t blk_sz) {
        if (!std::filesystem::exists(filename))
            throw std::runtime_error("file " + std::string(filename) + " not found.");
        stream_.open(filename, std::ios::binary | std::ios::ate);
        size_ = stream_.tellg();
        n_blk_ = (size_ + blk_sz - 1) / blk_sz;
        blk_sz_ = blk_sz;
        stream_.seekg(0, std::ios::beg);   // rewind to the beginning
        buff_ = new char_t[blk_sz];
        pos_ = 0;
    }
    void open(const std::string& filename, size_t blk_factor) { open(filename.c_str(), blk_factor); }
    void open(const std::filesystem::path& filename, size_t blk_factor) { open(filename.c_str(), blk_factor); }
    ~batched_istream() { close(); }
   private:
    std::ifstream stream_;
    size_t size_;      // size (in bytes) of stream
    size_t blk_sz_;    // blk_sz_ = blk_factor * blk_sz (size of block, expressed in KB)
    size_t n_blk_;     // number of blocks in stream
    size_t buff_sz_;   // number of valid bytes in r_buff and w_buff
    size_t pos_;       // index of last read block
    buff_t buff_ = nullptr;
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_BATCHED_ISTREAM_H__
