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

#ifndef __PARSING_H__
#define __PARSING_H__

#include <string>

namespace fdapde {
namespace internals {

// A collection of convinient parsing utils

template <typename CharBuff>
concept is_char_buff =
  requires(CharBuff c, std::size_t i) {
      { c.operator[](i) } -> std::convertible_to<char>;
  } || (std::is_pointer_v<CharBuff> && std::is_same_v<std::remove_cvref_t<CharBuff>, std::add_const_t<char>*>) ||
  (std::is_array_v<std::remove_cvref_t<CharBuff>> &&
   std::is_same_v <std::remove_cvref_t<decltype(std::declval<CharBuff>()[0])>, char>);

// returns the position of the next occurence of c in buff, buff_sz if no occurence is found
template <typename CharBuff>
    requires(is_char_buff<CharBuff>)
size_t next_char_(const CharBuff& buff, std::size_t begin, std::size_t end, char c) {
    int i = begin;
    while (i < end && buff[i] != c) { i++; }
    return i - begin;
}
// returns the position of the next occurence of c or '\n' in buff, end if no occurence is found
template <typename CharBuff>
    requires(is_char_buff<CharBuff>)
size_t next_char_or_newline_(const CharBuff& buff, std::size_t begin, std::size_t end, char c) {
    int i = begin;
    while (i < end && buff[i] != c && (buff[i] != EOF && buff[i] != '\n')) { i++; }
    return i - begin;
}

// transforms a char buffer into a stream of tokens separated by sep
template <typename CharT> struct token_stream {
    using buff_t = std::add_pointer_t<CharT>;
    using size_t = std::size_t;

    token_stream() = default;
    token_stream(CharT* buff, size_t buff_sz, char sep) :
        buff_(buff), buff_sz_(buff_sz), head_(0), tail_(0), sep_(sep) { }
    token_stream(const std::string& buff, char sep) :
        buff_(buff.c_str()), buff_sz_(buff.size()), head_(0), tail_(0), sep_(sep) { }

    // a line is a contiguous portion of buffer encolsed between newline chars '\n'
    struct line_iterator {
        using value_t = std::string_view;
        using reference = std::add_lvalue_reference_t<value_t>;
        using pointer = std::add_pointer_t<value_t>;
      
        line_iterator() noexcept = default;
        line_iterator(buff_t buff, size_t buff_sz, size_t begin, size_t end, char sep) :
            buff_(buff), token_sz_(0), buff_sz_(buff_sz), begin_(begin), end_(end), pos_(0), sep_(sep) {
            fetch_token_();
        }
        bool has_token() const { return has_token_; }
        operator bool() { return has_token(); }
        line_iterator& operator++() {
            fetch_token_();
            return *this;
        }
        reference get_token() { return token_; }
        bool eol() const { return pos_ >= (end_ - begin_); }     // true if token_ is the last of this line
        bool eof() const { return begin_ + pos_ >= buff_sz_; }   // true if token_ is the last of the stream
        size_t n_tokens() {
            size_t n = 0;
	    pos_ = 0;
            while (has_token()) {
                n++;
                fetch_token_();
            }
            // reset status
            token_sz_ = 0;
            pos_ = 0;
	    token_ = std::string_view{};
            return n;
        }
       private:
        void fetch_token_() {
            has_token_ = pos_ < (end_ - begin_);
            if (has_token_) {
                token_sz_ = next_char_or_newline_(buff_, begin_ + pos_, end_, sep_);
                token_ = value_t(buff_ + (begin_ + pos_), token_sz_);
                pos_ += token_sz_ + 1;
            }
        }
        value_t token_;
        const buff_t buff_;
        char sep_;
        size_t token_sz_, buff_sz_;
        size_t begin_, end_, pos_;
        mutable bool has_token_ = true;
    };
    // observers
    line_iterator get_line() {
        head_ = tail_;
        tail_ = tail_ + next_char_(buff_, head_, buff_sz_, '\n');
        return line_iterator {buff_, buff_sz_, head_, tail_++, sep_};
    }
    bool has_line() const { return head_ < buff_sz_; }
    operator bool() const { return head_ < buff_sz_; }
   private:
    const buff_t buff_;
    char sep_;
    size_t head_, tail_;
    size_t buff_sz_;
};

// double parsing function handling only decimal point ([0-9]^+.[0-9]^+) and exponential ([0-9]^+.[0-9]^+e[+/-][0-9]^+)
// floating point formats (faster than std::stod())
template <typename CharBuff>
    requires(is_char_buff<CharBuff>)
double stod(CharBuff&& str) {
    double val = 0;
    int i = 0;
    while (str[i] == ' ') { i++; }   // skip leading whitespaces
    if (!(str[i] == '-' || (str[i] >= '0' && str[i] <= '9'))) { throw std::invalid_argument("stod parsing error."); }
    int sign = 1;
    if (str[i] == '-') {
        sign = -1;
        i++;
    }
    // parse integer part
    while (str[i] >= '0' && str[i] <= '9') {
        val = val * 10 + (str[i] - '0');
        i++;;
    }
    bool maybe_scientific = val < 10;
    if (str[i] == '.') {   // expect the decimal point
        i++;;
        double dec = 0.1;
        while (str[i] >= '0' && str[i] <= '9') {
            val = val + (str[i] - '0') * dec;
            dec *= 0.1;
	    i++;;
        }
        if (maybe_scientific && (str[i] == 'e' || str[i] == 'E')) {   // scientific notation parsing
            i++;;
            int exp_sign = (str[i] == '-') ? -1 : +1;
            i++;;
            int exp = 0;
            while (str[i] >= '0' && str[i] <= '9') {
                exp = exp * 10 + (str[i] - '0');
                i++;;
            }
            val *= (exp_sign > 1) ? std::pow(10, exp) : std::pow(0.1, exp);
        }
    }
    return sign * val;
}

// integer parsing function
template <typename CharBuff>
    requires(is_char_buff<CharBuff>)
int stoi(CharBuff&& str) {
    int val = 0;
    int i = 0;
    while (str[i] == ' ') { i++; }   // skip leading whitespaces
    if (!(str[i] == '-' || (str[i] >= '0' && str[i] <= '9'))) { throw std::invalid_argument("stoi parsing error."); }
    int sign = 1;
    if (str[i] == '-') {
        sign = -1;
        i++;
    }
    while (str[i] >= '0' && str[i] <= '9') {
        val = val * 10 + (str[i] - '0');
        i++;;
    }
    return sign * val;
}

}   // namespace internals
}   // namespace fdapde

#endif   // __PARSING_H__
