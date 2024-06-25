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

#ifndef __SHAPEFILE_READER_H__
#define __SHAPEFILE_READER_H__

#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include "../assert.h"

namespace fdapde {

[[maybe_unused]] static struct tag_big_endian { } BigEndian;
[[maybe_unused]] static struct tag_little_endian { } LittleEndian;
  
// reads sizeof(T) bytes from buffer buff starting from the head-th byte in big-endian format
template <typename T> T read(char* buff, int& head, tag_big_endian) {
    char* result = new char[sizeof(T)];
    for (std::size_t j = 0; j < sizeof(T); ++j) { result[sizeof(T) - j - 1] = buff[head + j]; }
    head += sizeof(T);
    return *reinterpret_cast<T*>(result);
}
template <typename T> T read(char* buff, int& head, tag_little_endian) {
    char* result = new char[sizeof(T)];
    for (std::size_t j = 0; j < sizeof(T); ++j) { result[sizeof(T) - j - 1] = buff[head + (sizeof(T) - j - 1)]; }
    head += sizeof(T);
    return *reinterpret_cast<T*>(result);
}

class shp_reader {
    void skip(int n_bytes) { head_ += n_bytes; }   // move head_ n_bytes forward
  
    struct sf_header_t {
        static constexpr int size = 100;   // the size (in number of bytes) of the header
        int file_code;                     // file code: 9994
        int file_length;                   // total length of the file in 8-bit words
        int version;                       // shapefile version: 1000
        std::array<double, 8> bbox;        // x_min, y_min, x_max, y_max, z_min, z_max, m_min, m_max
        int shape_type;
    };
    struct sf_point_t {
        int record_number;
        int shape_type = ShapeType::Point;
        double x, y;

        sf_point_t() = default;
        sf_point_t(double x_, double y_) : x(x_), y(y_) { }
        sf_point_t(int record_number_, shp_reader& file) : record_number(record_number_) {
            // point specific content
            x = read<double>(file.buff_, file.head_, LittleEndian);
            y = read<double>(file.buff_, file.head_, LittleEndian);
        }
        friend std::ostream& operator<<(std::ostream& os, const sf_point_t& p) {
            os << "(" << p.x << "," << p.y << ")";
            return os;
        }
    };
    struct sf_polygon_t {
        int record_number;
        int shape_type = ShapeType::Polygon;
        std::array<double, 4> bbox;              // x_min, y_min, x_max, y_max
        int n_rings;                             // number of closed polygons in the record
        int n_points;                            // overall number of points
        std::vector<int> ring_begin, ring_end;   // first and last points in each ring, as offsets in points vector
        std::vector<sf_point_t> points;

        sf_polygon_t() = default;
        sf_polygon_t(int record_number_, shp_reader& file) : record_number(record_number_) {
            // polygon specific content
            for (int i = 0; i < 4; ++i) { bbox[i] = read<double>(file.buff_, file.head_, LittleEndian); }
            n_rings  = read<std::int32_t>(file.buff_, file.head_, LittleEndian);
            n_points = read<std::int32_t>(file.buff_, file.head_, LittleEndian);
            // the number of rings in the polygon
            ring_begin.resize(n_rings);
            for (int j = 0; j < n_rings; ++j) {
                ring_begin[j] = read<std::int32_t>(file.buff_, file.head_, LittleEndian);
            }
            ring_end.resize(n_rings);
            if (n_rings == 1) {
                ring_end[0] = n_points;
            } else {
                for (int j = 0; j < n_rings - 1; ++j) { ring_end[j] = ring_begin[j + 1] - 1; }
                ring_end[n_rings - 1] = n_points;
            }
            // store point coordinates
            points.reserve(n_points);
            for (int j = 0; j < n_points; ++j) {
                double x = read<double>(file.buff_, file.head_, LittleEndian);
                double y = read<double>(file.buff_, file.head_, LittleEndian);
                points.emplace_back(x, y);
            }
        }
        // iterator over rings
        class ring_iterator {
            const sf_polygon_t* polygon_;
            int index_;
            std::vector<sf_point_t>::const_iterator it;
           public:
            ring_iterator(const sf_polygon_t* polygon, int index) : polygon_(polygon), index_(index) {
                it = polygon_->points.begin();
            }
            std::vector<sf_point_t>::const_iterator begin() const { return it + polygon_->ring_begin[index_]; }
            std::vector<sf_point_t>::const_iterator end() const { return it + polygon_->ring_end[index_]; }
            const sf_point_t* operator->() { return it.operator->(); }
            const sf_point_t& operator*() { return *it; };
            ring_iterator& operator++() {
                index_++;
                return *this;
            }
            friend bool operator!=(const ring_iterator& it1, const ring_iterator& it2) {
                return it1.index_ != it2.index_;
            }
        };
        ring_iterator begin() const { return ring_iterator(this, 0); }
        ring_iterator end() const { return ring_iterator(this, n_rings); }
    };

    std::string file_name_;
    int n_records_;
    sf_header_t header_;     // shapefile header
    int head_ = 0;           // currently pointed byte in buff_
    char* buff_ = nullptr;   // loaded binary data
    std::variant<std::vector<sf_point_t>, std::vector<sf_polygon_t>> data_ {};
   public:
    // supported shapefile format
    // all the non-null shapes in a shapefile are required to be of the same shape type (cit. Shapefile standard)
    enum ShapeType {
        Null = 0, Point = 1, PolyLine = 3, Polygon = 5, MultiPoint = 8
    };
    shp_reader() = default;
    shp_reader(std::string file_name) : file_name_(file_name), n_records_(0), head_(0) {
        std::ifstream file;
        file.open(file_name, std::ios::in | std::ios::binary);
        if (file) {
            buff_ = new char[header_.size];   // read 100 bytes of header in buff_
            file.read(buff_, header_.size);
            header_.file_code = read<std::int32_t>(buff_, head_, BigEndian);
            skip(20);
            // file length: total length of the file in 16-bit words (including the fifty header's 16-bit words).
            header_.file_length = 2 * read<std::int32_t>(buff_, head_, BigEndian);   // store in 8 byte words
            header_.version = read<std::int32_t>(buff_, head_, LittleEndian);
            header_.shape_type = read<std::int32_t>(buff_, head_, LittleEndian);
            // shapefile bounding box
            for (int i = 0; i < 8; ++i) { header_.bbox[i] = read<double>(buff_, head_, LittleEndian); }
            head_ = 0;   // reset head_ pointer
            delete[] buff_;

	    // initialize variant
            if (header_.shape_type == ShapeType::Point)   { data_ = std::vector<sf_point_t>(); }
            if (header_.shape_type == ShapeType::Polygon) { data_ = std::vector<sf_polygon_t>(); }
            // read records
            buff_ = new char[header_.file_length - header_.size];
            file.read(buff_, header_.file_length - header_.size);
	    data_ = std::vector<sf_polygon_t>();
            while (head_ < header_.file_length - header_.size) {
                int record_number = read<std::int32_t>(buff_, head_, BigEndian);
                skip(4);   // skip content-length field (4 bytes)
                int shape_type = read<std::int32_t>(buff_, head_, LittleEndian);
		if (shape_type == ShapeType::Point)   { std::get<0>(data_).emplace_back(record_number, *this); }
                if (shape_type == ShapeType::Polygon) { std::get<1>(data_).emplace_back(record_number, *this); }
                n_records_++;
            }
            file.close();
            delete[] buff_;
        } else {
            std::cout << "unable to open file: " << file_name << "." << std::endl;
        }
    }
    // getters
    int shape_type() const { return header_.shape_type; }
    std::array<double, 4> bbox() const { return {header_.bbox[0], header_.bbox[1], header_.bbox[2], header_.bbox[3]}; }
    int n_records() const { return n_records_; }
    const sf_header_t& header() const { return header_; }
    const std::vector<sf_point_t>& points() const {
        fdapde_assert(shape_type() == ShapeType::Point);
        return std::get<0>(data_);
    }  
    const std::vector<sf_polygon_t>& polygons() const {
        fdapde_assert(shape_type() == ShapeType::Polygon);
        return std::get<1>(data_);
    }
};

// .dbf reader. file specification dBase level 5
class dbf_reader {
    void skip(int n_bytes) { head_ += n_bytes; }   // move head_ n_bytes forward

    struct field_descriptor {
        std::string name;   // column name
        char type;          // C: character, D: date, F: floating point, L: logical, N: numeric
        int length;
        field_descriptor() = default;
        field_descriptor(const std::string name_, char type_, int length_) :
            name(name_), type(type_), length(length_) { }
    };

    char* buff_;         // loaded binary data
    int head_ = 0;       // currently pointed byte in buff_
    std::string date_;   // date of last update in yyyy/mm/dd format
    std::vector<field_descriptor> fields_;
    std::unordered_map<std::string, std::vector<std::string>> data_;
    std::string file_name_;
   public:
    dbf_reader() = default;
    dbf_reader(std::string file_name) : file_name_(file_name) {
        std::ifstream file;
        file.open(file_name, std::ios::in | std::ios::binary);
        if (file) {
            buff_ = new char[32];   // read 32 bytes of header in buff_
            file.read(buff_, 32);

            skip(1); // skip first byte (dbf version number)
            date_ = std::to_string(1900 + read<std::int8_t>(buff_, head_, BigEndian)) + "/" +
                    std::to_string(read<std::int8_t>(buff_, head_, BigEndian)) + "/" +
                    std::to_string(read<std::int8_t>(buff_, head_, BigEndian));
            std::int32_t n_records     = read<std::int32_t>(buff_, head_, LittleEndian);
            std::int16_t header_length = read<std::int16_t>(buff_, head_, LittleEndian);
            std::int16_t record_length = read<std::int16_t>(buff_, head_, LittleEndian);
            skip(20);   // reserved
            int n_fields = (header_length - 32 - 1) / 32;
            // end of header
	    
            delete[] buff_;
            head_ = 0;
            buff_ = new char[n_fields * 32 + 1 + record_length * n_records];
            file.read(buff_, n_fields * 32 + 1 + record_length * n_records);
            // read field descriptors
            for (int i = 0; i < n_fields; ++i) {
                // first 11 bytes to contain field name
                std::string name;
                for (int i = 0; i < 11; ++i) { name += read<char>(buff_, head_, LittleEndian); }
                name.erase(std::find(name.begin(), name.end(), '\0'), name.end());   // remove \0 chars
                char type = read<char>(buff_, head_, LittleEndian);
                skip(4);    // reserved
                int length = read<std::int8_t>(buff_, head_, LittleEndian);
                skip(15);   // reserved
                fields_.emplace_back(name, type, length);
            }
            if (read<std::int8_t>(buff_, head_, LittleEndian) != 0x0D) {
                std::cout << "Error while reading .dbf file (bad header termination)." << std::endl;
                file.close();
                delete[] buff_;
                return;
            }
            // read records
            for (int i = 0; i < n_records; ++i) {
                skip(1);   // skip first byte (deletion flag)
                for (const auto& field : fields_) {
                    std::string parsed_field;
		    parsed_field.reserve(field.length);
                    for (int i = 0; i < field.length; ++i) { parsed_field += read<char>(buff_, head_, LittleEndian); }
		    data_[field.name].push_back(parsed_field);
                }
            }
            file.close();
            delete[] buff_;
        } else {
            std::cout << "unable to open file: " << file_name << "." << std::endl;
        }
    }
    template <typename T> std::vector<T> get_as(std::string colname) const {
        fdapde_assert(data_.count(colname) == 1);
        if constexpr (std::is_same<T, std::string>::value) {
            return data_.at(colname);
        } else {
            std::vector<T> values;
            T val {};
            for (const auto& v : data_.at(colname)) {
                std::from_chars(v.data(), v.data() + v.size(), val);
                values.push_back(val);
            }
            return values;
        }
    }
    std::vector<std::string> names() const {   // vector of fields names
        std::vector<std::string> result;
        for (const auto& field : fields_) { result.push_back(field.name); }
        return result;
    }
    std::vector<char> types() const {   // vector of fields types
        std::vector<char> result;
        for (const auto& field : fields_) { result.push_back(field.type); }
        return result;
    }
};

class ShapeFile {
   private:
    shp_reader shp_;
    dbf_reader dbf_;
    std::string gcs = "UNDEFINED";   // geographic coordinate system (GCS)
    std::string folder_name_;
   public:
    ShapeFile(std::string folder_name) : folder_name_(folder_name) {
        std::string file_name;
        std::size_t pos_start = 0, pos_end = 0;
        while ((pos_end = folder_name.find('/', pos_start)) != std::string::npos) { pos_start = pos_end + 1; }
        file_name = folder_name.substr(pos_start, pos_end - pos_start);
        // load geometric features and associated data
        shp_ = shp_reader(folder_name + "/" + file_name + ".shp");
        dbf_ = dbf_reader(folder_name + "/" + file_name + ".dbf");
        // retrieve GCS informations from .prj file
        std::ifstream prj_file;
        prj_file.open(folder_name + "/" + file_name + ".prj");
        if (prj_file) {
            std::string line;
            getline(prj_file, line);
            std::size_t i = line.find("GEOGCS", 0);
            i += std::string("GEOGCS[\"").size();
            std::size_t j = line.find("\"", i);
            gcs = line.substr(i, j - i);
        }
    }
    // getters
    const shp_reader& shp() const { return shp_; }
    template <typename T> std::vector<T> get_as(std::string colname) const { return dbf_.get_as<T>(colname); }
    std::vector<std::string> names() const { return dbf_.names(); }
    std::vector<char> types() const { return dbf_.types(); }
    friend std::ostream& operator<<(std::ostream& os, const ShapeFile& sf) {
        os << "file:              " << sf.folder_name_ << std::endl;
        std::string shape = "";
        switch (sf.shp().header().shape_type) {
        case int(shp_reader::ShapeType::Point):
            shape = "POINT";
            break;
        case int(shp_reader::ShapeType::Polygon):
            shape = "POLYGON";
            break;
        case int(shp_reader::ShapeType::PolyLine):
            shape = "POLYLINE";
            break;
        case int(shp_reader::ShapeType::MultiPoint):
            shape = "MULTIPOINT";
            break;
        }
        os << "shape_type:        " << shape << std::endl;
        os << "file size:         " << sf.shp().header().file_length * 2 << " Bytes" << std::endl;
        os << "number of records: " << sf.shp().n_records() << std::endl;
        os << "bounding box:      "
           << "(" << sf.shp().bbox()[0] << ", " << sf.shp().bbox()[1] << ", " << sf.shp().bbox()[2] << ", "
           << sf.shp().bbox()[3] << ")";
        os << "\n" << std::endl;
        std::vector<std::string> n_ = sf.names();
        std::vector<char> t_ = sf.types();
        for (std::size_t i = 0; i < n_.size() - 1; ++i) { os << n_[i] << "(" << t_[i] << "), "; }
        os << n_[n_.size() - 1] << "(" << t_[n_.size() - 1] << ")";
        return os;
    }
};

}   // namespace fdapde

#endif   // __SHAPEFILE_READER_H__
