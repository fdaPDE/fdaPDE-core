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

#ifndef __POLYGON_H__
#define __POLYGON_H__

namespace fdapde {
namespace core {

  // a polygon is an ordered set of edges
  // we assume points already sorted counter-clockwise
  
class Polygon {
private:
  DMatrix<double> points_;
public:
  Polygon() = default;
  Polygon(const DMatrix<double>& points) : points_(points) { fdapde_assert(points.cols() == 2); }
  template <typename InputIterator> Polygon(InputIterator first, InputIterator last) {
      int i = 0;
      points_.resize(std::distance(last, first), 2);
      for (InputIterator it = first; it != last; ++it) { points_.row(i++) = *it; }
  }
  
  

};

}   // namespace core
}   // namespace fdapde

#endif // __POLYGON_H__
