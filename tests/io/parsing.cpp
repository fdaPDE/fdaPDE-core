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

#include <fdaPDE/io.h>
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
/// @brief owns an isolated CSV fixture without changing the process working directory
class CsvFixture {
   public:
    /// @brief writes known finite binary fractions into a fresh temporary directory
    CsvFixture() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto stem = std::filesystem::temp_directory_path() / ("fdapde-csv-path-" + std::to_string(stamp));
        for (int suffix = 0;; ++suffix) {
            directory_ = stem.string() + "-" + std::to_string(suffix);
            if (std::filesystem::create_directory(directory_)) break;
        }
        std::ofstream file(directory_ / "values.csv");
        if (!file) throw std::runtime_error("cannot write CSV fixture");
        file << "left,right\n1.25,2.5\n-3,4.75\n0.125,-0.5\n";
    }
    /// @brief removes only this fixture while keeping cleanup nonthrowing
    ~CsvFixture() {
        std::error_code ignored;
        std::filesystem::remove_all(directory_, ignored);
    }
    /// @brief returns an absolute fixture path for an existing or missing file
    std::filesystem::path absolute(const char* name = "values.csv") const {
        return std::filesystem::absolute(directory_ / name);
    }
    /// @brief returns a fixture path relative to the unchanged caller working directory
    std::filesystem::path relative(const char* name = "values.csv") const {
        return std::filesystem::relative(absolute(name), std::filesystem::current_path());
    }
   private:
    std::filesystem::path directory_;
};

/// @brief compares parsed dimensions and ordered values with the literal CSV oracle
template <typename Table> void expect_fixture_values(const Table& table) {
    // the three literal data lines must produce exactly three rows
    EXPECT_EQ(table.rows(), 3);
    // neither path spelling may skip the first of the two explicit columns
    EXPECT_EQ(table.cols(), 2);
    // binary fractions and signs must match the row-major values written independently above
    EXPECT_EQ(table.data(), (std::vector<double> {1.25, 2.5, -3.0, 4.75, .125, -.5}));
    // both named columns must retain their written order with index_col disabled
    EXPECT_EQ(table.colnames(), (std::vector<std::string> {"left", "right"}));
}
}   // namespace

// verifies relative and absolute public CSV reads preserve the same independently specified table
TEST(IoParsing, CsvAcceptsRelativeAndAbsolutePaths) {
    const CsvFixture fixture;
    const auto relative = fixture.relative(), absolute = fixture.absolute();
    // the relative oracle must exercise the caller-relative path contract
    ASSERT_TRUE(relative.is_relative());
    // the absolute oracle must exercise the path case formerly prefixed with the working directory
    ASSERT_TRUE(absolute.is_absolute());

    expect_fixture_values(fdapde::read_csv<double>(relative.string(), true, false));
    expect_fixture_values(fdapde::read_csv<double>(absolute.string(), true, false));
}

// verifies public CSV reads report missing relative and absolute files through the same native exception
TEST(IoParsing, CsvRejectsMissingRelativeAndAbsolutePaths) {
    const CsvFixture fixture;
    const auto relative = fixture.relative("missing.csv"), absolute = fixture.absolute("missing.csv");
    // no file is created at this name so the relative input must report a missing-file runtime error
    EXPECT_THROW(static_cast<void>(fdapde::read_csv<double>(relative.string(), true, false)), std::runtime_error);
    // the same nonexistent absolute path must report the native missing-file exception
    EXPECT_THROW(static_cast<void>(fdapde::read_csv<double>(absolute.string(), true, false)), std::runtime_error);
}
