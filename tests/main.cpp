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

#include <gtest/gtest.h>   // testing framework
#include <cstring>

class fdapde_testing_printer : public ::testing::EmptyTestEventListener {
    int n_tests_ = 0;       // overall number of tests
    int current_ = 0;       // current executed test
    int suite_width_ = 0;   // maximum suite name length
    int test_width_ = 0;    // maximum test name length
    std::chrono::steady_clock::time_point test_start;
    const int gap_before_status_ = 5;

    std::string timestamp_() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        struct tm buf;
        localtime_r(&t, &buf);
        std::ostringstream ss;
        ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
   public:
    fdapde_testing_printer() {
        auto* unit = ::testing::UnitTest::GetInstance();
        n_tests_ = unit->test_to_run_count();
    }
    void OnTestProgramStart(const ::testing::UnitTest& unit) override {
        n_tests_ = unit.test_to_run_count();
        // compute max widths
        for (int i = 0; i < unit.total_test_suite_count(); ++i) {
            const ::testing::TestSuite* suite = unit.GetTestSuite(i);
            for (int j = 0; j < suite->total_test_count(); ++j) {
                const ::testing::TestInfo* info = suite->GetTestInfo(j);
                if (info->should_run()) {
                    suite_width_ = std::max(suite_width_, (int)std::strlen(info->test_suite_name()));
                    test_width_ = std::max(test_width_, (int)std::strlen(info->name()));
                }
            }
        }
    }
    void OnTestStart(const ::testing::TestInfo&) override {
        ++current_;
        test_start = std::chrono::steady_clock::now();
    }

    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        auto test_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = test_end - test_start;
        double elapsed = diff.count();

        std::string ts = timestamp_();
        // build test info string
        std::ostringstream row;
        row << "[" << ts << "] " << std::setw(3) << current_ << "/" << n_tests_ << "   \033[1m" << std::left
            << std::setw(suite_width_) << test_info.test_suite_name() << "\033[0m  " << std::left
            << std::setw(test_width_) << test_info.name();
        row << std::string(gap_before_status_, ' ');
        if (test_info.result()->Passed()) {
            row << "\033[32mPASSED\033[0m";
        } else {
            row << "\033[31mFAILED\033[0m";
        }
        const int time_field_width = 8;
        row << std::string(2, ' ') << std::right << std::setw(time_field_width) << std::fixed << std::setprecision(3)
            << elapsed;
        std::cout << row.str() << "\n";
        // print test failure informations
        if (!test_info.result()->Passed()) {
            for (int i = 0; i < test_info.result()->total_part_count(); ++i) {
                auto part = test_info.result()->GetTestPartResult(i);
                if (part.failed()) {
                    std::cout << part.file_name() << ":" << part.line_number() << "\n" << part.summary() << "\n";
                }
            }
        }
    }
    void OnTestProgramEnd(const ::testing::UnitTest& unit) override {
        int failed = unit.failed_test_count();
        int passed = unit.successful_test_count();
        std::string status = failed > 0 ? "FAILED" : "PASSED";
        std::string color = failed > 0 ? "\033[31m" : "\033[32m";

        std::cout << "\n" << color << "Status: " << status << "\033[0m ";
        std::cout << "  [Tests: " << n_tests_ << " | Passed: " << passed << " | Failed: " << failed << "] ";
        std::cout << "(Elapsed: " << std::fixed << std::setprecision(3) << unit.elapsed_time() / 1000.0 << " s)\n";
    }
};

#include "execution/parallel_algorithms.cpp"
#include "execution/queues.cpp"
#include "execution/task_graphs.cpp"
#include "geometry/triangle.cpp"
#include "linear_algebra/block.cpp"
#include "linear_algebra/bool_vector.cpp"
#include "linear_algebra/diagonal.cpp"
#include "linear_algebra/gmres.cpp"
#include "linear_algebra/matrix.cpp"
#include "linear_algebra/symmetric.cpp"
#include "linear_algebra/triangular.cpp"
#include "linear_algebra/xpr.cpp"

int main(int argc, char** argv) {
    // start testing
    testing::InitGoogleTest(&argc, argv);

    // install pretty-printer
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new fdapde_testing_printer);

    return RUN_ALL_TESTS();
}
