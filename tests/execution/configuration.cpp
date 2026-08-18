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
#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

class EnvironmentVariableGuard {
   public:
    explicit EnvironmentVariableGuard(const char* name) : name_(name) {
        if (const char* value = std::getenv(name_)) { original_ = value; }
    }

    EnvironmentVariableGuard(const EnvironmentVariableGuard&) = delete;
    EnvironmentVariableGuard& operator=(const EnvironmentVariableGuard&) = delete;

    ~EnvironmentVariableGuard() {
        if (original_) {
            static_cast<void>(setenv(name_, original_->c_str(), 1));
        } else {
            static_cast<void>(unsetenv(name_));
        }
    }

    void set(const char* value) { ASSERT_EQ(setenv(name_, value, 1), 0); }
    void unset() { ASSERT_EQ(unsetenv(name_), 0); }
   private:
    const char* name_;
    std::optional<std::string> original_;
};

}   // namespace

TEST(ExecutionConfiguration, StrictEnvironmentParsingAndThreadCountFreeze) {
    constexpr const char* variable = "FDAPDE_TEST_CONCURRENCY";
    EnvironmentVariableGuard environment(variable);

    environment.unset();
    EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), std::nullopt);

    environment.set("4");
    EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), 4u);
    for (const char* invalid : {"", "0", "-1", "+4", " 4", "4 ", "4x"}) {
        environment.set(invalid);
        EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), std::nullopt) << invalid;
    }
    const std::string overflow = std::to_string(std::numeric_limits<std::size_t>::max()) + "0";
    environment.set(overflow.c_str());
    EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), std::nullopt);
    const std::string above_int = std::to_string(static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1);
    environment.set(above_int.c_str());
    EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), std::nullopt);

    EXPECT_THROW(fdapde::parallel_set_num_threads(0), std::invalid_argument);
    EXPECT_THROW(fdapde::parallel_set_num_threads(-1), std::invalid_argument);
    fdapde::parallel_set_num_threads(2);
    EXPECT_EQ(fdapde::parallel_get_num_threads(), 2);

    auto initialized = fdapde::parallel_async([] { return fdapde::this_thread_id(); });
    EXPECT_GE(initialized.get(), 0);
    EXPECT_THROW(fdapde::parallel_set_num_threads(3), std::logic_error);
    EXPECT_EQ(fdapde::parallel_get_num_threads(), 2);
}
