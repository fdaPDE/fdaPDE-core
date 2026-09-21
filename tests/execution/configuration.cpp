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

/// @brief restores an environment variable after a parsing test
class EnvironmentVariableGuard {
   public:
    /// @brief saves the variable original value if present
    explicit EnvironmentVariableGuard(const char* name) : name_(name) {
        if (const char* value = std::getenv(name_)) { original_ = value; }
    }

    /// @brief disables copying of environment restoration ownership
    EnvironmentVariableGuard(const EnvironmentVariableGuard&) = delete;
    /// @brief disables assignment of environment restoration ownership
    EnvironmentVariableGuard& operator=(const EnvironmentVariableGuard&) = delete;

    /// @brief restores the original variable value or absence
    ~EnvironmentVariableGuard() {
        if (original_) {
            static_cast<void>(setenv(name_, original_->c_str(), 1));
        } else {
            static_cast<void>(unsetenv(name_));
        }
    }

    /// @brief sets the variable and verifies the operating system call succeeded
    void set(const char* value) {
        // checks setenv succeeded before testing environment parsing
        ASSERT_EQ(setenv(name_, value, 1), 0);
    }
    /// @brief removes the variable and verifies the operating system call succeeded
    void unset() {
        // checks unsetenv succeeded before testing absence handling
        ASSERT_EQ(unsetenv(name_), 0);
    }
   private:
    const char* name_;
    std::optional<std::string> original_;
};

}   // namespace

// verifies strict environment parsing and thread count freeze
TEST(ExecutionConfiguration, StrictEnvironmentParsingAndThreadCountFreeze) {
    constexpr const char* variable = "FDAPDE_TEST_CONCURRENCY";
    EnvironmentVariableGuard environment(variable);

    environment.unset();
    // checks an absent variable produces no configured count
    EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), std::nullopt);

    environment.set("4");
    // checks the complete decimal string is parsed as four workers
    EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), 4u);
    for (const char* invalid : {"", "0", "-1", "+4", " 4", "4 ", "4x"}) {
        environment.set(invalid);
        // checks malformed or nonpositive strings are rejected without partial parsing
        EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), std::nullopt) << invalid;
    }
    const std::string overflow = std::to_string(std::numeric_limits<std::size_t>::max()) + "0";
    environment.set(overflow.c_str());
    // checks a decimal value exceeding size_t produces no count
    EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), std::nullopt);
    const std::string above_int = std::to_string(static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1);
    environment.set(above_int.c_str());
    // checks a positive value above int range produces no count
    EXPECT_EQ(fdapde::internals::get_env_concurrency_count(variable), std::nullopt);

    // checks the public setter rejects zero workers
    EXPECT_THROW(fdapde::parallel_set_num_threads(0), std::invalid_argument);
    // checks the public setter rejects negative workers
    EXPECT_THROW(fdapde::parallel_set_num_threads(-1), std::invalid_argument);
    fdapde::parallel_set_num_threads(2);
    // checks the accepted worker count is visible before initialization
    EXPECT_EQ(fdapde::parallel_get_num_threads(), 2);

    auto initialized = fdapde::parallel_async([] { return fdapde::this_thread_id(); });
    // checks asynchronous work runs on a worker with a nonnegative logical id
    EXPECT_GE(initialized.get(), 0);
    // checks the worker count is frozen after the first asynchronous submission
    EXPECT_THROW(fdapde::parallel_set_num_threads(3), std::logic_error);
    // checks the rejected update leaves the configured count unchanged
    EXPECT_EQ(fdapde::parallel_get_num_threads(), 2);
}
