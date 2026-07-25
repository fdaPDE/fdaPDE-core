#include <fdaPDE/finite_elements.h>
#include <gtest/gtest.h>

namespace fdapde {

namespace {

template <typename Field>
concept supports_avg = requires(const Field& field) { avg(field); };

template <typename Field>
concept supports_average = requires(const Field& field) { average(field); };

template <typename Field>
concept supports_jump = requires(const Field& field) { jump(field); };

static_assert(!supports_avg<ScalarField<2>>);
static_assert(!supports_avg<MatrixField<2, 2, 2>>);
static_assert(!supports_average<ScalarField<2>>);
static_assert(!supports_average<MatrixField<2, 2, 2>>);
static_assert(!supports_jump<ScalarField<2>>);
static_assert(!supports_jump<MatrixField<2, 2, 2>>);

}   // namespace

TEST(dg_expression, scalar_traces_select_cell_support) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TestFunction v(space);
    TrialFunction u(space);
    static_assert(supports_avg<decltype(v)>);
    static_assert(supports_average<decltype(v)>);
    static_assert(supports_jump<decltype(v)>);
    internals::fe_assembler_packet<2> packet;
    packet.test_value(0) = 8.0;
    packet.trial_value(0) = 6.0;

    EXPECT_DOUBLE_EQ(v(packet), 8.0);
    EXPECT_DOUBLE_EQ(u(packet), 6.0);

    packet.interior_facet = true;
    packet.test_side = fe_facet_side::plus;
    packet.trial_side = fe_facet_side::minus;
    EXPECT_DOUBLE_EQ(v(packet), 0.0);
    EXPECT_DOUBLE_EQ(u(packet), 0.0);
    EXPECT_DOUBLE_EQ(avg(v)(packet), 4.0);
    EXPECT_DOUBLE_EQ(average(v)(packet), 4.0);
    EXPECT_DOUBLE_EQ(jump(v)(packet), 8.0);
    EXPECT_DOUBLE_EQ(avg(u)(packet), 3.0);
    EXPECT_DOUBLE_EQ(jump(u)(packet), -6.0);
    EXPECT_EQ(packet.trace_side, fe_facet_side::none);

    packet.test_side = fe_facet_side::minus;
    EXPECT_DOUBLE_EQ(jump(v)(packet), -8.0);
}

TEST(dg_expression, trace_semantics_propagate_to_derivatives_and_vectors) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace scalar_space(mesh, DG<1, 1>);
    TestFunction v(scalar_space);
    TrialFunction u(scalar_space);
    internals::fe_assembler_packet<2> scalar_packet;
    scalar_packet.interior_facet = true;
    scalar_packet.test_side = fe_facet_side::plus;
    scalar_packet.trial_side = fe_facet_side::minus;
    scalar_packet.test_grad(0, 0) = 4.0;
    scalar_packet.trial_grad(0, 0) = 10.0;

    EXPECT_DOUBLE_EQ(avg(grad(v)).eval(0, scalar_packet), 2.0);
    EXPECT_DOUBLE_EQ(jump(grad(v)).eval(0, scalar_packet), 4.0);
    EXPECT_DOUBLE_EQ(avg(grad(u)).eval(0, scalar_packet), 5.0);
    EXPECT_DOUBLE_EQ(jump(grad(u)).eval(0, scalar_packet), -10.0);

    FeSpace vector_space(mesh, DG<1, 2>);
    TestFunction vector_test(vector_space);
    static_assert(supports_avg<decltype(vector_test)>);
    static_assert(supports_average<decltype(vector_test)>);
    static_assert(supports_jump<decltype(vector_test)>);
    internals::fe_assembler_packet<2> vector_packet(2);
    vector_packet.interior_facet = true;
    vector_packet.test_side = fe_facet_side::plus;
    vector_packet.test_value(0) = 2.0;
    vector_packet.test_value(1) = 4.0;

    EXPECT_DOUBLE_EQ(vector_test.eval(0, vector_packet), 0.0);
    EXPECT_DOUBLE_EQ(avg(vector_test).eval(0, vector_packet), 1.0);
    EXPECT_DOUBLE_EQ(avg(vector_test).eval(1, vector_packet), 2.0);
    EXPECT_DOUBLE_EQ(jump(vector_test).eval(0, vector_packet), 2.0);
    EXPECT_DOUBLE_EQ(jump(vector_test).eval(1, vector_packet), 4.0);
}

TEST(dg_expression, facet_geometry_reads_packet_state_and_sets_dispatch_bit) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TestFunction v(space);
    TrialFunction u(space);
    internals::fe_assembler_packet<2> packet;
    packet.interior_facet = true;
    packet.test_side = fe_facet_side::plus;
    packet.trial_side = fe_facet_side::minus;
    packet.test_value(0) = 8.0;
    packet.trial_value(0) = 6.0;
    packet.facet_size = 0.25;
    packet.facet_normal(0, 0) = 1.0;
    packet.facet_normal(1, 0) = 0.0;

    auto h = facet_size(mesh);
    auto n = facet_normal(mesh);
    EXPECT_DOUBLE_EQ(h(packet), 0.25);
    EXPECT_DOUBLE_EQ(n.eval(0, packet), 1.0);
    EXPECT_DOUBLE_EQ(n.eval(1, packet), 0.0);

    auto form = h * jump(u) * jump(v);
    EXPECT_DOUBLE_EQ(form(packet), -12.0);
    constexpr int interior_facet_bit = int(fe_assembler_flags::interior_facet);
    EXPECT_NE(decltype(avg(v))::XprBits & interior_facet_bit, 0);
    EXPECT_NE(decltype(jump(u))::XprBits & interior_facet_bit, 0);
    EXPECT_NE(decltype(h)::XprBits & interior_facet_bit, 0);
    EXPECT_NE(decltype(n)::XprBits & interior_facet_bit, 0);
    EXPECT_NE(decltype(form)::XprBits & interior_facet_bit, 0);
}

}   // namespace fdapde
