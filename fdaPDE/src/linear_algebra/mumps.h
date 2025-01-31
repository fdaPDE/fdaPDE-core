#ifndef __MUMPS_H__
#define __MUMPS_H__

#include <dmumps_c.h>
#include <mpi.h>

#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "../../utility.h"

using namespace Eigen;

namespace fdapde {

namespace mumps {

// concepts
template <typename T>
concept isEigenSparseMatrix = std::derived_from<T, Eigen::SparseMatrixBase<T>>;

template <typename Container, typename T>
concept isStdContainerOf = requires(Container c) {
    typename Container::value_type;
    typename Container::iterator;
    typename Container::size_type;
    { c.begin() } -> std::input_iterator;
    { c.end() } -> std::input_iterator;
    { c.size() } -> std::integral;
} && std::is_same_v<typename Container::value_type, T>;

template <typename Iterator, typename T>
concept isStdIteratorOf =
  std::input_iterator<Iterator> && std::is_same_v<typename std::iterator_traits<Iterator>::value_type, T>;

// SINGLETON class for MPI initialization and finalization [!!!!TEMPORARY!!!!]
class MPI_Manager {
   public:
    static MPI_Manager& getInstance() {
        static MPI_Manager instance;
        return instance;
    }

    MPI_Manager(const MPI_Manager&) = delete;
    MPI_Manager& operator=(const MPI_Manager&) = delete;
    MPI_Manager(MPI_Manager&&) = delete;
    MPI_Manager& operator=(MPI_Manager&&) = delete;

    bool isMPIinitialized() const { return mpi_initialized; }
   private:
    bool mpi_initialized;

    MPI_Manager() : mpi_initialized(false) {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(NULL, NULL);
            mpi_initialized = true;
        }
    }

    ~MPI_Manager() {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized && mpi_initialized) { MPI_Finalize(); }
    }
};

template <typename T, int Size> class MumpsParameterArray {
   public:
    MumpsParameterArray() : mp_data(nullptr) { }
    MumpsParameterArray(T* data_ptr) : mp_data(data_ptr) { }

    inline T* data() { return mp_data; }
    inline const T* data() const { return mp_data; }
    inline size_t size() const { return Size; }

    inline T& operator[](size_t i) {
        fdapde_assert(i >= 0 && "Index must be positive");
        fdapde_assert(i < Size && "Index out of bounds");
        return mp_data[i];
    }

    inline const T& operator[](size_t i) const {
        fdapde_assert(i >= 0 && "Index must be positive");
        fdapde_assert(i < Size && "Index out of bounds");
        return mp_data[i];
    }
   protected:
    T* mp_data;
};

// MUMPS FLAGS

// Base class flags
constexpr unsigned int NoDeterminant = (1 << 0);
constexpr unsigned int Verbose = (1 << 1);
constexpr unsigned int WorkingHost = (1 << 2);

// BLR flags
constexpr unsigned int UFSC = (1 << 3);   // default
constexpr unsigned int UCFS = (1 << 4);
constexpr unsigned int Compressed = (1 << 5);

// FORWARD DECLARATIONS
template <isEigenSparseMatrix MatrixType_> class MumpsLU;
template <isEigenSparseMatrix MatrixType_, int Options = Upper> class MumpsLDLT;
template <isEigenSparseMatrix MatrixType_> class MumpsBLR;
template <isEigenSparseMatrix MatrixType_> class MumpsRR;
template <isEigenSparseMatrix MatrixType_> class MumpsSchur;

namespace internal {

template <class Derived> struct mumps_traits;

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsLU<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsLDLT<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsBLR<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsRR<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

template <isEigenSparseMatrix MatrixType_> struct mumps_traits<MumpsSchur<MatrixType_>> {
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType_::Scalar;
    using RealScalar = typename MatrixType_::RealScalar;
    using StorageIndex = typename MatrixType_::StorageIndex;
};

}   // namespace internal

// MUMPS BASE CLASS
template <class Derived> class MumpsBase : public SparseSolverBase<Derived> {
   protected:
    using Base = SparseSolverBase<Derived>;
    using Base::derived;
    using Base::m_isInitialized;

    using Traits = internal::mumps_traits<Derived>;
   public:
    using Base::_solve_impl;

    using MatrixType = typename Traits::MatrixType;
    using Scalar = typename Traits::Scalar;
    using RealScalar = typename Traits::RealScalar;
    using StorageIndex = typename Traits::StorageIndex;
    using MumpsIcntlArray = MumpsParameterArray<int, 60>;
    using MumpsCntlArray = MumpsParameterArray<double, 15>;
    using MumpsInfoArray = MumpsParameterArray<int, 80>;
    using MumpsRinfoArray = MumpsParameterArray<double, 40>;
    enum {
        ColsAtCompileTime = MatrixType::ColsAtCompileTime,
        MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

    inline Index cols() const { return m_size; }
    inline Index rows() const { return m_size; }

    inline int getProcessRank() const { return m_mpiRank; }
    inline int getProcessSize() const { return m_mpiSize; }

    inline MumpsIcntlArray& mumpsIcntl() { return m_mumpsIcntl; }
    inline MumpsCntlArray& mumpsCntl() { return m_mumpsCntl; }

    inline const MumpsIcntlArray& mumpsIcntl() const { return m_mumpsIcntl; }
    inline const MumpsCntlArray& mumpsCntl() const { return m_mumpsCntl; }
    inline const MumpsInfoArray& mumpsInfo() const { return m_mumpsInfo; }
    inline const MumpsInfoArray& mumpsInfog() const { return m_mumpsInfog; }
    inline const MumpsRinfoArray& mumpsRinfo() const { return m_mumpsRinfo; }
    inline const MumpsRinfoArray& mumpsRinfog() const { return m_mumpsRinfog; }

    // setter & getter for the raw mumps struct, tinkering with it is unadvides unless the user has an already good
    // understanding of mumps and this wrapper
    // [MSG for A.Palummo] Ho deciso di aggiungere questi getter in modo da permettere a persone più esperte di poter
    // accedere direttamente alla struttura di MUMPS, in modo da poter settare parametri non accessibili tramite i
    // metodi pubblici della classe. Questo permette di accedere a membri della struttura che non sono utilizzati da
    // questa classe che potrebbero però essere utili nel caso vengano modificati manualmente alcuni icntl o cntl.
    inline DMUMPS_STRUC_C& mumpsRawStruct() { return m_mumps; }
    inline const DMUMPS_STRUC_C& mumpsRawStruct() const { return m_mumps; }

    Derived& analyzePattern(const MatrixType& matrix) {
        define_matrix(matrix);
        analyzePattern_impl();
        return derived();
    }

    Derived& factorize(const MatrixType& matrix) {
        define_matrix(matrix);
        factorize_impl();
        return derived();
    }

    Derived& compute(const MatrixType& matrix) {
        define_matrix(matrix);
        compute_impl();
        return derived();
    }

    template <typename BDerived, typename XDerived>
    void _solve_impl(const Eigen::DenseBase<BDerived>& b, Eigen::DenseBase<XDerived>& x) const {
        fdapde_assert(m_factorizationIsOk && "The matrix must be factorized with factorize() before calling solve()");

        if (b.derived().data() == x.derived().data()) {   // inplace solve
            fdapde_assert((BDerived::Flags & Eigen::RowMajorBit) == 0 && "Inplace solve doesn't support row-major rhs");
            if (getProcessRank() == 0) {
                m_mumps.nrhs = b.cols();
                m_mumps.lrhs = b.rows();
                m_mumps.rhs = const_cast<Scalar*>(b.derived().data());
            }
            mumps_execute(3);   // 3: solve
        }

        else {
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> buff;   // mumps requires the rhs to be ColMajor
            if (getProcessRank() == 0) {
                buff = b;
                m_mumps.nrhs = buff.cols();
                m_mumps.lrhs = buff.rows();
                m_mumps.rhs = const_cast<Scalar*>(buff.data());
            }
            mumps_execute(3);   // 3: solve
            x.derived().resizeLike(b);
            if (getProcessRank() == 0) { x.derived() = buff; }
        }

        MPI_Bcast(x.derived().data(), x.size(), MPI_DOUBLE, 0, m_mpiComm);
    }

    // the following method implements sparse to sparse (keep in mind that it is possible to assign a sparse matrix to a
    // dense one, so this method will work for sparse to dense as well)
    template <typename BDerived, typename XDerived>
    void _solve_impl(const Eigen::SparseMatrixBase<BDerived>& b, Eigen::SparseMatrixBase<XDerived>& x) const {
        fdapde_assert(m_factorizationIsOk && "The matrix must be factorized with factorize() before calling solve()");

        Eigen::SparseMatrix<Scalar, Eigen::ColMajor> loc_b;

        std::vector<StorageIndex> irhs_ptr;   // Stores the indices of the non zeros in valuePts that start a new column
                                              // like innerIndexPtr does but requires 1-based indexing and an extra
                                              // element at the end equal to the number of nonzeros + 1
        std::vector<StorageIndex> irhs_sparse;   // Stores the row indices of the nonzeros like outerIndexPtr does, but
                                                 // requires 1-based indexing

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> res(b.rows(), b.cols());

        if (getProcessRank() == 0) {
            loc_b = b;   // I save a local copy because I need the matrix in compressed format in order for the accesses
                         // to the needed pointers to be correct (not needed in define_matrix for example because there
                         // i use the InnerIterator)
            loc_b.makeCompressed();

            irhs_ptr.reserve(loc_b.cols() + 1);
            for (StorageIndex i = 0; i < loc_b.cols(); ++i) {   // Already shifts to 1-based indexing
                irhs_ptr.push_back(loc_b.outerIndexPtr()[i] + 1);
            }
            irhs_ptr.push_back(loc_b.nonZeros() + 1);   // Mumps needs an extra value

            irhs_sparse.reserve(loc_b.nonZeros());
            for (StorageIndex i = 0; i < loc_b.nonZeros(); ++i) {   // Already shifts to 1-based indexing
                irhs_sparse.push_back(loc_b.innerIndexPtr()[i] + 1);
            }

            m_mumps.nz_rhs = loc_b.nonZeros();
            m_mumps.nrhs = loc_b.cols();
            m_mumps.lrhs = loc_b.rows();
            m_mumps.rhs_sparse = loc_b.valuePtr();
            m_mumps.irhs_sparse = irhs_sparse.data();
            m_mumps.irhs_ptr = irhs_ptr.data();
            m_mumps.rhs = const_cast<Scalar*>(res.data());
        }

        m_mumps.icntl[19] = 1;   // 1: sparse right-hand side
        mumps_execute(3);
        m_mumps.icntl[19] = 0;   // reset to default

        MPI_Bcast(res.data(), res.size(), MPI_DOUBLE, 0, m_mpiComm);

        // Unfortunately Mumps only outputs the solution in dense format, i need to make sparse to adhere to Solve.h
        // (only Dense2Dense or Sparse2Sparse are allowed)
        if (XDerived::Flags & Eigen::RowMajorBit) {
            Eigen::SparseMatrix<Scalar, Eigen::RowMajor> loc_x;
            loc_x = res.sparseView();
            x = loc_x;
        } else {
            x = res.sparseView();
        }
    }

    // template <typename BDerived, typename XDerived>
    // void solveSparse(const Eigen::SparseMatrix<BDerived>& b, Eigen::SparseMatrixBase<XDerived>& x) const {
    //     Eigen::SparseMatrix<Scalar, ColMajor> loc_b;
    //     Eigen::SparseMatrix<Scalar, ColMajor> loc_x;
    //     if (BDerived::Flags & RowMajorBit || XDerived::Flags & RowMajorBit) {
    //         loc_b = b;
    //         loc_b.makeCompressed();
    //         loc_x = x;
    //         loc_x.makeCompressed();
    //         loc_x = solve(loc_b);
    //         x = loc_x;
    //     } else {
    //         x = solve(b);
    //     }
    // }

    Scalar determinant() const {
        fdapde_assert(m_computeDeterminant && "The determinant computation must be enabled");
        fdapde_assert(
          m_factorizationIsOk && "The matrix must be factoried with factorize() before calling determinant()");
        return mumpsRinfog()[11] *
               std::pow(2, mumpsInfog()[33]);   // Keep in mind that if infog[33] is very high, std::pow will return inf
    }

    template <typename Container>
        requires isStdContainerOf<Container, std::pair<int, int>>
    std::vector<Eigen::Triplet<Scalar>> inverseElements(const Container& elements) {
        return inverseElements(elements.begin(), elements.end());
    }

    template <typename Iterator>
        requires isStdIteratorOf<Iterator, std::pair<int, int>>
    std::vector<Eigen::Triplet<Scalar>> inverseElements(Iterator begin, Iterator end) {
        fdapde_assert(
          mumpsIcntl()[18] == 0 &&
          "Incompatible with Schur complement");   // The Mumps icntls 18 and 29 are incompatible
        fdapde_assert(
          m_factorizationIsOk && "The matrix must be factorized with factorize() before calling inverseElements()");

        std::vector<std::pair<int, int>> elements(begin, end);

        fdapde_assert(elements.size() > 0 && "The container cannot be empty");

        // reored elements by columns
        std::sort(elements.begin(), elements.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            if (a.second != b.second) {
                return a.second < b.second;   // Compare by second element first
            }
            return a.first < b.first;   // If second elements are equal, compare by first element
        });

        fdapde_assert(   // check if the selected elements are unique (the adjacent_find could be done outside the
                         // assert and save the ... == ... to a bool to then pun into the assert, I don't know
                         // wheteher it would be peferred)
          std::adjacent_find(
            elements.begin(), elements.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.first == b.first && a.second == b.second;
            }) == elements.end() &&
          "The selected elements must be unique");

        for (auto& elem : elements) {
            fdapde_assert(elem.first >= 0 && elem.first < m_size && "The selected rows must be within the matrix size");
            fdapde_assert(
              elem.second >= 0 && elem.second < m_size && "The selected columns must be within the matrix size");
        }

        std::vector<Eigen::Triplet<Scalar>> inv_elements;
        inv_elements.reserve(elements.size());

        std::vector<int> irhs_ptr;   // Stores the indices of the fisrt requested element of each column, requires
                                     // 1-based indexing and an extra element at the end equal to the number of columns
                                     // + 1. It should have size = number of columns + 1
                                     // In general, if a column doesn't have any requested element, the respective entry
                                     // should be equal to the next column's first requested element (if this is empty
                                     // as well the process should be repeated until a column with requested elements is
                                     // found and that index is used for the empty columns preceeding it as well)
        std::vector<int> irhs_sparse;     // Stores the row indices of the requested elements, requires 1-based indexing
        std::vector<Scalar> rhs_sparse;   // Stores the inverse of the requested elements
        rhs_sparse.resize(elements.size());

        if (getProcessRank() == 0) {
            irhs_ptr.reserve(m_size + 1);
            for (int i = 0; i <= elements.front().second; ++i) {
                irhs_ptr.push_back(1);
            }   // i nedd to fill the vector with 1 until the column of the first requested element (included)
            for (size_t i = 1; i < elements.size(); ++i) {
                if (elements[i].second != elements[i - 1].second) {
                    for (int j = 0; j < elements[i].second - elements[i - 1].second; ++j) { irhs_ptr.push_back(i + 1); }
                }
            }
            // When the column changes, I need to insert into the vector the index of the first requested element of
            //  the new column, if there are empty columns I need to fill the vector with the index of the first
            //  requested element of the next column, with as many elements as the number of empty columns before the
            //  next column with requested elements (alredy shifted to 1-based indexing)
            for (int i = elements.back().second; i <= m_size; ++i) {
                irhs_ptr.push_back(elements.size() + 1);
            }   // Adding the last entry as mumps requires (number of columns + 1 in position of the last column + 1)

            irhs_sparse.reserve(elements.size());
            for (const auto& elem : elements) {
                irhs_sparse.push_back(elem.first + 1);
            }   // already scales the row indices to 1-based indexing

            m_mumps.nz_rhs = elements.size();
            m_mumps.nrhs = m_size;
            m_mumps.lrhs = m_size;
            m_mumps.irhs_ptr = irhs_ptr.data();
            m_mumps.irhs_sparse = irhs_sparse.data();
            m_mumps.rhs_sparse = rhs_sparse.data();
        }

        mumpsIcntl()[29] = 1;   // 1: compute selected elements of the inverse
        mumps_execute(3);
        mumpsIcntl()[29] = 0;   // reset to default

        MPI_Bcast(rhs_sparse.data(), rhs_sparse.size(), MPI_DOUBLE, 0, m_mpiComm);

        for (size_t i = 0; i < elements.size(); ++i) {
            inv_elements.emplace_back(elements[i].first, elements[i].second, rhs_sparse[i]);
        }   // the elements indices were not changed, so I can just copy them from the input vector (0-based indexing)

        return inv_elements;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> inverse() {
        fdapde_assert(m_factorizationIsOk && "The matrix must be factorized with factorize() before calling inverse()");
        fdapde_assert(mumpsIcntl()[18] == 0 && "Incompatible with Schur complement");

        std::vector<StorageIndex> irhs_ptr;   // Stores the indices of the non zeros in valuePts that start a new column
                                              // like innerIndexPtr does but requires 1-based indexing and an extra
                                              // element at the end equal to the number of nonzeros + 1
        std::vector<StorageIndex> irhs_sparse;   // Stores the row indices of the nonzeros like outerIndexPtr does, but
                                                 // requires 1-based indexing
        std::vector<Scalar> rhs_sparse;          // Stores the values of the identity matrix

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> inv(m_size, m_size);

        if (getProcessRank() == 0) {
            irhs_ptr.resize(m_size + 1);
            std::iota(irhs_ptr.begin(), irhs_ptr.end(), 1);   // 1-based indexing

            irhs_sparse.resize(m_size);
            std::iota(irhs_sparse.begin(), irhs_sparse.end(), 1);   // 1-based indexing

            rhs_sparse.resize(m_size, 1);

            m_mumps.nz_rhs = m_size;
            m_mumps.nrhs = m_size;
            m_mumps.lrhs = m_size;
            m_mumps.rhs_sparse = rhs_sparse.data();
            m_mumps.irhs_sparse = irhs_sparse.data();
            m_mumps.irhs_ptr = irhs_ptr.data();
            m_mumps.rhs = const_cast<Scalar*>(inv.data());
        }

        m_mumps.icntl[19] = 1;   // 1: sparse right-hand side
        mumps_execute(3);
        m_mumps.icntl[19] = 0;   // reset to default

        MPI_Bcast(inv.data(), inv.size(), MPI_DOUBLE, 0, m_mpiComm);

        return inv;
    }

    void mumpsFinalize() {
        mumps_execute(-2);   // -2: finalize
        m_mumpsFinalized = true;
    }

    ComputationInfo info() const {
        fdapde_assert(m_isInitialized && "Decomposition is not initialized.");
        return m_info;
    }
   protected:
    MumpsBase(MPI_Comm comm, unsigned int flags) :
        m_size(0),
        m_mpiComm(comm),
        m_info(InvalidInput),
        m_analysisIsOk(false),
        m_factorizationIsOk(false),
        m_mumpsFinalized(false),
        m_mumpsIcntl(m_mumps.icntl),
        m_mumpsCntl(m_mumps.cntl),
        m_mumpsInfo(m_mumps.info),
        m_mumpsInfog(m_mumps.infog),
        m_mumpsRinfo(m_mumps.rinfo),
        m_mumpsRinfog(m_mumps.rinfog) {
        m_verbose = (flags & Verbose) ? true : false;                     // default is non verbose
        m_computeDeterminant = !(flags & NoDeterminant) ? true : false;   // default is to compute the determinant
        m_mumps.par = (flags & WorkingHost) ? 1 : 0;                      // default is delegating host (par = 0)

        MPI_Manager::getInstance();   // initialize MPI if not already done, getInstance() will create th singleton only
                                      // if it doesn't exist (static function (always available even if the class
                                      // doesn't have an instance yet) that creates a static object)

        MPI_Comm_rank(m_mpiComm, &m_mpiRank);
        MPI_Comm_size(m_mpiComm, &m_mpiSize);

        m_mumps.comm_fortran = (MUMPS_INT)MPI_Comm_c2f(m_mpiComm);

        m_isInitialized = false;
    }

    virtual ~MumpsBase() { }

    // [MSG for A.Palummo] Per quanto riguarda la versione "iper-parallelizzata" la trova commantata dopo il metodo
    // define_matrix, ho fatto un paio di prove veloci con delle matrici prese da matrix market ma non ho riscontrato
    // particolari differenze in termini di velocità di computazione.
    virtual void define_matrix(const MatrixType& matrix) {
        if (getProcessRank() == 0) {
            fdapde_assert(matrix.rows() == matrix.cols() && "The matrix must be square");
            fdapde_assert(matrix.rows() > 0 && "The matrix must be non-empty");

            m_size = matrix.rows();
            m_colIndices.clear();
            m_rowIndices.clear();
            m_values.clear();

            m_colIndices.reserve(matrix.nonZeros());
            m_rowIndices.reserve(matrix.nonZeros());
            m_values.reserve(matrix.nonZeros());

            for (int k = 0; k < matrix.outerSize(); ++k) {   // already scales to 1-based indexing
                for (typename MatrixType::InnerIterator it(matrix, k); it; ++it) {
                    m_rowIndices.push_back(it.row() + 1);
                    m_colIndices.push_back(it.col() + 1);
                    m_values.push_back(it.value());
                }
            }

            // defining the problem on the host
            m_mumps.n = m_size;
            m_mumps.nnz = matrix.nonZeros();
            m_mumps.irn = m_rowIndices.data();
            m_mumps.jcn = m_colIndices.data();
            m_mumps.a = m_values.data();
        }

        MPI_Bcast(&m_size, 1, MPI_INT, 0, m_mpiComm);
    }

    // virtual void define_matrix(const MatrixType &matrix) {

    //   MatrixType temp = matrix;
    //   temp.makeCompressed();
    //   m_size = temp.rows();

    //   std::vector<int> loc_row_indices;
    //   std::vector<int> loc_col_indices;
    //   std::vector<Scalar> loc_values;

    //   int loc_start;
    //   int loc_end;

    //   int loc_size = temp.outerSize() / getProcessSize();
    //   int loc_size_remainder = temp.outerSize() % getProcessSize();

    //   if (getProcessRank() < loc_size_remainder) {
    //     loc_start = getProcessRank() * (loc_size + 1);
    //     loc_end = loc_start + loc_size + 1;
    //   } else {
    //     loc_start = getProcessRank() * loc_size + loc_size_remainder;
    //     loc_end = loc_start + loc_size;
    //   }

    //   loc_row_indices.reserve(temp.nonZeros());
    //   loc_col_indices.reserve(temp.nonZeros());
    //   loc_values.reserve(temp.nonZeros());

    //   for (int k = loc_start; k < loc_end; ++k) {
    //     for (typename MatrixType::InnerIterator it(temp, k); it; ++it) {
    //       loc_row_indices.push_back(it.row() + 1);
    //       loc_col_indices.push_back(it.col() + 1);
    //       loc_values.push_back(it.value());
    //     }
    //   }

    //   // Gather local sizes first
    //   int local_size = loc_row_indices.size();
    //   std::vector<int> all_sizes(getProcessSize());
    //   MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, m_mpiComm);

    //   // Calculate displacements for gathering
    //   std::vector<int> displacements(getProcessSize());
    //   if (getProcessRank() == 0) {
    //     displacements[0] = 0;
    //     for (int i = 1; i < getProcessSize(); ++i) {
    //       displacements[i] = displacements[i - 1] + all_sizes[i - 1];
    //     }

    //     m_rowIndices.clear();
    //     m_colIndices.clear();
    //     m_values.clear();

    //     m_rowIndices.resize(std::accumulate(all_sizes.begin(), all_sizes.end(), 0));
    //     m_colIndices.resize(std::accumulate(all_sizes.begin(), all_sizes.end(), 0));
    //     m_values.resize(std::accumulate(all_sizes.begin(), all_sizes.end(), 0));
    //   }

    //   // Gather all local indices into the global vectors on the root process
    //   MPI_Gatherv(loc_row_indices.data(), local_size, MPI_INT, m_rowIndices.data(), all_sizes.data(),
    //               displacements.data(), MPI_INT, 0, m_mpiComm);

    //   MPI_Gatherv(loc_col_indices.data(), local_size, MPI_INT, m_colIndices.data(), all_sizes.data(),
    //               displacements.data(), MPI_INT, 0, m_mpiComm);

    //   MPI_Gatherv(loc_values.data(), local_size, MPI_DOUBLE, m_values.data(), all_sizes.data(), displacements.data(),
    //               MPI_DOUBLE, 0, m_mpiComm);

    //   if (getProcessRank() == 0) {
    //     m_mumps.n = m_size;
    //     m_mumps.nnz = matrix.nonZeros();
    //     m_mumps.irn = m_rowIndices.data();
    //     m_mumps.jcn = m_colIndices.data();
    //     m_mumps.a = m_values.data();
    //   }
    // }

    void mumps_execute(const int job) const {
        if (m_mumps.job == -2 && job != -1) {
            if (getProcessRank() == 0) {
                std::cerr << "MUMPS is already finalized. You cannot execute any job." << std::endl;
            }
            exit(1);
        }
        m_mumps.job = job;
        dmumps_c(&m_mumps);

        if (job == -1) {
            if (getProcessRank() == 0) {
                m_mumps.icntl[0] = (m_verbose) ? 6 : -1;   // 6: verbose output, -1: no output
                m_mumps.icntl[1] = (m_verbose) ? 6 : -1;   // 6: verbose output, -1: no output
                m_mumps.icntl[2] = (m_verbose) ? 6 : -1;   // 6: verbose output, -1: no output
                m_mumps.icntl[3] = (m_verbose) ? 4 : 0;    // 4: verbose output, 0: no output
            } else {
                m_mumps.icntl[0] = -1;
                m_mumps.icntl[1] = -1;
                m_mumps.icntl[2] = -1;
                m_mumps.icntl[3] = 0;
            }

            m_mumps.icntl[32] =
              (m_computeDeterminant) ? 1 : 0;   // 1: compute determinant, 0: do not compute determinant
        }

        errorCheck();
    }

    void analyzePattern_impl() {
        m_info = InvalidInput;
        mumps_execute(1);   // 1: analyze
        m_info = Success;
        m_isInitialized = true;
        m_analysisIsOk = true;
        m_factorizationIsOk = false;
    }

    void factorize_impl() {
        fdapde_assert(m_analysisIsOk && "The matrix must be analyzed with analyzePattern() before calling factorize()");
        m_info = NumericalIssue;
        mumps_execute(2);   // 2: factorize
        m_info = Success;
        m_factorizationIsOk = true;
    }

    void compute_impl() {
        analyzePattern_impl();
        factorize_impl();
    }

    void errorCheck() const {
        if (mumpsInfog()[0] < 0) {
            if (!m_verbose && getProcessRank() == 0) {
                std::string job_name;
                switch (m_mumps.job) {
                case -2:
                    job_name = "Finalization";
                    break;
                case -1:
                    job_name = "Initialization";
                    break;
                case 1:
                    job_name = "Analysis";
                    break;
                case 2:
                    job_name = "Factorization";
                    break;
                case 3:
                    job_name = "Solve";
                    break;
                default:
                    job_name = "Unknown";
                }
                std::cerr << "Error occured during " + job_name + " phase." << std::endl;
                std::cerr << "MUMPS error mumpsInfog()[0]: " << mumpsInfog()[0] << std::endl;
                std::cerr << "MUMPS error mumpsInfog()[1]: " << mumpsInfog()[1] << std::endl;
            }
            exit(1);
        }
    }
   protected:
    mutable DMUMPS_STRUC_C m_mumps;

    // matrix data
    Index m_size;   // only matrix related member defined on all processes
    std::vector<Scalar> m_values;
    std::vector<int> m_colIndices;
    std::vector<int> m_rowIndices;

    // mpi
    int m_mpiRank;
    int m_mpiSize;
    MPI_Comm m_mpiComm;

    // flags
    bool m_verbose;
    bool m_computeDeterminant;

    // computation flags
    bool m_analysisIsOk;
    bool m_factorizationIsOk;
    bool m_mumpsFinalized;

    mutable ComputationInfo m_info;

    // parameter wrappers
    MumpsIcntlArray m_mumpsIcntl;
    MumpsCntlArray m_mumpsCntl;
    MumpsInfoArray m_mumpsInfo;
    MumpsInfoArray m_mumpsInfog;
    MumpsRinfoArray m_mumpsRinfo;
    MumpsRinfoArray m_mumpsRinfog;
};

// MUMPS LU SOLVER
template <isEigenSparseMatrix MatrixType> class MumpsLU : public MumpsBase<MumpsLU<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsLU<MatrixType>>;
   public:
    MumpsLU() : MumpsLU(MPI_COMM_WORLD, 0) { }
    explicit MumpsLU(MPI_Comm comm) : MumpsLU(comm, 0) { }
    explicit MumpsLU(unsigned int flags) : MumpsLU(MPI_COMM_WORLD, flags) { }

    MumpsLU(MPI_Comm comm, unsigned int flags) : Base(comm, flags) {
        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // -1: initialization
    }

    explicit MumpsLU(const MatrixType& matrix) : MumpsLU(matrix, MPI_COMM_WORLD, 0) { }
    MumpsLU(const MatrixType& matrix, MPI_Comm comm) : MumpsLU(matrix, comm, 0) { }
    MumpsLU(const MatrixType& matrix, unsigned int flags) : MumpsLU(matrix, MPI_COMM_WORLD, flags) { }

    MumpsLU(const MatrixType& matrix, MPI_Comm comm, unsigned int flags) : MumpsLU(comm, flags) {
        Base::compute(matrix);
    }

    ~MumpsLU() override {
        if (!Base::m_mumpsFinalized) { Base::mumpsFinalize(); }
    }
};

// MUMPS LDLT SOLVER
template <isEigenSparseMatrix MatrixType, int Options> class MumpsLDLT : public MumpsBase<MumpsLDLT<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsLDLT<MatrixType>>;
   public:
    MumpsLDLT() : MumpsLDLT(MPI_COMM_WORLD, 0) { }
    explicit MumpsLDLT(MPI_Comm comm) : MumpsLDLT(comm, 0) { }
    explicit MumpsLDLT(unsigned int flags) : MumpsLDLT(MPI_COMM_WORLD, flags) { }

    MumpsLDLT(MPI_Comm comm, unsigned int flags) : Base(comm, flags) {
        Base::m_mumps.sym = 1;     // symmetric and positive definite
        Base::mumps_execute(-1);   // -1: initialization
    }

    explicit MumpsLDLT(const MatrixType& matrix) : MumpsLDLT(matrix, MPI_COMM_WORLD, 0) { }
    MumpsLDLT(const MatrixType& matrix, MPI_Comm comm) : MumpsLDLT(matrix, comm, 0) { }
    MumpsLDLT(const MatrixType& matrix, unsigned int flags) : MumpsLDLT(matrix, MPI_COMM_WORLD, flags) { }

    MumpsLDLT(const MatrixType& matrix, MPI_Comm comm, unsigned int flags) : MumpsLDLT(comm, flags) {
        Base::compute(matrix);
    }

    ~MumpsLDLT() override {
        if (!Base::m_mumpsFinalized) { Base::mumpsFinalize(); }
    }
   protected:
    void define_matrix(const MatrixType& matrix) override {
        Base::define_matrix(matrix.template triangularView<Options>());
    }
};

// MUMPS BLR SOLVER

// USAGE: flags = [UFSC | UCFS] | [uncompressed_CB | compressed_CB] ---- (flags can be omitted and default will be used)

// estimated compression rate of LU factors
// mumpsIcntl()[37] = ...
// between 0 and 1000 (default: 600)
// mumpsIcntl()[37]/10 is a percentage representing the typical compression of the compressed factors factor matrices in
// BLR fronts

// estimated compression rate of contribution blocks
// mumpsIcntl()[38] = ...
// between 0 and 1000 (default: 500)
// mumpsIcntl()[38]/10 is a percentage representing the typical compression of the compressed CB CB in BLR fronts

template <isEigenSparseMatrix MatrixType> class MumpsBLR : public MumpsBase<MumpsBLR<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsBLR<MatrixType>>;
   public:
    MumpsBLR() : MumpsBLR(MPI_COMM_WORLD, 0, 0) { }
    explicit MumpsBLR(MPI_Comm comm) : MumpsBLR(comm, 0, 0) { }
    explicit MumpsBLR(unsigned int flags) : MumpsBLR(MPI_COMM_WORLD, flags, 0) { }
    explicit MumpsBLR(double dropping_parameter) : MumpsBLR(MPI_COMM_WORLD, 0, dropping_parameter) { }
    MumpsBLR(MPI_Comm comm, unsigned int flags) : MumpsBLR(comm, flags, 0) { }
    MumpsBLR(unsigned int flags, double dropping_parameter) : MumpsBLR(MPI_COMM_WORLD, flags, dropping_parameter) { }
    MumpsBLR(MPI_Comm comm, double dropping_parameter) : MumpsBLR(comm, 0, dropping_parameter) { }

    MumpsBLR(MPI_Comm comm, unsigned int flags, double dropping_parameter) : Base(comm, flags) {
        fdapde_assert(!(flags & UFSC && flags & UCFS) && "UFSC and UCFS cannot be set at the same time");
        fdapde_assert(dropping_parameter >= 0 && "Dropping parameter must be non-negative");

        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // -1: initialization

        // activate the BLR feature & set the BLR parameters
        Base::mumpsIcntl()[34] = 1;   // 1: automatic choice of the BLR memory management strategy
        Base::mumpsIcntl()[35] =
          (flags & UCFS) ? 1 : 0;   // set mumpsIcntl()[35] = 1 if UCFS is set, otherwise it remains 0 (default -> UFSC)
        Base::mumpsIcntl()[36] = (flags & Compressed) ? 1 : 0;   // set mumpsIcntl()[36] = 1 if compressed_CB is set,
                                                                 // otherwise it remains 0 (default -> uncompressed_CB)
        Base::mumpsCntl()[6] = dropping_parameter;               // set the dropping parameter
    }

    explicit MumpsBLR(const MatrixType& matrix) : MumpsBLR(matrix, MPI_COMM_WORLD, 0, 0) { }
    MumpsBLR(const MatrixType& matrix, MPI_Comm comm) : MumpsBLR(matrix, comm, 0, 0) { }
    MumpsBLR(const MatrixType& matrix, unsigned int flags) : MumpsBLR(matrix, MPI_COMM_WORLD, flags, 0) { }
    MumpsBLR(const MatrixType& matrix, double dropping_parameter) :
        MumpsBLR(matrix, MPI_COMM_WORLD, 0, dropping_parameter) { }
    MumpsBLR(const MatrixType& matrix, MPI_Comm comm, unsigned int flags) : MumpsBLR(matrix, comm, flags, 0) { }
    MumpsBLR(const MatrixType& matrix, MPI_Comm comm, double dropping_parameter) :
        MumpsBLR(matrix, comm, 0, dropping_parameter) { }
    MumpsBLR(const MatrixType& matrix, unsigned int flags, double dropping_parameter) :
        MumpsBLR(matrix, MPI_COMM_WORLD, flags, dropping_parameter) { }

    MumpsBLR(const MatrixType& matrix, MPI_Comm comm, unsigned int flags, double dropping_parameter) :
        MumpsBLR(comm, flags, dropping_parameter) {
        Base::compute(matrix);
    }

    ~MumpsBLR() override {
        if (!Base::m_mumpsFinalized) { Base::mumpsFinalize(); }
    }
};

// MUMPS RANK REVEALING SOLVER
template <isEigenSparseMatrix MatrixType> class MumpsRR : public MumpsBase<MumpsRR<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsRR<MatrixType>>;
   public:
    using Scalar = typename Base::Scalar;

    using Base::solve;

    MumpsRR() : MumpsRR(MPI_COMM_WORLD, 0) { }
    explicit MumpsRR(MPI_Comm comm) : MumpsRR(comm, 0) { }
    explicit MumpsRR(unsigned int flags) : MumpsRR(MPI_COMM_WORLD, flags) { }

    MumpsRR(MPI_Comm comm, unsigned int flags) : Base(comm, flags) {
        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // -1: initialization

        Base::mumpsIcntl()[23] = 1;   // 1: null pivot detection
        Base::mumpsIcntl()[55] = 1;   // 1: perform rank revealing factorization
    }

    explicit MumpsRR(const MatrixType& matrix) : MumpsRR(matrix, MPI_COMM_WORLD, 0) { }
    MumpsRR(const MatrixType& matrix, MPI_Comm comm) : MumpsRR(matrix, comm, 0) { }
    MumpsRR(const MatrixType& matrix, unsigned int flags) : MumpsRR(matrix, MPI_COMM_WORLD, flags) { }

    MumpsRR(const MatrixType& matrix, MPI_Comm comm, unsigned int flags) : MumpsRR(comm, flags) {
        Base::compute(matrix);
    }

    ~MumpsRR() override {
        if (!Base::m_mumpsFinalized) { Base::mumpsFinalize(); }
    }

    int nullSpaceSize() {
        fdapde_assert(
          Base::m_factorizationIsOk && "The matrix must be factorized with factorize() before calling nullSpaceSize()");
        return Base::mumpsInfog()[27];
    }

    int rank() {
        fdapde_assert(
          Base::m_factorizationIsOk && "The matrix must be factorized with factorize() before calling rank()");
        return Base::m_size - Base::mumpsInfog()[27];
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> nullSpaceBasis() {
        fdapde_assert(
          Base::m_factorizationIsOk &&
          "The matrix must be factorized with factorize() before calling nullSpaceBasis()");

        int null_space_size_ = Base::mumpsInfog()[27];

        if (null_space_size_ == 0) { return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(Base::m_size, 0); }

        // allocate memory for the null space basis
        std::vector<Scalar> buff;
        buff.resize(Base::m_size * null_space_size_);

        if (Base::getProcessRank() == 0) {
            Base::m_mumps.nrhs = null_space_size_;
            Base::m_mumps.lrhs = Base::m_size;
            Base::m_mumps.rhs = buff.data();
        }

        Base::mumpsIcntl()[24] = -1;   // -1: perform a null space basis computation step
        Base::mumps_execute(3);        // 3 : solve
        Base::mumpsIcntl()[24] = 0;    // reset mumpsIcntl()[24] to 0 -> a normal solve can be called

        MPI_Bcast(buff.data(), buff.size(), MPI_DOUBLE, 0, Base::m_mpiComm);

        // --> matrix columns are the null space basis vectors
        return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(buff.data(), Base::m_size, null_space_size_);
    }
};

// CLASS FOR COMPUTING THE SHUR COMPLEMENT
template <isEigenSparseMatrix MatrixType> class MumpsSchur : public MumpsBase<MumpsSchur<MatrixType>> {
   protected:
    using Base = MumpsBase<MumpsSchur<MatrixType>>;
   public:
    using Scalar = typename Base::Scalar;

    MumpsSchur() : MumpsSchur(MPI_COMM_WORLD, 0) { }
    explicit MumpsSchur(MPI_Comm comm) : MumpsSchur(comm, 0) { }
    explicit MumpsSchur(unsigned int flags) : MumpsSchur(MPI_COMM_WORLD, flags) { }

    MumpsSchur(MPI_Comm comm, unsigned int flags) : Base(comm, flags), m_schurSizeSet(false) {
        Base::m_mumps.sym = 0;
        Base::mumps_execute(-1);   // -1: initialization

        Base::mumpsIcntl()[18] = 3;   // 3: distributed by columns

        // changing parameters to centralize it according to Mumps documentation
        Base::m_mumps.nprow = 1;
        Base::m_mumps.npcol = 1;
        Base::m_mumps.mblock = 100;
        Base::m_mumps.nblock = 100;
    }

    MumpsSchur(const MatrixType& matrix, int schur_size) : MumpsSchur(matrix, schur_size, MPI_COMM_WORLD, 0) { }
    MumpsSchur(const MatrixType& matrix, int schur_size, MPI_Comm comm) : MumpsSchur(matrix, schur_size, comm, 0) { }
    MumpsSchur(const MatrixType& matrix, int schur_size, unsigned int flags) :
        MumpsSchur(matrix, schur_size, MPI_COMM_WORLD, flags) { }
    MumpsSchur(const MatrixType& matrix, int schur_size, MPI_Comm comm, unsigned int flags) : MumpsSchur(comm, flags) {
        setSchurSize(schur_size);
        Base::compute(matrix);
    }

    ~MumpsSchur() override {
        if (!Base::m_mumpsFinalized) {
            Base::mumpsFinalize();   // -2: finalization
        }
    }

    MumpsSchur<MatrixType>& setSchurSize(int schur_size) {
        fdapde_assert(schur_size > 0 && "The Schur complement size must be greater than 0");
        Base::m_mumps.size_schur = schur_size;
        m_schurSizeSet = true;
        return *this;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> complement() {
        fdapde_assert(
          Base::m_factorizationIsOk && "The matrix must be factorized with factorize() before calling complement()");
        return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(
          m_schurBuff.data(), Base::m_mumps.size_schur, Base::m_mumps.size_schur);
    }
   protected:
    // override define_matrix to add assertions and the matrix-dependent Schur parameters
    void define_matrix(const MatrixType& matrix) override {
        fdapde_assert(
          m_schurSizeSet && "The Schur size must be set with setSchurSize() before calling either alanlyzePattern() or "
                            "facorize() or compute()");
        if (Base::getProcessRank() == 0) {
            fdapde_assert(
              Base::m_mumps.size_schur < matrix.rows() &&
              "The Schur complement size must be smaller than the matrix size");
            fdapde_assert(
              Base::m_mumps.size_schur < matrix.cols() &&
              "The Schur complement size must be smaller than the matrix size");
        }
        Base::define_matrix(matrix);

        m_schurIndices.resize(Base::m_mumps.size_schur);
        std::iota(m_schurIndices.begin(), m_schurIndices.end(), Base::m_size - Base::m_mumps.size_schur + 1);
        Base::m_mumps.listvar_schur = m_schurIndices.data();
        m_schurBuff.resize(Base::m_mumps.size_schur * Base::m_mumps.size_schur);

        // I need to define these on the first working processor: PAR=1 -> rank 0, PAR=0 -> rank 1
        if (Base::getProcessRank() == (Base::m_mumps.par + 1) % 2) {
            Base::m_mumps.schur_lld = Base::m_mumps.size_schur;
            Base::m_mumps.schur = m_schurBuff.data();
        }
    }
   protected:
    std::vector<int> m_schurIndices;
    std::vector<Scalar> m_schurBuff;

    bool m_schurSizeSet;
};

}   // namespace mumps

}   // namespace fdapde

#endif   // __MUMPS_H__
