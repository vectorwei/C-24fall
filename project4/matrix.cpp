#include <iostream>
#include <memory>
using namespace std;

template<typename T>

class Matrix {
    private:
        size_t numRows, numCols;
        shared_ptr<T> elements;
        T* originalElements;
        bool isSubMatrix;

    public:

        Matrix(size_t numRows, size_t numCols) 
            : numRows(numRows), numCols(numCols), isSubMatrix(false), originalElements(nullptr) {
            elements = shared_ptr<T>(new T[numRows * numCols]);
        }


        Matrix(const Matrix& other) 
            : numRows(other.numRows), numCols(other.numCols), elements(other.elements) {}


        Matrix(size_t rowOffset, size_t colOffset, size_t numRows, size_t numCols, Matrix& parent)
            : numRows(numRows), numCols(numCols), isSubMatrix(true), originalElements(parent.elements.get()) {
            elements = shared_ptr<T>(parent.elements, originalElements + (rowOffset * parent.numCols + colOffset));
        }

        ~Matrix() {
            if (!isSubMatrix && elements) {
                elements.reset();
            }
        }

        T& operator()(size_t row, size_t col) {
            return elements.get()[row * numCols + col];
        }

        const T& operator()(size_t row, size_t col) const {
            return elements.get()[row * numCols + col];
        }

        Matrix& operator=(const Matrix& other) {
            if (this != &other) {
                numRows = other.numRows;
                numCols = other.numCols;
                isSubMatrix = other.isSubMatrix;
                originalElements = other.originalElements;
                if (!isSubMatrix) {
                    elements = shared_ptr<T>(new T[numRows * numCols]);
                    std::copy(other.elements.get(), other.elements.get() + numRows * numCols, elements.get());
                } else {
                    elements = other.elements;
                }
            }
            return *this;
        }

        Matrix operator+(const Matrix& other) const {
            if (numRows != other.numRows || numCols != other.numCols) {
                throw std::invalid_argument("矩阵维度必须匹配以进行加法操作。");
            }
            Matrix result(numRows, numCols);
            for (size_t i = 0; i < numRows * numCols; i++) {
                result.elements.get()[i] = elements.get()[i] + other.elements.get()[i];
            }
            return result;
        }


        Matrix operator-(const Matrix& other) const {
            if (numRows != other.numRows || numCols != other.numCols) {
                throw std::invalid_argument("矩阵维度必须匹配以进行减法操作。");
            }
            Matrix result(numRows, numCols);
            for (size_t i = 0; i < numRows * numCols; ++i) {
                result.elements.get()[i] = this->elements.get()[i] - other.elements.get()[i];
            }
            return result;
        }

        Matrix operator*(const Matrix& other) const {
            if (numRows != other.numRows || numCols != other.numCols) {
                throw std::invalid_argument("矩阵维度必须匹配以进行元素逐一相乘操作。");
            }
            Matrix result(numRows, numCols);
            for (size_t i = 0; i < numRows * numCols; ++i) {
                result.elements.get()[i] = this->elements.get()[i] * other.elements.get()[i];
            }
            return result;
        }
        

        Matrix dot(const Matrix& other) const {
            if (numCols != other.numRows) {
                throw std::invalid_argument("矩阵维度必须匹配以进行矩阵乘法操作。");
            }
            Matrix result(numRows, other.numCols);
            for (size_t i = 0; i < numRows; i++) {
                for (size_t j = 0; j < other.numCols; j++) {
                    result(i, j) = 0;
                    for (size_t k = 0; k < numCols; k++) {
                        result(i, j) += (*this)(i, k) * other(k, j);
                    }
                }
            }
            return result;
        }


        bool operator==(const Matrix& other) const {
            if (numRows != other.numRows || numCols != other.numCols) return false;
            for (size_t i = 0; i < numRows * numCols; i++) {
                if (elements.get()[i] != other.elements.get()[i]) return false;
            }
            return true;
        }

        void print() const {
            for (size_t i = 0; i < numRows; i++) {
                for (size_t j = 0; j < numCols; j++) {
                    std::cout << (*this)(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }

};

int main() {
    Matrix<int> matrix1(3, 3);

    int count = 1;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            matrix1(i, j) = count++;
        }
    }
    cout << "Matrix matrix1:\n";
    matrix1.print();
    Matrix<int> matrix2(3, 3);

    count =0;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            matrix2(i, j) = count++;
        }
    }
    cout << "Matrix matrix2:\n";
    matrix2.print();


    Matrix<int> matrix_add = matrix1 + matrix2; 
    cout << "Matrix matrix_add (matrix + matrix):\n";
    matrix_add.print();


    Matrix<int> matrix_mul= matrix1 * matrix2; 
    cout << "Matrix matrix_mul (matrix * matrix):\n";
    matrix_mul.print();


    Matrix<int> matrix_dot = matrix1.dot(matrix2);
    cout << "Matrix matrix_dot (matrix.dot(matrix)):\n";
    matrix_dot.print();


    Matrix<int> matrix_ROI(1, 1, 2, 2, matrix1);
    cout << "Matrix matrix_ROI (ROI of matrix):\n";
    matrix_ROI.print();
    

    Matrix<int> matrix_assign = matrix1;
    cout << "Matrix matrix_assign (assigned from matrix):\n";
    matrix_assign.print();

    return 0;
}
