#include <iostream>
#include <memory>
using namespace std;

template<typename T>

class Matrix {
    private:
        size_t rows, cols;
        shared_ptr<T> data;
        T* originalData; 
        bool isROI;

    public:
        // 主构造函数
        Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), isROI(false), originalData(nullptr) {
            data = shared_ptr<T>(new T[rows * cols]);
        }

        // Copy constructor
        Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}
        

        // ROI 构造函数
        Matrix(size_t rowOffset, size_t colOffset, size_t rows, size_t cols, Matrix& parent)
            : rows(rows), cols(cols), isROI(true), originalData(parent.data.get()) {
            data = shared_ptr<T>(parent.data, originalData + (rowOffset * parent.cols + colOffset));
        }

        ~Matrix() {
            if (!isROI && data) {
                data.reset(); 
            }
        }

        // 索引运算符
        T& operator()(size_t row, size_t col) {
            return data.get()[row * cols + col];
        }

        const T& operator()(size_t row, size_t col) const {
            return data.get()[row * cols + col];
        }

        // Copy assignment operator
        Matrix& operator=(const Matrix& other) {
            if (this != &other) {
                rows = other.rows;
                cols = other.cols;
                isROI = other.isROI;
                originalData = other.originalData;
                if (!isROI) {
                    data = shared_ptr<T>(new T[rows * cols]);
                    std::copy(other.data.get(), other.data.get() + rows * cols, data.get());
                } else {
                    data = other.data;
                }
            }
            return *this;
        }

        // addition
        Matrix operator+(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::invalid_argument("Matrices dimensions must match for addition.");
            }
            Matrix result(rows, cols);
            for (size_t i = 0; i < rows * cols; i++) {
                result.data.get()[i] = data.get()[i] + other.data.get()[i];
            }
            return result;
        }

        // subtraction
        Matrix operator-(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::invalid_argument("Matrices dimensions must match for subtraction.");
            }
            Matrix result(rows, cols);
            for (int i = 0; i < rows * cols; ++i) {
                result.data.get()[i] = this->data.get()[i] - other.data.get()[i];
            }
            return result;
        }
        
        // multiplication, element-wise
        Matrix operator*(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) {
                throw std::invalid_argument("Matrices dimensions must match for element-wise multiplication.");
            }
            Matrix result(rows, cols);
            for (int i = 0; i < rows * cols; ++i) {
                result.data.get()[i] = this->data.get()[i] * other.data.get()[i];
            }
            return result;
        }
        
        // matrix multiplication
        Matrix dot(const Matrix& other) const {
            if (cols != other.rows) {
                throw std::invalid_argument("Matrices dimensions must match for multiplication.");
            }
            Matrix result(rows, other.cols);
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < other.cols; j++) {
                    result(i, j) = 0;
                    for (size_t k = 0; k < cols; k++) {
                        result(i, j) += (*this)(i, k) * other(k, j);
                    }
                }
            }
            return result;
        }

        // Equality operator
        bool operator==(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) return false;
            for (size_t i = 0; i < rows * cols; i++) {
                if (data.get()[i] != other.data.get()[i]) return false;
            }
            return true;
        }

        // print
        void print() const {
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    std::cout << (*this)(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }

};

int main() {
    Matrix<int> mat(3, 3);

    int count = 1;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            mat(i, j) = count++;
        }
    }

    cout << "Matrix mat:\n";
    mat.print();

    // addition
    Matrix<int> mat2 = mat + mat; 
    cout << "Matrix mat2 (result of mat + mat):\n";
    mat2.print();

    // matrix element wise multiplication
    Matrix<int> mat3 = mat * mat; 
    cout << "Matrix mat3 (result of mat * mat):\n";
    mat3.print();

    // matrix multiplication
    Matrix<int> mat4 = mat.dot(mat);
    cout << "Matrix mat4 (result of mat.dot(mat)):\n";
    mat4.print();

    // roi test
    Matrix<int> mat5(1, 1, 2, 2, mat);
    cout << "Matrix mat5 (ROI of mat):\n";
    mat5.print();
    
    // matrix assignment
    Matrix<int> mat6 = mat;
    cout << "Matrix mat6 (assigned from mat):\n";
    mat6.print();

    return 0;
}
