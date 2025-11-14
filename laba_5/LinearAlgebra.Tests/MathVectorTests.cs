using System;
using LinearAlgebra;
using Xunit;

namespace LinearAlgebra.Tests
{
    public class MathVectorTests
    {
        private const double Eps = 1e-9;

        // ---------- Конструктор ----------

        [Fact]
        public void Constructor_NullSource_ThrowsArgumentNullException()
        {
            Assert.Throws<ArgumentNullException>(() => new MathVector(null!));
        }

        [Fact]
        public void Constructor_EmptySource_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => new MathVector(Array.Empty<double>()));
        }

        [Fact]
        public void Constructor_ValidArray_SetsDimensionsCorrectly()
        {
            var v = new MathVector(new[] { 1.0, 2.0, 3.0 });

            Assert.Equal(3, v.Dimensions);
        }

        // ---------- Индексатор ----------

        [Fact]
        public void Indexer_GetAndSet_WorksCorrectly()
        {
            var v = new MathVector(new[] { 1.0, 2.0, 3.0 });

            v[1] = 10.0;

            Assert.Equal(1.0, v[0], 9);
            Assert.Equal(10.0, v[1], 9);
            Assert.Equal(3.0, v[2], 9);
        }

        [Fact]
        public void Indexer_NegativeIndex_Throws()
        {
            var v = new MathVector(new[] { 1.0, 2.0 });

            Assert.Throws<IndexOutOfRangeException>(() => v[-1] = 5.0);
        }

        [Fact]
        public void Indexer_IndexEqualDimensions_Throws()
        {
            var v = new MathVector(new[] { 1.0, 2.0 });

            Assert.Throws<IndexOutOfRangeException>(() => v[2] = 5.0);
        }

        // ---------- Length ----------

        [Fact]
        public void Length_CalculatedCorrectly()
        {
            var v = new MathVector(new[] { 3.0, 4.0, 12.0 }); // sqrt(9+16+144) = 13

            var length = v.Length;

            Assert.True(Math.Abs(13.0 - length) < Eps);
        }

        // ---------- Операции с числом ----------

        [Fact]
        public void SumNumber_ReturnsNewVector_AndDoesNotChangeOriginal()
        {
            var original = new MathVector(new[] { 1.0, 2.0, 3.0 });

            var result = (MathVector)original.SumNumber(5.0);

            Assert.Equal(new[] { 6.0, 7.0, 8.0 }, new[] { result[0], result[1], result[2] });
            Assert.Equal(new[] { 1.0, 2.0, 3.0 }, new[] { original[0], original[1], original[2] });
        }

        [Fact]
        public void MultiplyNumber_MultipliesEachComponent()
        {
            var v = new MathVector(new[] { 1.0, -2.0, 3.5 });

            var result = (MathVector)v.MultiplyNumber(2.0);

            Assert.Equal(new[] { 2.0, -4.0, 7.0 }, new[] { result[0], result[1], result[2] });
        }

        [Fact]
        public void DivideNumber_ByZero_Throws()
        {
            var v = new MathVector(new[] { 1.0, 2.0 });

            Assert.Throws<DivideByZeroException>(() => v.DivideNumber(0.0));
        }

        // ---------- Операции с другим вектором ----------

        [Fact]
        public void Sum_DifferentDimensions_Throws()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0 });
            var v2 = new MathVector(new[] { 1.0, 2.0, 3.0 });

            Assert.Throws<ArgumentException>(() => v1.Sum(v2));
        }

        [Fact]
        public void Sum_SameDimensions_AddsComponents()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0, 3.0 });
            var v2 = new MathVector(new[] { 4.0, 5.0, 6.0 });

            var result = (MathVector)v1.Sum(v2);

            Assert.Equal(new[] { 5.0, 7.0, 9.0 }, new[] { result[0], result[1], result[2] });
        }

        [Fact]
        public void Multiply_Componentwise_WorksCorrectly()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0, 3.0 });
            var v2 = new MathVector(new[] { 4.0, 5.0, 6.0 });

            var result = (MathVector)v1.Multiply(v2);

            Assert.Equal(new[] { 4.0, 10.0, 18.0 }, new[] { result[0], result[1], result[2] });
        }

        [Fact]
        public void Divide_Componentwise_ZeroInSecondVector_Throws()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0, 3.0 });
            var v2 = new MathVector(new[] { 1.0, 0.0, 1.0 });

            Assert.Throws<DivideByZeroException>(() => v1.Divide(v2));
        }

        // ---------- ScalarMultiply ----------

        [Fact]
        public void ScalarMultiply_DifferentDimensions_Throws()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0 });
            var v2 = new MathVector(new[] { 1.0, 2.0, 3.0 });

            Assert.Throws<ArgumentException>(() => v1.ScalarMultiply(v2));
        }

        [Fact]
        public void ScalarMultiply_ReturnsCorrectValue()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0, 3.0 });
            var v2 = new MathVector(new[] { 4.0, 5.0, 6.0 });

            var result = v1.ScalarMultiply(v2); // 1*4 + 2*5 + 3*6 = 32

            Assert.Equal(32.0, result, 9);
        }

        // ---------- CalcDistance ----------

        [Fact]
        public void CalcDistance_DifferentDimensions_Throws()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0 });
            var v2 = new MathVector(new[] { 1.0, 2.0, 3.0 });

            Assert.Throws<ArgumentException>(() => v1.CalcDistance(v2));
        }

        [Fact]
        public void CalcDistance_ReturnsCorrectValue()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0, 3.0 });
            var v2 = new MathVector(new[] { 4.0, 6.0, 3.0 });

            var distance = v1.CalcDistance(v2); // sqrt(3^2 + 4^2 + 0^2) = 5

            Assert.True(Math.Abs(5.0 - distance) < Eps);
        }

        // ---------- Операторы ----------

        [Fact]
        public void OperatorPlus_Number_WorksCorrectly()
        {
            var v = new MathVector(new[] { 1.0, 2.0 });

            var result = v + 3.0;

            Assert.Equal(new[] { 4.0, 5.0 }, new[] { result[0], result[1] });
        }

        [Fact]
        public void OperatorPlus_Vector_WorksCorrectly()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0 });
            var v2 = new MathVector(new[] { 3.0, 4.0 });

            var result = v1 + v2;

            Assert.Equal(new[] { 4.0, 6.0 }, new[] { result[0], result[1] });
        }

        [Fact]
        public void OperatorMinus_Vector_WorksCorrectly()
        {
            var v1 = new MathVector(new[] { 5.0, 7.0 });
            var v2 = new MathVector(new[] { 3.0, 4.0 });

            var result = v1 - v2;

            Assert.Equal(new[] { 2.0, 3.0 }, new[] { result[0], result[1] });
        }

        [Fact]
        public void OperatorMultiply_Number_WorksCorrectly()
        {
            var v = new MathVector(new[] { 1.0, 2.0 });

            var result = v * 2.0;

            Assert.Equal(new[] { 2.0, 4.0 }, new[] { result[0], result[1] });
        }

        [Fact]
        public void OperatorMultiply_Vector_WorksCorrectly()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0 });
            var v2 = new MathVector(new[] { 3.0, 4.0 });

            var result = v1 * v2;

            Assert.Equal(new[] { 3.0, 8.0 }, new[] { result[0], result[1] });
        }

        [Fact]
        public void OperatorDivide_Number_WorksCorrectly()
        {
            var v = new MathVector(new[] { 2.0, 4.0 });

            var result = v / 2.0;

            Assert.Equal(new[] { 1.0, 2.0 }, new[] { result[0], result[1] });
        }

        [Fact]
        public void OperatorPercent_ReturnsScalarProduct()
        {
            var v1 = new MathVector(new[] { 1.0, 2.0, 3.0 });
            var v2 = new MathVector(new[] { 4.0, 5.0, 6.0 });

            var result = v1 % v2;

            Assert.Equal(32.0, result, 9);
        }
    }
}
