using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace LinearAlgebra
{
    public class MathVector : IMathVector
    {
        private readonly double[] _values;

        public MathVector(IEnumerable<double> source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            _values = source.ToArray();

            if (_values.Length == 0)
                throw new ArgumentException("Вектор не может быть пустым.");
        }

        public int Dimensions => _values.Length;

        public double this[int i]
        {
            get
            {
                if (i < 0 || i >= Dimensions)
                    throw new IndexOutOfRangeException();
                return _values[i];
            }
            set
            {
                if (i < 0 || i >= Dimensions)
                    throw new IndexOutOfRangeException();
                _values[i] = value;
            }
        }

        public double Length => Math.Sqrt(_values.Sum(x => x * x));

        private void CheckDimensions(IMathVector other)
        {
            if (other == null)
                throw new ArgumentNullException(nameof(other));
            if (other.Dimensions != Dimensions)
                throw new ArgumentException("Размерности векторов не совпадают.");
        }

        public IMathVector SumNumber(double number) =>
            new MathVector(_values.Select(v => v + number));

        public IMathVector MultiplyNumber(double number) =>
            new MathVector(_values.Select(v => v * number));

        public IMathVector DivideNumber(double number)
        {
            if (number == 0)
                throw new DivideByZeroException("Деление на ноль.");
            return new MathVector(_values.Select(v => v / number));
        }

        public IMathVector Sum(IMathVector vector)
        {
            CheckDimensions(vector);
            return new MathVector(_values.Zip(vector.Cast<double>(), (a, b) => a + b));
        }

        public IMathVector Multiply(IMathVector vector)
        {
            CheckDimensions(vector);
            return new MathVector(_values.Zip(vector.Cast<double>(), (a, b) => a * b));
        }

        public IMathVector Divide(IMathVector vector)
        {
            CheckDimensions(vector);
            return new MathVector(_values.Zip(vector.Cast<double>(), (a, b) =>
            {
                if (b == 0) throw new DivideByZeroException("Деление на ноль.");
                return a / b;
            }));
        }

        public double ScalarMultiply(IMathVector vector)
        {
            CheckDimensions(vector);
            return _values.Zip(vector.Cast<double>(), (a, b) => a * b).Sum();
        }

        public double CalcDistance(IMathVector vector)
        {
            CheckDimensions(vector);
            double sum = 0;
            for (int i = 0; i < Dimensions; i++)
                sum += Math.Pow(this[i] - vector[i], 2);
            return Math.Sqrt(sum);
        }

        public IEnumerator GetEnumerator() => _values.GetEnumerator();

        public static MathVector operator +(MathVector a, double b) => (MathVector)a.SumNumber(b);
        public static MathVector operator +(double b, MathVector a) => (MathVector)a.SumNumber(b);
        public static MathVector operator +(MathVector a, MathVector b) => (MathVector)a.Sum(b);

        public static MathVector operator -(MathVector a, double b) => (MathVector)a.SumNumber(-b);
        public static MathVector operator -(MathVector a, MathVector b) =>
            (MathVector)a.Sum(b.MultiplyNumber(-1));

        public static MathVector operator *(MathVector a, double b) => (MathVector)a.MultiplyNumber(b);
        public static MathVector operator *(double b, MathVector a) => (MathVector)a.MultiplyNumber(b);
        public static MathVector operator *(MathVector a, MathVector b) => (MathVector)a.Multiply(b);

        public static MathVector operator /(MathVector a, double b) => (MathVector)a.DivideNumber(b);
        public static MathVector operator /(MathVector a, MathVector b) => (MathVector)a.Divide(b);

        public static double operator %(MathVector a, MathVector b) => a.ScalarMultiply(b);

        public override string ToString()
        {
            return string.Join(", ", _values);
        }
    }
}
