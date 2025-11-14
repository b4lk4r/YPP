using System;
using LinearAlgebra;

class Program
{
    static void Main()
    {
        IMathVector v1 = new MathVector(new[] { 1.0, 2.0, 3.0 });
        IMathVector v2 = new MathVector(new[] { 4.0, 5.0, 6.0 });

        Console.WriteLine("v1 length: " + v1.Length);
        Console.WriteLine("v1 + 2: " + string.Join(", ", (MathVector)v1.SumNumber(2)));

        Console.WriteLine("v1 + v2: " +
            string.Join(", ", (MathVector)v1.Sum(v2)));

        Console.WriteLine("v1 * v2 (покомпонентно): " +
            string.Join(", ", (MathVector)v1.Multiply(v2)));

        Console.WriteLine("Скалярное произведение: " +
            v1.ScalarMultiply(v2));

        Console.WriteLine("Расстояние: " +
            v1.CalcDistance(v2));
    }
}
