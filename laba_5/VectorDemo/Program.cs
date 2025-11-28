using System;
using LinearAlgebra;

class Program
{
    static void Main()
    {
        IMathVector v1 = new MathVector(new[] { 1.0, 2.0, 3.0 });
        IMathVector v2 = new MathVector(new[] { 4.0, 5.0, 6.0 });

        Console.Write("v1: ");
        foreach (double x in v1)
            Console.Write(x + " ");
        Console.WriteLine();

        Console.Write("v2: ");
        foreach (double x in v2)
            Console.Write(x + " ");
        Console.WriteLine();

        IMathVector sum = v1.Sum(v2);
        Console.Write("v1 + v2: ");
        foreach (double x in sum)
            Console.Write(x + " ");
        Console.WriteLine();
    }
}
