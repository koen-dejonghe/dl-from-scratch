package botkop.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j

/**
Explains analytical and naive derivative calculation for operators with 1 operand
  */
object AutoGrad1 extends App {

  /* when using naive derivative calculation, we must use double precision to avoid rounding errors */
  Nd4j.setDataType(DataBuffer.Type.DOUBLE)

  // initialize operand
  val x = Variable(ns.randn(3, 3))
  println(s"x = $x\n")

  // define our function
  val f: Tensor => Tensor = ns.tanh

  // execute the function
  val t = x.tanh()
  // back prop
  t.backward()
  // save the gradient
  val dx = x.g.copy()

  println(s"dx = $dx\n")

  // reset the gradient in x, because we will reuse x
  x.g := 0

  // now calculate the gradient in a naive way
  // execute the function
  val nt = x.naive1(f)
  // back prop
  nt.backward()
  // save the gradient
  val ndx = x.g.copy()

  println(s"ndx = $ndx\n")

  // print the relative error
  println(s"error = ${DerivativeUtil.relativeError(dx, ndx)}\n")
}

/**
Explains naive and analytical derivative calculation for operators with 2 operands
  */
object AutoGrad2 extends App {

  /* when using naive derivative calculation, we must use double precision to avoid rounding errors */
  Nd4j.setDataType(DataBuffer.Type.DOUBLE)

  // initialize operands
  val a = Variable(ns.randn(3, 3))
  val b = Variable(ns.randn(3, 3))
  println(s"a = $a,\nb = $b\n")

  // execute the function (element-wise multiplication)
  val r = a * b
  // back prop
  r.backward()

  // save the gradients
  val da = a.g.copy()
  val db = b.g.copy()
  println(s"da = $da,")
  println(s"db = $db\n")

  // reset the gradients, since we will be reusing the operands
  a.g := 0
  b.g := 0

  // the function we want to test
  val f: (Tensor, Tensor) => Tensor = ns.multiply

  val nr = a.naive2(f, b)
  nr.backward()
  val (nda, ndb) = (a.g.copy(), b.g.copy())

  println(s"nda = $nda,")
  println(s"ndb = $ndb\n")

  // print the differences
  println(s"error in a = ${DerivativeUtil.relativeError(da, nda)}\n")
  println(s"error in b = ${DerivativeUtil.relativeError(db, ndb)}\n")
}

object ChainRule1 extends App {
  val x = Variable(-2)
  val y = Variable(5)
  val z = Variable(-4)

  val q = x + y // 3

  val f = q * z // -12

  // print the computation graph
  println(s"computation graph: $f")

  // backprop
  f.backward()

  // print gradients
  println(s"gradient of f = ${f.g}")
  println(s"gradient of q = ${q.g}")
  println(s"gradient of x = ${x.g}")
  println(s"gradient of y = ${y.g}")
  println(s"gradient of z = ${z.g}")
}

case class NaiveDerivative1(f: Tensor => Tensor, x: Variable) extends Function {

  def forward(): Variable = Variable(f(x.data), Some(this))

  def backward(df: Tensor): Unit = {
    val grad = DerivativeUtil.naive1(f, x.data, df)
    x.backward(grad)
  }
}

case class NaiveDerivative2(f: (Tensor, Tensor) => Tensor,
                            a: Variable,
                            b: Variable)
    extends Function {

  def forward(): Variable = Variable(f(a.data, b.data), Some(this))

  def backward(df: Tensor): Unit = {
    val (ga, gb) = DerivativeUtil.naive2(f, a.data, b.data, df)
    a.backward(ga)
    b.backward(gb)
  }
}

object DerivativeUtil {

  /**
    * Compute the gradient of f with respect to x
    * @param f function with 1 operand to compute the derivative of
    * @param x the example tensor to compute the gradient of
    * @param df the derivative of earlier computations, to apply the chain rule
    * @return the gradient of f with respect to x
    */
  def naive1(f: Tensor => Tensor, x: Tensor, df: Tensor): Tensor = {

    /*
    Keep in mind what the derivatives tell you:
    They indicate the rate of change of a function with respect to that variable
    surrounding an infinitesimally small region near a particular point
     */
    val h = 1e-5 // tiny amount (lim h -> 0)

    val grad = ns.zeros(x.shape) // initialize the gradient to 0

    // loop through indices of the tensor
    ns.nditer(x).foreach { ix: Array[Int] => // ix will be of size 1 for a vector, 2 for a matrix, n for a tensor

      val oldVal = x(ix).squeeze() // save current value at this location

      x(ix) := oldVal + h // add tiny amount at this position
      val pos = f(x) // execute function with tiny amount added
      x(ix) := oldVal - h // subtract tiny amount at this position
      val neg = f(x) // execute function with tiny amount subtracted
      x(ix) := oldVal // reset old value at this position

      // calculate the limit for both add and sub
      // and apply chain rule (multiply by df)
      val g: Double = ns.sum((pos - neg) * df) / (2.0 * h)

      grad(ix) := g // store the calculated gradient at this position
    }

    // return the computed gradient
    grad
  }

  /**
    * Computes the gradients of f with respect to a and b
    * @param f the function to evaluate
    * @param a first parameter to the function f
    * @param b second parameter to function f
    * @param df the derivative of earlier computations, to apply the chain rule
    * @return the gradients of f with respect to a and b
    */
  def naive2(f: (Tensor, Tensor) => Tensor,
             a: Tensor,
             b: Tensor,
             df: Tensor): (Tensor, Tensor) = {

    // keep b fixed and evaluate around a
    def fa(t: Tensor): Tensor = f(t, b)
    val da = naive1(fa, a, df)

    // keep a fixed and evaluate around b
    def fb(t: Tensor): Tensor = f(a, t)
    val db = naive1(fb, b, df)

    (da, db)
  }

  /**
    * Compute the relative difference between 2 tensors
    */
  def relativeError(x: Tensor, y: Tensor): Double =
    ns.max(ns.abs(x - y) / ns.maximum(ns.abs(x) + ns.abs(y), 1e-8)).squeeze()

}

