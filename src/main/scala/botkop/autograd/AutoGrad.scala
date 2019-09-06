package botkop.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j

/*
Explains naive and analytical derivative calculation for operators with 1 operand
 */
object AutoGrad1 extends App {

  /* when using naive derivative calculation, we must use double precision to avoid underflow */
  Nd4j.setDataType(DataBuffer.Type.DOUBLE)

  val x = Variable(ns.randn(3, 3))
  println(s"x = $x\n")

  // execute the function
  val t = x.tanh()
  // back prop
  t.backward()
  // save the gradient
  val dx = x.g.copy()

  println(s"dx = $dx\n")

  // reset the gradient in x
  x.g := 0

  // now calculate the gradient in a naive way
  // define our function
  val f: Tensor => Tensor = ns.tanh
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

/*
Explains naive and analytical derivative calculation for operators with 2 operands
 */
object AutoGrad2 extends App {

  /* when using naive derivative calculation, we must use double precision to avoid underflow */
  Nd4j.setDataType(DataBuffer.Type.DOUBLE)

  val a = Variable(ns.randn(3, 3))
  val b = Variable(ns.randn(3, 3))
  println(s"a = $a,\nb = $b\n")

  val r = a * b
  r.backward()

  val da = a.g.copy()
  val db = b.g.copy()
  println(s"da = $da,")
  println(s"db = $db\n")

  a.g := 0
  b.g := 0

  val f: (Tensor, Tensor) => Tensor = ns.multiply

  val nr = a.naive2(f, b)
  nr.backward()
  val (nda, ndb) = (a.g.copy(), b.g.copy())

  println(s"nda = $nda,")
  println(s"ndb = $ndb\n")

  // print the relative error
  println(s"error in a = ${DerivativeUtil.relativeError(da, nda)}\n")
  println(s"error in b = ${DerivativeUtil.relativeError(db, ndb)}\n")
}

object ChainRule extends App {


}

case class Variable(data: Tensor, f: Option[Function] = None) {

  lazy val g: Tensor = ns.zerosLike(data)

  def backward(gradOutput: Tensor = ns.ones(data.shape)): Unit = {
    g += gradOutput
    for (gf <- f) gf.backward(gradOutput)
  }

  def tanh(): Variable = Tanh(this).forward()

  def +(other: Variable): Variable = Add(this, other).forward()
  def *(other: Variable): Variable = Mul(this, other).forward()
  def dot(other: Variable): Variable = Dot(this, other).forward()

  def naive1(f: Tensor => Tensor): Variable =
    NaiveDerivative1(f, this).forward()

  def naive2(f: (Tensor, Tensor) => Tensor, other: Variable): Variable =
    NaiveDerivative2(f, this, other).forward()

}

trait Function {
  def forward(): Variable
  def backward(g: Tensor): Unit
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

  def naive1(f: Tensor => Tensor, x: Tensor, df: Tensor): Tensor = {

    /*
    Keep in mind what the derivatives tell you:
    They indicate the rate of change of a function with respect to that variable
    surrounding an infinitesimally small region near a particular point
     */
    val h = 1e-5 // tiny amount (lim h -> 0)

    val grad = ns.zeros(x.shape) // initialize is set to 0

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

  def naive2(f: (Tensor, Tensor) => Tensor,
             a: Tensor,
             b: Tensor,
             df: Tensor): (Tensor, Tensor) = {

    def fa(t: Tensor): Tensor = f(t, b)
    def fb(t: Tensor): Tensor = f(a, t)

    val da = naive1(fa, a, df)
    val db = naive1(fb, b, df)

    (da, db)
  }

  def relativeError(x: Tensor, y: Tensor): Double =
    ns.max(ns.abs(x - y) / ns.maximum(ns.abs(x) + ns.abs(y), 1e-8)).squeeze()

}

case class Add(v1: Variable, v2: Variable) extends Function {
  def forward(): Variable = Variable(v1.data + v2.data, f = Some(this))
  def backward(g: Tensor): Unit = {
    v1.backward(g * 1.0)
    v2.backward(g * 1.0)
  }
}

case class Mul(v1: Variable, v2: Variable) extends Function {
  override def forward(): Variable = Variable(v1.data * v2.data, f = Some(this))
  override def backward(g: Tensor): Unit = {
    val dv2 = v2.data * g
    val dv1 = v1.data * g
    v1.backward(dv2)
    v2.backward(dv1)
  }
}

case class Dot(v1: Variable, v2: Variable) extends Function {
  val w: Tensor = v1.data
  val x: Tensor = v2.data

  override def forward(): Variable = Variable(w dot x, f = Some(this))
  override def backward(g: Tensor): Unit = {
    val dw = g dot x.T
    val dx = w.T dot g
    v1.backward(dw)
    v2.backward(dx)
  }
}

case class Tanh(v: Variable) extends Function {
  val cache: Tensor = ns.tanh(v.data)
  override def forward(): Variable = Variable(cache, Some(this))
  override def backward(g: Tensor): Unit =
    v.backward((1 - ns.square(cache)) * g)
}
