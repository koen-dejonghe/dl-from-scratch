package botkop.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor

object AutoGrad extends App {


}

case class Variable(data: Tensor, f: Option[Function] = None) {

  lazy val g: Tensor = ns.zerosLike(data)

  def backward(): Unit = {
    backward(ns.ones(data.shape))
  }

  def backward(gradOutput: Tensor): Unit = {
    g += gradOutput
    for (gf <- f) gf.backward(gradOutput)
  }

  def +(other: Variable): Variable = Add(this, other).forward()
  def *(other: Variable): Variable = Mul(this, other).forward()
  def dot(other: Variable): Variable = Dot(this, other).forward()
}

trait Function {
  def forward(): Variable
  def backward(g: Tensor): Unit
}

case class Add(v1: Variable, v2: Variable) extends Function {
  def forward(): Variable = Variable(v1.data + v2.data, f = Some(this))
  def backward(g: Tensor): Unit = {
    v1.backward(g)
    v2.backward(g)
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

