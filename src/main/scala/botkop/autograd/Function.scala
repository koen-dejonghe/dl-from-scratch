package botkop.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor


trait Function {
  def forward(): Variable
  def backward(g: Tensor): Unit
}

/*
Functions with 1 operand
 */

case class Tanh(x: Variable) extends Function {
  val tanh: Tensor = ns.tanh(x.data)
  override def forward(): Variable = Variable(tanh, Some(this))
  override def backward(g: Tensor): Unit = {
    val dx = (1 - ns.square(tanh)) * g
    x.backward(dx)
  }
}

case class Exp(v: Variable) extends Function {
  val cache: Tensor = ns.exp(v.data)
  def forward() = Variable(data = cache, Some(this))
  def backward(gradOutput: Tensor): Unit = {
    v.backward(gradOutput * cache)
  }
}

/*
Functions with 2 operands
 */

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

case class Max(x: Variable, y: Variable) extends Function {
  def forward(): Variable = {
    val max: Tensor = ns.maximum(x.data, y.data)
    Variable(max, Some(this))
  }
  override def backward(gradOutput: Tensor): Unit = {
    x.backward((x.data >= y.data) * gradOutput)
    y.backward((x.data <= y.data) * gradOutput)
  }
}

case class PowConstant(v: Variable, d: Double) extends Function {
  override def forward(): Variable = Variable(ns.power(v.data, d), Some(this))
  override def backward(gradOutput: Tensor): Unit = {
    val dv = d * ns.power(v.data, d - 1) * gradOutput
    v.backward(dv)
  }
}

