package botkop.autograd

import botkop.{numsca => ns}
import botkop.numsca.Tensor

/**
  * Wrapper around a tensor.
  * Keeps track of the computation graph by storing the originating function of this variable, if any.
  * The function will be None if the variable is the result of simple wrapping around a tensor.
  * If the variable is the result of a computation, then f will be set to the function behind this computation.
  * For example, f will be None if x = Variable(ns.zeros(3, 3,)), but f will be Mul if x = a * b,
  * where a and b are also Variables.
  *
  * @param data the tensor
  * @param f function that produced this variable, if any.
  */
case class Variable(data: Tensor, f: Option[Function] = None) {

  def shape: List[Int] = data.shape.toList

  /**
    * the local gradient
    */
  lazy val g: Tensor = ns.zerosLike(data)

  /**
    * Accumulates the incoming gradient in the local gradient.
    * Pushes the incoming gradient back through the network,
    * by means of the originating function, if any.
    * @param gradOutput the gradient that is being pushed back through the network
    */
  def backward(gradOutput: Tensor = ns.ones(data.shape)): Unit = {

    // gradients may have been broadcasted
    // squash the dimensions to fit the original shape
    val ug = ubc(gradOutput)

    // Gradients add up at forks.
    // If the forward expression involves the variables x,y multiple times,
    // then when we perform backpropagation we must be careful to use += instead of =
    // to accumulate the gradient on these variables (otherwise we would overwrite it).
    // This follows the multivariable chain rule in Calculus,
    // which states that if a variable branches out to different parts of the circuit,
    // then the gradients that flow back to it will add.
    // http://cs231n.github.io/optimization-2/#staged
    g += ug

    // backprop thru function that generated this variable, if any
    for (gf <- f) gf.backward(ug)
  }

  /*
  unbroadcast
   */
  def ubc(t: Tensor): Tensor =
    if (t.shape.sameElements(data.shape))
      t
    else
      data.shape.zip(t.shape).zipWithIndex.foldLeft(t) {
        case (d: Tensor, ((oi, ni), i)) =>
          if (oi == ni)
            d
          else if (oi == 1)
            ns.sum(d, axis = i)
          else
            throw new Exception(
              s"unable to reduce broadcasted shape ${t.shape.toList} as ${data.shape.toList}")
      }

  /**
    *  Functions with 1 operand
    */
  def mean(): Variable = Mean(this).forward()
  def tanh(): Variable = Tanh(this).forward()
  def relu(): Variable = Threshold(this, 0.0).forward()

  /**
    * Functions with 2 operands
    */
  def +(other: Variable): Variable = Add(this, other).forward()
  def -(other: Variable): Variable = Sub(this, other).forward()
  def *(other: Variable): Variable = Mul(this, other).forward()
  def /(other: Variable): Variable = Div(this, other).forward()
  def dot(other: Variable): Variable = Dot(this, other).forward()

  def +(d: Double): Variable = AddScalar(this, d).forward()
  def -(d: Double): Variable = SubScalar(this, d).forward()
  def *(d: Double): Variable = MulScalar(this, d).forward()
  def /(d: Double): Variable = DivScalar(this, d).forward()
  def **(d: Double): Variable = PowScalar(this, d).forward()

}

object Variable {
  def apply(ds: Double*): Variable = Variable(Tensor(ds:_*))
}

