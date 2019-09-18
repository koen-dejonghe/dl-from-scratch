package botkop.module

import botkop.autograd.Variable
import botkop.numsca.Tensor
import botkop.autograd.Function
import botkop.{numsca => ns}


object BatchNorm {
  def apply(d: Int): BatchNorm = {
    val gamma = Variable(ns.ones(1, d))
    val beta = Variable(ns.zeros(1, d))
    BatchNorm(gamma, beta)
  }

  def apply(d: Int, eps: Double, momentum: Double): BatchNorm = {
    val gamma = Variable(ns.ones(1, d))
    val beta = Variable(ns.zeros(1, d))
    BatchNorm(gamma, beta, eps, momentum)
  }
}

case class BatchNorm(gamma: Variable,
                     beta: Variable,
                     eps: Double = 1e-5,
                     momentum: Double = 0.9)
  extends Module(Seq(gamma, beta)) {
  val runningMean: Tensor = ns.zerosLike(gamma.data)
  val runningVar: Tensor = ns.zerosLike(gamma.data)

  override def forward(x: Variable): Variable =
    BatchNormFunction(x,
      eps,
      momentum,
      runningMean,
      runningVar,
      gamma,
      beta,
      this.isTraining)
      .forward()
}

case class BatchNormFunction(x: Variable,
                             eps: Double,
                             momentum: Double,
                             runningMean: Tensor,
                             runningVar: Tensor,
                             gamma: Variable,
                             beta: Variable,
                             inTrainingMode: Boolean)
  extends Function {

  val List(n, d) = x.shape

  // all below variables are needed in training phase only
  // making them lazy, so they don't get evaluated in test phase

  // compute per-dimension mean and std deviation
  lazy val mean: Tensor = ns.mean(x.data, axis = 0)
  lazy val variance: Tensor = ns.variance(x.data, axis = 0)
  // normalize and zero-center (explicit for caching purposes)
  lazy val xMu: Tensor = x.data - mean
  lazy val invVar: Tensor = 1.0 / ns.sqrt(variance + eps)
  lazy val xHat: Tensor = xMu * invVar

  override def forward(): Variable =
    if (inTrainingMode) {
      runningMean := (momentum * runningMean) + ((1.0 - momentum) * mean)
      runningVar := (momentum * runningVar) + ((1.0 - momentum) * variance)

      // squash
      val out = (xHat * gamma.data) + beta.data
      Variable(out, Some(this))
    } else {
      val out = ((x.data - runningMean) / ns.sqrt(runningVar + eps)) * gamma.data + beta.data
      Variable(out) // no need to backprop when in test mode
    }

  override def backward(dOut: Tensor): Unit = {
    beta.g := ns.sum(dOut, axis = 0)
    gamma.g := ns.sum(xHat * dOut, axis = 0)

    // intermediate partial derivatives
    val dxHat = dOut * gamma.data
    val dx = (invVar / n) * ((dxHat * n) - ns.sum(dxHat, axis = 0) -
      (xHat * ns.sum(dxHat * xHat, axis = 0)))
    x.backward(dx)
  }
}
