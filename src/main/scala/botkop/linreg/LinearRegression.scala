package botkop.linreg

import botkop.{numsca => ns}
import ns._

object LinearRegression extends App {

  val numInputs = 2
  val numExamples = 1000

  val trueW = Tensor(2, -3.4)
  val trueB = Tensor(4.2)

  val features = ns.randn(numExamples, numInputs)
  val labels = (features dot trueW.T) + trueB
  val sigma = ns.randn(labels.shape) * 0.01
  labels += sigma

  println(labels)

}
