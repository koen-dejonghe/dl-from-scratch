package botkop.regression

import botkop.{numsca => ns}
import botkop.data.FashionMnistDataLoader
import scorch.autograd.Variable

object SoftmaxRegression extends App {

  val batchSize = 256

  val trainDl = new FashionMnistDataLoader("train", batchSize)
  println("data set read")

  val numFeatures = trainDl.numFeatures
  val numClasses = 10

  val w = Variable(ns.randn(numFeatures, numClasses) * 0.01)
  val b = Variable(ns.zeros(numClasses))

  def net(x: Variable): Variable = (x dot w) + b

  def loss(yHat: Variable, y: Variable): Variable = scorch.softmaxLoss(yHat, y)

  def sgd(params: Seq[Variable], lr: Double, batchSize: Int): Unit =
    params.foreach { p =>
      p.data -= lr * p.grad.data / batchSize
      p.grad.data := 0.0
    }

  val numEpochs = 5
  val lr = 0.1

  (0 until numEpochs).foreach { epoch =>
    trainDl.foreach {
      case (x, y) =>
        val yh = net(x)
        val l = loss(yh, y)
        println(l)
        l.backward()
        sgd(Seq(w, b), lr, batchSize) // update parameters using their gradient
    }

  }

}
